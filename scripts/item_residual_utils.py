"""Utilities for scalar item-difficulty residual training.

This module intentionally mirrors the locked Canon V2 prior hierarchy:
global -> benchmark -> benchmark/condition -> subject -> subject/benchmark
-> subject/benchmark/condition. It does not use the older Lever L family
tables because those are not present in data/stage2/priors_v1_locked/.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from lever_l_utils import clean_str, coerce_binary_label, hash_to_unit, mean_log_likelihood, normalize_subject

KEY_SEP = "\x1f"
PRIOR_COLUMNS_5 = (
    "prior_global",
    "prior_benchmark",
    "prior_benchmark_condition",
    "prior_subject",
    "prior_subject_benchmark",
    "prior_subject_category",
)
REQUIRED_LOCKED_KAPPAS = (
    "benchmark",
    "benchmark_condition",
    "subject",
    "subject_benchmark",
    "subject_category",
)
PIPE_SEP = re.compile(r"\s*\|\s*")
WHITESPACE = re.compile(r"\s+")
BENCHMARK_ALIAS_SEP = re.compile(r"[-_\s]+")
CONDITION_EMPTY_ALIASES = {"", "none", "null", "n/a", "na", "-"}


def logit_prob(p: np.ndarray | float) -> np.ndarray | float:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def _clean_text(value: object) -> str:
    return unicodedata.normalize("NFKC", str(value)).strip()


def _normalize_benchmark_basic(benchmark: object) -> str:
    return WHITESPACE.sub(" ", _clean_text(benchmark).lower())


def _benchmark_signature(benchmark: object) -> str:
    return BENCHMARK_ALIAS_SEP.sub("", _normalize_benchmark_basic(benchmark))


def _normalize_condition_basic(condition: object) -> str:
    text = PIPE_SEP.sub("|", _clean_text(condition))
    return WHITESPACE.sub(" ", text)


def _condition_signature(condition: object) -> str:
    basic = _normalize_condition_basic(condition)
    lowered = basic.lower()
    if lowered in CONDITION_EMPTY_ALIASES:
        return "none"
    return lowered


def _unique_alias_map(keys: set[str], signature_fn) -> dict[str, str]:
    buckets: dict[str, list[str]] = {}
    for key in keys:
        buckets.setdefault(signature_fn(key), []).append(key)
    return {
        signature: values[0]
        for signature, values in buckets.items()
        if len(set(values)) == 1
    }


def _split_prior_key(key: str, sep: str, expected_parts: int) -> list[str]:
    parts = str(key).split(sep)
    if len(parts) == expected_parts:
        return parts
    return []


@dataclass(frozen=True)
class RuntimeCanonicalizer:
    """Predict-time benchmark/condition canonicalization mirror."""

    benchmark_keys: frozenset[str]
    condition_keys: frozenset[str]
    benchmark_aliases: Mapping[str, str]
    condition_aliases: Mapping[str, str]

    @classmethod
    def from_runtime_priors(cls, runtime_priors: Mapping[str, Any]) -> "RuntimeCanonicalizer":
        sep = str(runtime_priors.get("key_sep", KEY_SEP))
        benchmark_keys = {str(k) for k in runtime_priors.get("benchmark", {})}
        for key in runtime_priors.get("benchmark_condition", {}):
            parts = _split_prior_key(str(key), sep, 2)
            if parts:
                benchmark_keys.add(parts[0])
        for key in runtime_priors.get("subject_benchmark", {}):
            parts = _split_prior_key(str(key), sep, 2)
            if parts:
                benchmark_keys.add(parts[1])
        for key in runtime_priors.get("subject_category", {}):
            parts = _split_prior_key(str(key), sep, 3)
            if parts:
                benchmark_keys.add(parts[1])

        condition_keys: set[str] = set()
        for key in runtime_priors.get("benchmark_condition", {}):
            parts = _split_prior_key(str(key), sep, 2)
            if parts:
                condition_keys.add(parts[1])
        for key in runtime_priors.get("subject_category", {}):
            parts = _split_prior_key(str(key), sep, 3)
            if parts:
                condition_keys.add(parts[2])

        return cls(
            benchmark_keys=frozenset(benchmark_keys),
            condition_keys=frozenset(condition_keys),
            benchmark_aliases=_unique_alias_map(benchmark_keys, _benchmark_signature),
            condition_aliases=_unique_alias_map(condition_keys, _condition_signature),
        )

    def normalize_benchmark(self, benchmark: object) -> str:
        basic = _normalize_benchmark_basic(benchmark)
        if basic in self.benchmark_keys:
            return basic
        return self.benchmark_aliases.get(_benchmark_signature(basic), basic)

    def normalize_condition(self, condition: object) -> str:
        basic = _normalize_condition_basic(condition)
        if basic in self.condition_keys:
            return basic
        lowered = basic.lower()
        if "none" in self.condition_keys and "" not in self.condition_keys and lowered in CONDITION_EMPTY_ALIASES:
            return "none"
        if "" in self.condition_keys and "none" not in self.condition_keys and lowered in CONDITION_EMPTY_ALIASES:
            return ""
        return self.condition_aliases.get(_condition_signature(basic), basic)


@dataclass
class LockedPriorTables:
    global_p: float
    benchmark: dict[str, float]
    benchmark_condition: dict[str, float]
    subject: dict[str, float]
    subject_benchmark: dict[str, float]
    subject_category: dict[str, float]
    kappas: dict[str, float]
    key_sep: str = KEY_SEP


def join_key(*parts: str, sep: str = KEY_SEP) -> str:
    return sep.join(clean_str(part) for part in parts)


def load_runtime_priors(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def locked_kappas(runtime_priors: Mapping[str, Any]) -> dict[str, float]:
    raw = runtime_priors.get("kappas", {})
    missing = [name for name in REQUIRED_LOCKED_KAPPAS if name not in raw]
    if missing:
        raise KeyError(f"runtime_priors missing locked kappa(s): {missing}")
    return {name: float(raw[name]) for name in REQUIRED_LOCKED_KAPPAS}


def load_joined_canonical(joined: Path, runtime_priors_path: Path):
    import pandas as pd

    runtime_priors = load_runtime_priors(runtime_priors_path)
    canonicalizer = RuntimeCanonicalizer.from_runtime_priors(runtime_priors)
    columns = [
        "subject_id",
        "item_id",
        "subject_content",
        "item_content",
        "benchmark",
        "condition",
        "label",
    ]
    df = pd.read_parquet(joined, columns=columns)
    df["label"] = df["label"].map(coerce_binary_label)
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype("int8")
    for col in ("subject_id", "item_id", "subject_content", "item_content", "benchmark", "condition"):
        df[col] = df[col].map(clean_str)
    df["row_index"] = np.arange(len(df), dtype=np.int64)
    df["subject_key"] = df["subject_content"].map(normalize_subject)
    df["benchmark_key"] = df["benchmark"].map(canonicalizer.normalize_benchmark)
    df["condition_key"] = df["condition"].map(canonicalizer.normalize_condition)
    return df.reset_index(drop=True), runtime_priors, canonicalizer


def _group_counts(df, keys: list[str]):
    grouped = df.groupby(keys, observed=True)["label"].agg(["count", "sum"]).reset_index()
    grouped = grouped.rename(columns={"count": "n", "sum": "k"})
    grouped["n"] = grouped["n"].astype("int64")
    grouped["k"] = grouped["k"].astype("float64")
    return grouped


def _smoothed(k: np.ndarray, n: np.ndarray, parent_p: np.ndarray, kappa: float) -> np.ndarray:
    return (k.astype(np.float64) + float(kappa) * parent_p.astype(np.float64)) / (
        n.astype(np.float64) + float(kappa)
    )


def _safe_parent(parent: Mapping[str, float], key: str, fallback: float) -> float:
    value = parent.get(key, fallback)
    if not math.isfinite(float(value)):
        return fallback
    return float(value)


def fit_locked_priors(df, kappas: Mapping[str, float]) -> LockedPriorTables:
    """Fit the exact 5-level locked prior hierarchy on canonicalized columns."""

    required_cols = {"subject_key", "benchmark_key", "condition_key", "label"}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise KeyError(f"fit_locked_priors missing column(s): {missing_cols}")

    global_p = float(df["label"].mean())

    bench_df = _group_counts(df, ["benchmark_key"])
    bench_df["parent_p"] = global_p
    bench_df["p"] = _smoothed(
        bench_df["k"].to_numpy(),
        bench_df["n"].to_numpy(),
        bench_df["parent_p"].to_numpy(),
        float(kappas["benchmark"]),
    )
    benchmark = {str(r.benchmark_key): float(r.p) for r in bench_df.itertuples(index=False)}

    bc_df = _group_counts(df, ["benchmark_key", "condition_key"])
    bc_df["parent_p"] = [_safe_parent(benchmark, str(bench), global_p) for bench in bc_df["benchmark_key"].tolist()]
    bc_df["p"] = _smoothed(
        bc_df["k"].to_numpy(),
        bc_df["n"].to_numpy(),
        bc_df["parent_p"].to_numpy(),
        float(kappas["benchmark_condition"]),
    )
    benchmark_condition = {
        join_key(str(r.benchmark_key), str(r.condition_key)): float(r.p)
        for r in bc_df.itertuples(index=False)
    }

    subj_df = _group_counts(df, ["subject_key"])
    subj_df["parent_p"] = global_p
    subj_df["p"] = _smoothed(
        subj_df["k"].to_numpy(),
        subj_df["n"].to_numpy(),
        subj_df["parent_p"].to_numpy(),
        float(kappas["subject"]),
    )
    subject = {str(r.subject_key): float(r.p) for r in subj_df.itertuples(index=False)}

    sb_df = _group_counts(df, ["subject_key", "benchmark_key"])
    sb_df["parent_p"] = [_safe_parent(subject, str(subject_key), global_p) for subject_key in sb_df["subject_key"].tolist()]
    sb_df["p"] = _smoothed(
        sb_df["k"].to_numpy(),
        sb_df["n"].to_numpy(),
        sb_df["parent_p"].to_numpy(),
        float(kappas["subject_benchmark"]),
    )
    subject_benchmark = {
        join_key(str(r.subject_key), str(r.benchmark_key)): float(r.p)
        for r in sb_df.itertuples(index=False)
    }

    sc_df = _group_counts(df, ["subject_key", "benchmark_key", "condition_key"])
    sc_df["parent_p"] = [
        _safe_parent(subject_benchmark, join_key(str(subject_key), str(benchmark_key)), _safe_parent(subject, str(subject_key), global_p))
        for subject_key, benchmark_key in zip(sc_df["subject_key"].tolist(), sc_df["benchmark_key"].tolist())
    ]
    sc_df["p"] = _smoothed(
        sc_df["k"].to_numpy(),
        sc_df["n"].to_numpy(),
        sc_df["parent_p"].to_numpy(),
        float(kappas["subject_category"]),
    )
    subject_category = {
        join_key(str(r.subject_key), str(r.benchmark_key), str(r.condition_key)): float(r.p)
        for r in sc_df.itertuples(index=False)
    }

    return LockedPriorTables(
        global_p=global_p,
        benchmark=benchmark,
        benchmark_condition=benchmark_condition,
        subject=subject,
        subject_benchmark=subject_benchmark,
        subject_category=subject_category,
        kappas={k: float(v) for k, v in kappas.items()},
    )


def prior_probs_for_frame(df, priors: LockedPriorTables) -> np.ndarray:
    """Vectorized subject-category fallback lookup for canonicalized rows."""

    import pandas as pd

    n = len(df)
    probs = np.full(n, float(priors.global_p), dtype=np.float64)

    bench = df["benchmark_key"].astype(str)
    mapped = bench.map(priors.benchmark).to_numpy(dtype=np.float64, na_value=np.nan)
    probs = np.where(np.isfinite(mapped), mapped, probs)

    bc_keys = pd.Series(
        np.char.add(np.char.add(bench.to_numpy(dtype=str), priors.key_sep), df["condition_key"].astype(str).to_numpy(dtype=str)),
        index=df.index,
    )
    mapped = bc_keys.map(priors.benchmark_condition).to_numpy(dtype=np.float64, na_value=np.nan)
    probs = np.where(np.isfinite(mapped), mapped, probs)

    subject = df["subject_key"].astype(str)
    mapped = subject.map(priors.subject).to_numpy(dtype=np.float64, na_value=np.nan)
    probs = np.where(np.isfinite(mapped), mapped, probs)

    sb_keys = pd.Series(
        np.char.add(np.char.add(subject.to_numpy(dtype=str), priors.key_sep), bench.to_numpy(dtype=str)),
        index=df.index,
    )
    mapped = sb_keys.map(priors.subject_benchmark).to_numpy(dtype=np.float64, na_value=np.nan)
    probs = np.where(np.isfinite(mapped), mapped, probs)

    sc_keys = pd.Series(
        np.char.add(
            np.char.add(sb_keys.to_numpy(dtype=str), priors.key_sep),
            df["condition_key"].astype(str).to_numpy(dtype=str),
        ),
        index=df.index,
    )
    mapped = sc_keys.map(priors.subject_category).to_numpy(dtype=np.float64, na_value=np.nan)
    probs = np.where(np.isfinite(mapped), mapped, probs)
    return np.clip(probs, 1e-6, 1.0 - 1e-6)


def item_fold_map(item_ids: list[str], n_folds: int, seed: int) -> dict[str, int]:
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    folds: dict[str, int] = {}
    for item_id in sorted(dict.fromkeys(str(item_id) for item_id in item_ids)):
        unit = hash_to_unit(item_id, seed)
        folds[item_id] = min(int(unit * n_folds), n_folds - 1)
    return folds


def weighted_feature_stats(x: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weight = np.asarray(weight, dtype=np.float64)
    weight = np.where(np.isfinite(weight) & (weight > 0.0), weight, 0.0)
    denom = float(weight.sum())
    if denom <= 0.0:
        raise ValueError("feature weights sum to zero")
    mean = (x.astype(np.float64) * weight[:, None]).sum(axis=0) / denom
    centered = x.astype(np.float64) - mean
    var = (centered * centered * weight[:, None]).sum(axis=0) / denom
    std = np.sqrt(np.maximum(var, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


def fit_weighted_ridge(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray | float]:
    """Fit weighted ridge with unpenalized intercept on standardized features."""

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    keep = np.isfinite(y) & np.isfinite(weight) & (weight > 0.0) & np.isfinite(x).all(axis=1)
    if int(keep.sum()) <= x.shape[1]:
        raise ValueError(f"not enough finite weighted samples for ridge: n={int(keep.sum())}, d={x.shape[1]}")
    x = x[keep]
    y = y[keep]
    weight = weight[keep]

    mean, std = weighted_feature_stats(x, weight)
    x_std = (x.astype(np.float64) - mean.astype(np.float64)) / std.astype(np.float64)
    design = np.concatenate([np.ones((len(x_std), 1), dtype=np.float64), x_std], axis=1)
    sqrt_w = np.sqrt(weight).reshape(-1, 1)
    design_w = design * sqrt_w
    y_w = y * sqrt_w.reshape(-1)
    lhs = design_w.T @ design_w
    penalty = np.eye(lhs.shape[0], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    rhs = design_w.T @ y_w
    beta = np.linalg.solve(lhs + penalty, rhs)
    return {
        "intercept": float(beta[0]),
        "coef": beta[1:].astype(np.float32),
        "feature_mean": mean,
        "feature_std": std,
    }


def predict_weighted_ridge(x: np.ndarray, model: Mapping[str, Any]) -> np.ndarray:
    mean = np.asarray(model["feature_mean"], dtype=np.float32)
    std = np.asarray(model["feature_std"], dtype=np.float32)
    coef = np.asarray(model["coef"], dtype=np.float32)
    intercept = float(model["intercept"])
    return ((np.asarray(x, dtype=np.float32) - mean) / std).astype(np.float32) @ coef + intercept


def weighted_rmse(pred: np.ndarray, target: np.ndarray, weight: np.ndarray) -> float:
    weight = np.asarray(weight, dtype=np.float64)
    denom = float(weight.sum())
    if denom <= 0.0:
        return float("nan")
    err = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.sqrt(np.sum(weight * err * err) / denom))


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    if int(keep.sum()) < 2:
        return float("nan")
    a = a[keep]
    b = b[keep]
    if float(a.std()) <= 0.0 or float(b.std()) <= 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def bce_mll_from_prior_delta(labels: np.ndarray, prior_p: np.ndarray, delta: np.ndarray, weight_w: float) -> float:
    logits = logit_prob(np.asarray(prior_p, dtype=np.float64)) + float(weight_w) * np.asarray(delta, dtype=np.float64)
    probs = np.asarray(sigmoid(logits), dtype=np.float64)
    return mean_log_likelihood(probs, np.asarray(labels, dtype=np.int8))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

