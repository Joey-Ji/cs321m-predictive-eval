"""Shared helpers for Lever L split-faithful priors and residual training."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

NAME_LINE = re.compile(r"^\s*Name:\s*(.+?)\s*$", re.MULTILINE)
CATEGORY_KEYS = ("benchmark", "condition")
PRIOR_COLUMNS = [
    "prior_global",
    "prior_benchmark",
    "prior_benchmark_condition",
    "prior_subject",
    "prior_subject_benchmark",
    "prior_subject_category",
]
KAPPA_GRID = (2.0, 5.0, 10.0, 20.0, 50.0)
KEY_SEP = "\x1f"


def clean_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def normalize_subject(subject_content: str) -> str:
    if not subject_content:
        return ""
    match = NAME_LINE.search(subject_content)
    return match.group(1).strip().lower() if match else subject_content.strip().lower()


def coerce_binary_label(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value if value in (0, 1) else None
    if isinstance(value, float) and math.isfinite(value) and value in (0.0, 1.0):
        return int(value)
    return None


def hash_to_unit(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}|{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def validation_item_ids(item_ids: list[str], val_frac: float, seed: int) -> set[str]:
    if not 0.0 <= val_frac < 1.0:
        raise ValueError(f"val_frac must be in [0, 1), got {val_frac}")
    ordered = list(dict.fromkeys(str(iid) for iid in item_ids))
    held = {iid for iid in ordered if hash_to_unit(iid, seed) < val_frac}
    if val_frac > 0.0 and not held and len(ordered) > 1:
        held.add(min(ordered, key=lambda iid: hash_to_unit(iid, seed)))
    if len(held) == len(ordered) and len(ordered) > 1:
        held.remove(max(held, key=lambda iid: hash_to_unit(iid, seed)))
    return held


def category_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return tuple(clean_str(row.get(key)) for key in CATEGORY_KEYS)  # type: ignore[return-value]


def group_rows_by_category(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(category_key(row), []).append(row)
    return groups


def allocate_category_quotas(
    groups: Mapping[tuple[str, str], list[dict[str, Any]]],
    max_rows: int,
    max_per_category: int,
) -> dict[tuple[str, str], int]:
    if max_rows <= 0:
        raise ValueError(f"max_rows must be > 0, got {max_rows}")
    if max_per_category <= 0:
        raise ValueError(f"max_per_category must be > 0, got {max_per_category}")
    caps = {
        category: min(len(rows), max_per_category)
        for category, rows in groups.items()
        if rows
    }
    quotas = {category: 0 for category in caps}
    remaining = min(max_rows, sum(caps.values()))
    active = sorted(caps)
    while remaining > 0 and active:
        next_active: list[tuple[str, str]] = []
        for category in active:
            if remaining <= 0:
                break
            if quotas[category] >= caps[category]:
                continue
            quotas[category] += 1
            remaining -= 1
            if quotas[category] < caps[category]:
                next_active.append(category)
        active = next_active
    return quotas


def sample_groups(
    groups: Mapping[tuple[str, str], list[dict[str, Any]]],
    rng: random.Random,
    max_rows: int,
    max_per_category: int,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    quotas = allocate_category_quotas(groups, max_rows=max_rows, max_per_category=max_per_category)
    sampled: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for category in sorted(quotas):
        quota = quotas[category]
        if quota <= 0:
            continue
        rows = groups[category]
        sampled[category] = list(rows) if len(rows) <= quota else rng.sample(rows, quota)
    return sampled


def sigmoid(logit: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-logit))


def logit_prob(p: float | np.ndarray) -> float | np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def mean_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.clip(probs.astype(np.float64), 1e-6, 1.0 - 1e-6)
    labels = labels.astype(np.float64)
    return float(np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)))


def auc_roc(probs: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(np.int8)
    pos = int(labels.sum())
    neg = int(len(labels) - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(probs, kind="mergesort")
    ranks = np.empty(len(probs), dtype=np.float64)
    sorted_probs = probs[order]
    i = 0
    while i < len(probs):
        j = i
        while j + 1 < len(probs) and sorted_probs[j + 1] == sorted_probs[i]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


@dataclass
class PriorTables:
    global_p: float
    benchmark: dict[str, float]
    benchmark_condition: dict[str, float]
    subject: dict[str, float]
    subject_benchmark: dict[str, float]
    subject_category: dict[str, float]
    kappas: dict[str, float]

    def lookup(self, subject_key: str, benchmark: str, condition: str) -> tuple[float, ...]:
        global_p = self.global_p
        bench_p = self.benchmark.get(benchmark, global_p)
        bc_p = self.benchmark_condition.get(join_key(benchmark, condition), bench_p)
        subj_p = self.subject.get(subject_key, global_p)
        sb_p = self.subject_benchmark.get(join_key(subject_key, benchmark), subj_p)
        sc_p = self.subject_category.get(join_key(subject_key, benchmark, condition), sb_p)
        return global_p, bench_p, bc_p, subj_p, sb_p, sc_p


def join_key(*parts: str) -> str:
    return KEY_SEP.join(clean_str(part) for part in parts)


def split_key(key: str) -> list[str]:
    return key.split(KEY_SEP)


def _group_counts(df, keys: list[str]):
    grouped = df.groupby(keys, observed=True)["label"].agg(["count", "sum"]).reset_index()
    grouped = grouped.rename(columns={"count": "n", "sum": "k"})
    grouped["n"] = grouped["n"].astype("int64")
    grouped["k"] = grouped["k"].astype("float64")
    return grouped


def _safe_parent(parent: Mapping[str, float], key: str, fallback: float) -> float:
    value = parent.get(key, fallback)
    if not math.isfinite(float(value)):
        return fallback
    return float(value)


def _smoothed(k: np.ndarray, n: np.ndarray, parent_p: np.ndarray, kappa: float) -> np.ndarray:
    return (k.astype(np.float64) + float(kappa) * parent_p.astype(np.float64)) / (
        n.astype(np.float64) + float(kappa)
    )


def fit_priors(df, kappas: Mapping[str, float]) -> PriorTables:
    global_p = float(df["label"].mean())

    bench_df = _group_counts(df, ["benchmark"])
    bench_df["parent_p"] = global_p
    bench_df["p"] = _smoothed(
        bench_df["k"].to_numpy(),
        bench_df["n"].to_numpy(),
        bench_df["parent_p"].to_numpy(),
        float(kappas["benchmark"]),
    )
    benchmark = {str(r.benchmark): float(r.p) for r in bench_df.itertuples(index=False)}

    bc_df = _group_counts(df, ["benchmark", "condition"])
    bc_df["parent_p"] = [
        _safe_parent(benchmark, str(bench), global_p)
        for bench in bc_df["benchmark"].tolist()
    ]
    bc_df["p"] = _smoothed(
        bc_df["k"].to_numpy(),
        bc_df["n"].to_numpy(),
        bc_df["parent_p"].to_numpy(),
        float(kappas["benchmark_condition"]),
    )
    benchmark_condition = {
        join_key(str(r.benchmark), str(r.condition)): float(r.p)
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

    sb_df = _group_counts(df, ["subject_key", "benchmark"])
    sb_df["parent_p"] = [
        _safe_parent(subject, str(subject_key), global_p)
        for subject_key in sb_df["subject_key"].tolist()
    ]
    sb_df["p"] = _smoothed(
        sb_df["k"].to_numpy(),
        sb_df["n"].to_numpy(),
        sb_df["parent_p"].to_numpy(),
        float(kappas["subject_benchmark"]),
    )
    subject_benchmark = {
        join_key(str(r.subject_key), str(r.benchmark)): float(r.p)
        for r in sb_df.itertuples(index=False)
    }

    sc_df = _group_counts(df, ["subject_key", "benchmark", "condition"])
    sc_df["parent_p"] = [
        _safe_parent(subject_benchmark, join_key(str(subject_key), str(benchmark)), global_p)
        for subject_key, benchmark in zip(sc_df["subject_key"].tolist(), sc_df["benchmark"].tolist())
    ]
    sc_df["p"] = _smoothed(
        sc_df["k"].to_numpy(),
        sc_df["n"].to_numpy(),
        sc_df["parent_p"].to_numpy(),
        float(kappas["subject_category"]),
    )
    subject_category = {
        join_key(str(r.subject_key), str(r.benchmark), str(r.condition)): float(r.p)
        for r in sc_df.itertuples(index=False)
    }

    return PriorTables(
        global_p=global_p,
        benchmark=benchmark,
        benchmark_condition=benchmark_condition,
        subject=subject,
        subject_benchmark=subject_benchmark,
        subject_category=subject_category,
        kappas={k: float(v) for k, v in kappas.items()},
    )


def prior_vector_for_frame(df, priors: PriorTables) -> np.ndarray:
    out = np.empty((len(df), len(PRIOR_COLUMNS)), dtype=np.float32)
    for row_idx, row in enumerate(df[["subject_key", "benchmark", "condition"]].itertuples(index=False)):
        out[row_idx] = priors.lookup(str(row.subject_key), str(row.benchmark), str(row.condition))
    return out


def subject_category_probs_for_frame(df, priors: PriorTables) -> np.ndarray:
    vec = prior_vector_for_frame(df, priors)
    return vec[:, -1].astype(np.float64)


def tune_priors(train_df, val_df, grid: tuple[float, ...] = KAPPA_GRID) -> tuple[PriorTables, dict[str, float]]:
    kappas = {
        "benchmark": 10.0,
        "benchmark_condition": 10.0,
        "subject": 10.0,
        "subject_benchmark": 10.0,
        "subject_category": 10.0,
    }
    labels = val_df["label"].to_numpy(dtype=np.int8)
    for level in list(kappas):
        best_kappa = kappas[level]
        best_mll = -float("inf")
        for candidate in grid:
            trial = dict(kappas)
            trial[level] = float(candidate)
            priors = fit_priors(train_df, trial)
            probs = subject_category_probs_for_frame(val_df, priors)
            mll = mean_log_likelihood(probs, labels)
            if mll > best_mll:
                best_mll = mll
                best_kappa = float(candidate)
        kappas[level] = best_kappa
    priors = fit_priors(train_df, kappas)
    return priors, kappas


def write_prior_artifacts(df, kappas: Mapping[str, float], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    priors = fit_priors(df, kappas)

    (out_dir / "global.json").write_text(
        json.dumps(
            {
                "p": priors.global_p,
                "n": int(len(df)),
                "k": int(df["label"].sum()),
                "kappas": priors.kappas,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    bench_df = _group_counts(df, ["benchmark"])
    bench_df["parent_p"] = priors.global_p
    bench_df["kappa"] = float(kappas["benchmark"])
    bench_df["p"] = [priors.benchmark[str(v)] for v in bench_df["benchmark"].tolist()]
    bench_df.to_parquet(out_dir / "benchmark.parquet", index=False)

    bc_df = _group_counts(df, ["benchmark", "condition"])
    bc_df["parent_p"] = [
        priors.benchmark.get(str(bench), priors.global_p)
        for bench in bc_df["benchmark"].tolist()
    ]
    bc_df["kappa"] = float(kappas["benchmark_condition"])
    bc_df["p"] = [
        priors.benchmark_condition[join_key(str(bench), str(cond))]
        for bench, cond in zip(bc_df["benchmark"].tolist(), bc_df["condition"].tolist())
    ]
    bc_df.to_parquet(out_dir / "benchmark_condition.parquet", index=False)

    subj_df = _group_counts(df, ["subject_key"])
    subj_df["parent_p"] = priors.global_p
    subj_df["kappa"] = float(kappas["subject"])
    subj_df["p"] = [priors.subject[str(v)] for v in subj_df["subject_key"].tolist()]
    subj_df.to_parquet(out_dir / "subject.parquet", index=False)

    sb_df = _group_counts(df, ["subject_key", "benchmark"])
    sb_df["parent_p"] = [
        priors.subject.get(str(subject_key), priors.global_p)
        for subject_key in sb_df["subject_key"].tolist()
    ]
    sb_df["kappa"] = float(kappas["subject_benchmark"])
    sb_df["p"] = [
        priors.subject_benchmark[join_key(str(subject_key), str(bench))]
        for subject_key, bench in zip(sb_df["subject_key"].tolist(), sb_df["benchmark"].tolist())
    ]
    sb_df.to_parquet(out_dir / "subject_benchmark.parquet", index=False)

    sc_df = _group_counts(df, ["subject_key", "benchmark", "condition"])
    sc_df["parent_p"] = [
        priors.subject_benchmark.get(join_key(str(subject_key), str(bench)), priors.global_p)
        for subject_key, bench in zip(sc_df["subject_key"].tolist(), sc_df["benchmark"].tolist())
    ]
    sc_df["kappa"] = float(kappas["subject_category"])
    sc_df["p"] = [
        priors.subject_category[join_key(str(subject_key), str(bench), str(cond))]
        for subject_key, bench, cond in zip(
            sc_df["subject_key"].tolist(),
            sc_df["benchmark"].tolist(),
            sc_df["condition"].tolist(),
        )
    ]
    sc_df.to_parquet(out_dir / "subject_category.parquet", index=False)

    runtime = {
        "global": priors.global_p,
        "kappas": priors.kappas,
        "benchmark": priors.benchmark,
        "benchmark_condition": priors.benchmark_condition,
        "subject": priors.subject,
        "subject_benchmark": priors.subject_benchmark,
        "subject_category": priors.subject_category,
        "key_sep": KEY_SEP,
        "prior_columns": PRIOR_COLUMNS,
    }
    (out_dir / "runtime_priors.json").write_text(json.dumps(runtime, separators=(",", ":")) + "\n")
