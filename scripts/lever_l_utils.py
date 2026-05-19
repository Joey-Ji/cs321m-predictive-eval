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
SIZE_B_RE = re.compile(r"(?:(\d+(?:\.\d+)?)\s*x\s*)?(\d+(?:\.\d+)?)\s*b\b", re.IGNORECASE)
SIZE_M_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*m\b", re.IGNORECASE)
SUBJECT_METADATA_COLUMNS = ("subject_family", "subject_size_bucket", "subject_specialization")
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

FAMILY_PATTERNS = (
    ("openbiollm", ("openbiollm",)),
    ("biomistral", ("biomistral",)),
    ("meditron", ("meditron",)),
    ("medpalm", ("medpalm",)),
    ("medlm", ("medlm",)),
    ("llava", ("llava",)),
    ("mixtral", ("mixtral", "open-mixtral")),
    ("mistral", ("mistral", "codestral", "devstral", "ministral", "magistral", "mathstral", "pixtral")),
    ("llama", ("llama", "alpaca", "vicuna", "guanaco", "koala", "baize", "swellama")),
    ("claude", ("claude", "sonnet", "opus")),
    ("gpt", ("gpt", "chatgpt", "openai", "o1", "o3", "o4", "grokking")),
    ("gemini", ("gemini", "bard")),
    ("gemma", ("gemma", "paligemma")),
    ("qwen", ("qwen", "qwq", "qvq")),
    ("phi", ("phi",)),
    ("deepseek", ("deepseek", "deepcoder", "dscoder")),
    ("glm", ("glm", "chatglm")),
    ("grok", ("grok",)),
    ("command", ("command-r", "c4ai-command", "qodo_command")),
    ("kimi", ("kimi", "moonshot")),
    ("yi", ("yi-", "yi_", "points-yi", "faro-yi")),
    ("internvl", ("internvl",)),
    ("internlm", ("internlm",)),
    ("ovis", ("ovis",)),
    ("minicpm", ("minicpm",)),
    ("smolvlm", ("smolvlm",)),
    ("molmo", ("molmo",)),
    ("idefics", ("idefics",)),
    ("mplug", ("mplug",)),
    ("xcomposer", ("xcomposer",)),
    ("chameleon", ("chameleon",)),
    ("mantis", ("mantis",)),
    ("janus", ("janus",)),
    ("hunyuan", ("hunyuan",)),
    ("step", ("step",)),
    ("reka", ("reka",)),
    ("nemotron", ("nemotron",)),
    ("skywork", ("skywork",)),
    ("xlam", ("xlam",)),
    ("tulu", ("tulu",)),
    ("olmo", ("olmo",)),
    ("pythia", ("pythia",)),
    ("openbmb", ("openbmb", "eurus", "ultrarm")),
    ("compassjudger", ("compassjudger",)),
    ("mpt", ("mpt",)),
    ("stablelm", ("stablelm", "stable-code")),
    ("starchat", ("starchat",)),
    ("rwkv", ("rwkv",)),
    ("jamba", ("jamba",)),
    ("solar", ("solar",)),
    ("exaone", ("exaone",)),
    ("aquila", ("aquila",)),
    ("ziya", ("ziya",)),
    ("dbrx", ("dbrx",)),
    ("dolly", ("dolly",)),
    ("t5", ("t5",)),
    ("falcon", ("falcon",)),
    ("palm", ("palm",)),
    ("smoltulu", ("smoltulu",)),
    ("starling", ("starling",)),
    ("prometheus", ("prometheus",)),
    ("beaver", ("beaver",)),
    ("slime", ("slime",)),
    ("matter", ("matter",)),
    ("eagle", ("eagle",)),
    ("vila", ("vila",)),
    ("cambrian", ("cambrian",)),
    ("minigpt", ("minigpt",)),
    ("moondream", ("moondream",)),
    ("bailingmm", ("bailingmm",)),
    ("vision", (
        "360vl",
        "aria",
        "bluelm",
        "doubaovl",
        "jtvl",
        "kosmos",
        "mmalaya",
        "mug-u",
        "nvlm",
        "omchat",
        "omnilmm",
        "sail-vl",
        "sensechat",
        "taichu-vlr",
        "taiyi",
        "telemm",
        "vlm-r1",
        "vxverse",
        "vintern",
        "wemm",
        "xinyuan-vl",
        "cogvlm",
        "emu2",
        "emu3",
        "flamingo",
        "h2ovl",
        "instructblip",
        "monkey",
        "sharecaptioner",
        "varco-vision",
        "vita",
    )),
    ("wizardlm", ("wizardlm",)),
    ("functionary", ("firefunction", "functionary", "toolace", "watt-tool")),
    ("granite", ("granite",)),
    ("minimax", ("minimax",)),
    ("reasoning", ("k2-think", "openthinker", "qed-nano", "limo", "s1.1")),
    ("zephyr", ("zephyr",)),
    ("ultralm", ("ultralm",)),
    ("nova", ("nova",)),
    ("frog", ("frogboss", "frogmini")),
    ("agent", (
        "agent",
        "openhands",
        "agentless",
        "sweagent",
        "swe-agent",
        "autocoderover",
        "solver",
        "gru",
        "nfactorial",
        "devlo",
        "trae",
        "augment",
        "amazon-q",
        "codestory",
        "composio",
        "emergent",
        "epam-ai-run",
        "factory",
        "navie",
        "zencoder",
        "artemis",
        "warp",
        "lingma",
        "marscode",
        "patchpilot",
        "refact",
        "cortexa",
        "qodo",
        "qodo-command",
        "prometheus-v",
        "sonar-foundation",
        "livesweagent",
        "joycode",
        "atlassian-rovo",
        "blackboxai",
        "codeshellagent",
        "codesweep",
        "deepswerl",
        "entropo",
        "harness_ai",
        "learn_by_interact",
        "moatless",
        "swe-fixer",
        "swe-exp",
        "swe-rizzo",
        "ugaiforge",
        "wandb-programmer",
        "acoder",
        "aime-coder",
        "bracket",
        "droidrun",
        "finalrun",
        "harness-ai",
        "sage",
    )),
)

SPECIALIZATION_PATTERNS = (
    ("bio", ("bio", "pmc", "biomistral", "openbiollm")),
    ("med", ("med", "palm", "clinical")),
    ("code", ("code", "coder", "codestral", "devstral", "swe", "programmer", "function", "tool", "patch")),
    ("math", ("math", "aime", "imo", "qwq", "qvq", "reasoning")),
    ("vision", (
        "vision",
        "vl",
        "vlm",
        "llava",
        "internvl",
        "gpt4v",
        "qwen-vl",
        "qwen2-vl",
        "qwen2.5-vl",
        "geminiprovision",
        "pixtral",
        "ovis",
        "minigpt",
        "mplug",
        "molmo",
        "idefics",
        "cogvlm",
        "visual",
        "vlaa",
        "xcomposer",
        "cambrian",
        "mantis",
        "janus",
        "bunny",
        "eagle",
        "paligemma",
        "moondream",
        "smolvlm",
        "chameleon",
        "panda",
    )),
)


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


def _metadata_text(display_name: str) -> str:
    text = normalize_subject(clean_str(display_name))
    text = re.sub(r"(?<=\d)_(?=\d)", ".", text)
    return re.sub(r"[^a-z0-9.]+", "-", text.lower()).strip("-")


def _contains_token(text: str, token: str) -> bool:
    token = token.lower()
    if not token:
        return False
    if len(token) <= 2:
        return re.search(rf"(^|[^a-z0-9]){re.escape(token)}([^a-z0-9]|$)", text) is not None
    return token in text


def _parse_family(text: str) -> str:
    if not text:
        return "other"
    for family, tokens in FAMILY_PATTERNS:
        if any(_contains_token(text, token) for token in tokens):
            return family
    return "other"


def _parse_size_bucket(text: str) -> str:
    sizes_b: list[float] = []
    for match in SIZE_B_RE.finditer(text):
        multiplier = float(match.group(1)) if match.group(1) is not None else 1.0
        sizes_b.append(multiplier * float(match.group(2)))
    for match in SIZE_M_RE.finditer(text):
        sizes_b.append(float(match.group(1)) / 1000.0)
    if not sizes_b:
        return "unknown"
    size_b = max(sizes_b)
    if size_b < 3.0:
        return "tiny"
    if size_b < 10.0:
        return "small"
    if size_b < 30.0:
        return "mid"
    if size_b < 100.0:
        return "large"
    return "xl"


def _parse_specialization(text: str) -> str:
    if not text:
        return "unknown"
    for specialization, tokens in SPECIALIZATION_PATTERNS:
        if any(_contains_token(text, token) for token in tokens):
            return specialization
    return "general"


def parse_subject_metadata(display_name: str) -> dict[str, str]:
    """Extract deterministic subject metadata from a display name or subject_content."""
    text = _metadata_text(display_name)
    return {
        "family": _parse_family(text),
        "size_bucket": _parse_size_bucket(text),
        "specialization": _parse_specialization(text),
    }


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
    family: dict[str, float]
    family_size: dict[str, float]
    family_benchmark: dict[str, float]
    subject: dict[str, float]
    subject_benchmark: dict[str, float]
    subject_category: dict[str, float]
    kappas: dict[str, float]

    def lookup(self, subject_key: str, benchmark: str, condition: str) -> tuple[float, ...]:
        global_p = self.global_p
        bench_p = self.benchmark.get(benchmark, global_p)
        bc_p = self.benchmark_condition.get(join_key(benchmark, condition), bench_p)
        metadata = parse_subject_metadata(subject_key)
        family_key = metadata["family"]
        size_bucket = metadata["size_bucket"]
        family_p = self.family.get(family_key, global_p)
        family_size_p = self.family_size.get(join_key(family_key, size_bucket), family_p)
        family_benchmark_p = self.family_benchmark.get(join_key(family_key, benchmark), family_p)
        subj_p = self.subject.get(subject_key, family_size_p)
        sb_parent = _subject_benchmark_parent(subj_p, family_benchmark_p, family_p)
        sb_p = self.subject_benchmark.get(join_key(subject_key, benchmark), sb_parent)
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


def _ensure_subject_metadata(df):
    if all(col in df.columns for col in SUBJECT_METADATA_COLUMNS):
        return df
    if "subject_key" not in df.columns:
        raise KeyError("fit_priors requires a subject_key column")
    out = df.copy()
    metadata_by_subject = {
        str(subject_key): parse_subject_metadata(str(subject_key))
        for subject_key in out["subject_key"].drop_duplicates().tolist()
    }
    out["subject_family"] = out["subject_key"].map(
        lambda subject_key: metadata_by_subject[str(subject_key)]["family"]
    )
    out["subject_size_bucket"] = out["subject_key"].map(
        lambda subject_key: metadata_by_subject[str(subject_key)]["size_bucket"]
    )
    out["subject_specialization"] = out["subject_key"].map(
        lambda subject_key: metadata_by_subject[str(subject_key)]["specialization"]
    )
    return out


def _safe_parent(parent: Mapping[str, float], key: str, fallback: float) -> float:
    value = parent.get(key, fallback)
    if not math.isfinite(float(value)):
        return fallback
    return float(value)


def _smoothed(k: np.ndarray, n: np.ndarray, parent_p: np.ndarray, kappa: float) -> np.ndarray:
    return (k.astype(np.float64) + float(kappa) * parent_p.astype(np.float64)) / (
        n.astype(np.float64) + float(kappa)
    )


def _subject_benchmark_parent(subject_p: float, family_benchmark_p: float, family_p: float) -> float:
    parent_logit = float(logit_prob(subject_p)) + float(logit_prob(family_benchmark_p)) - float(logit_prob(family_p))
    return float(np.clip(sigmoid(parent_logit), 1e-6, 1.0 - 1e-6))


def fit_priors(df, kappas: Mapping[str, float]) -> PriorTables:
    df = _ensure_subject_metadata(df)
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

    family_df = _group_counts(df, ["subject_family"])
    family_df["parent_p"] = global_p
    family_df["p"] = _smoothed(
        family_df["k"].to_numpy(),
        family_df["n"].to_numpy(),
        family_df["parent_p"].to_numpy(),
        float(kappas["family"]),
    )
    family = {str(r.subject_family): float(r.p) for r in family_df.itertuples(index=False)}

    fs_df = _group_counts(df, ["subject_family", "subject_size_bucket"])
    fs_df["parent_p"] = [
        _safe_parent(family, str(family_key), global_p)
        for family_key in fs_df["subject_family"].tolist()
    ]
    fs_df["p"] = _smoothed(
        fs_df["k"].to_numpy(),
        fs_df["n"].to_numpy(),
        fs_df["parent_p"].to_numpy(),
        float(kappas["family_size"]),
    )
    family_size = {
        join_key(str(r.subject_family), str(r.subject_size_bucket)): float(r.p)
        for r in fs_df.itertuples(index=False)
    }

    fb_df = _group_counts(df, ["subject_family", "benchmark"])
    fb_df["parent_p"] = [
        _safe_parent(family, str(family_key), global_p)
        for family_key in fb_df["subject_family"].tolist()
    ]
    fb_df["p"] = _smoothed(
        fb_df["k"].to_numpy(),
        fb_df["n"].to_numpy(),
        fb_df["parent_p"].to_numpy(),
        float(kappas["family_benchmark"]),
    )
    family_benchmark = {
        join_key(str(r.subject_family), str(r.benchmark)): float(r.p)
        for r in fb_df.itertuples(index=False)
    }

    subj_df = _group_counts(df, ["subject_key", "subject_family", "subject_size_bucket"])
    subj_df["parent_p"] = [
        _safe_parent(
            family_size,
            join_key(str(family_key), str(size_bucket)),
            _safe_parent(family, str(family_key), global_p),
        )
        for family_key, size_bucket in zip(
            subj_df["subject_family"].tolist(),
            subj_df["subject_size_bucket"].tolist(),
        )
    ]
    subj_df["p"] = _smoothed(
        subj_df["k"].to_numpy(),
        subj_df["n"].to_numpy(),
        subj_df["parent_p"].to_numpy(),
        float(kappas["subject"]),
    )
    subject = {str(r.subject_key): float(r.p) for r in subj_df.itertuples(index=False)}

    sb_df = _group_counts(df, ["subject_key", "subject_family", "benchmark"])
    sb_df["parent_p"] = [
        _subject_benchmark_parent(
            _safe_parent(subject, str(subject_key), global_p),
            _safe_parent(family_benchmark, join_key(str(family_key), str(benchmark)), _safe_parent(family, str(family_key), global_p)),
            _safe_parent(family, str(family_key), global_p),
        )
        for subject_key, family_key, benchmark in zip(
            sb_df["subject_key"].tolist(),
            sb_df["subject_family"].tolist(),
            sb_df["benchmark"].tolist(),
        )
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
        family=family,
        family_size=family_size,
        family_benchmark=family_benchmark,
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
    train_df = _ensure_subject_metadata(train_df)
    kappas = {
        "benchmark": 10.0,
        "benchmark_condition": 10.0,
        "family": 10.0,
        "family_size": 10.0,
        "family_benchmark": 10.0,
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
    df = _ensure_subject_metadata(df)
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

    family_df = _group_counts(df, ["subject_family"])
    family_df["parent_p"] = priors.global_p
    family_df["kappa"] = float(kappas["family"])
    family_df["p"] = [priors.family[str(v)] for v in family_df["subject_family"].tolist()]
    family_df.to_parquet(out_dir / "family.parquet", index=False)

    fs_df = _group_counts(df, ["subject_family", "subject_size_bucket"])
    fs_df["parent_p"] = [
        priors.family.get(str(family_key), priors.global_p)
        for family_key in fs_df["subject_family"].tolist()
    ]
    fs_df["kappa"] = float(kappas["family_size"])
    fs_df["p"] = [
        priors.family_size[join_key(str(family_key), str(size_bucket))]
        for family_key, size_bucket in zip(fs_df["subject_family"].tolist(), fs_df["subject_size_bucket"].tolist())
    ]
    fs_df.to_parquet(out_dir / "family_size.parquet", index=False)

    fb_df = _group_counts(df, ["subject_family", "benchmark"])
    fb_df["parent_p"] = [
        priors.family.get(str(family_key), priors.global_p)
        for family_key in fb_df["subject_family"].tolist()
    ]
    fb_df["kappa"] = float(kappas["family_benchmark"])
    fb_df["p"] = [
        priors.family_benchmark[join_key(str(family_key), str(bench))]
        for family_key, bench in zip(fb_df["subject_family"].tolist(), fb_df["benchmark"].tolist())
    ]
    fb_df.to_parquet(out_dir / "family_benchmark.parquet", index=False)

    subj_df = _group_counts(df, ["subject_key", "subject_family", "subject_size_bucket"])
    subj_df["parent_p"] = [
        priors.family_size.get(
            join_key(str(family_key), str(size_bucket)),
            priors.family.get(str(family_key), priors.global_p),
        )
        for family_key, size_bucket in zip(
            subj_df["subject_family"].tolist(),
            subj_df["subject_size_bucket"].tolist(),
        )
    ]
    subj_df["kappa"] = float(kappas["subject"])
    subj_df["p"] = [priors.subject[str(v)] for v in subj_df["subject_key"].tolist()]
    subj_df.to_parquet(out_dir / "subject.parquet", index=False)

    sb_df = _group_counts(df, ["subject_key", "subject_family", "benchmark"])
    sb_df["parent_p"] = [
        _subject_benchmark_parent(
            priors.subject.get(str(subject_key), priors.global_p),
            priors.family_benchmark.get(
                join_key(str(family_key), str(bench)),
                priors.family.get(str(family_key), priors.global_p),
            ),
            priors.family.get(str(family_key), priors.global_p),
        )
        for subject_key, family_key, bench in zip(
            sb_df["subject_key"].tolist(),
            sb_df["subject_family"].tolist(),
            sb_df["benchmark"].tolist(),
        )
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
        "family": priors.family,
        "family_size": priors.family_size,
        "family_benchmark": priors.family_benchmark,
        "subject": priors.subject,
        "subject_benchmark": priors.subject_benchmark,
        "subject_category": priors.subject_category,
        "key_sep": KEY_SEP,
        "prior_columns": PRIOR_COLUMNS,
        "subject_metadata_contract": {
            "family_size_key": ["family", "size_bucket"],
            "family_benchmark_key": ["family", "benchmark"],
            "size_buckets": ["tiny", "small", "mid", "large", "xl", "unknown"],
            "specializations": ["general", "bio", "med", "code", "math", "vision", "unknown"],
        },
    }
    (out_dir / "runtime_priors.json").write_text(json.dumps(runtime, separators=(",", ":")) + "\n")
