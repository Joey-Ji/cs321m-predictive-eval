"""Shared feature construction for item-content models."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Iterable

FEATURE_TEXT_VERSION = "benchmark_condition_item_v1"
RAW_ITEM_TEXT_VERSION = "item_content_v1"
EMBEDDING_REPRESENTATION_VERSION = "item_content_v1"
REPRESENTATION_VERSION = "item_text_plus_side_features_v1"


def _clean_field(value: Any) -> str:
    """Convert common missing/null sentinels into stable empty strings."""
    if value is None:
        return ""
    if isinstance(value, Real) and math.isnan(float(value)):
        return ""
    return str(value)


def build_item_feature_text(row_or_input: dict, max_chars: int = 4000) -> str:
    """Build deterministic item text from benchmark, condition, and content.

    Retained as a documented ablation. The default Stage 2 pipeline encodes
    only `item_content` and treats benchmark/condition as one-hot side
    features at head input (see `build_side_feature_vocab`).
    """
    if max_chars < 0:
        raise ValueError(f"max_chars must be non-negative, got {max_chars}")

    benchmark = _clean_field(row_or_input.get("benchmark", ""))
    condition = _clean_field(row_or_input.get("condition", ""))
    item_content = _clean_field(row_or_input.get("item_content", ""))[:max_chars]
    return f"Benchmark: {benchmark}\nCondition: {condition}\nItem:\n{item_content}"


def build_side_feature_vocab(rows: Iterable[dict]) -> dict:
    """Build a deterministic one-hot vocabulary for benchmark + condition.

    Walks all rows once, collects distinct cleaned values for each field
    (missing/null becomes ""), and assigns contiguous indices in sorted order.
    """
    bench_values: set[str] = set()
    cond_values: set[str] = set()
    for row in rows:
        bench_values.add(_clean_field(row.get("benchmark", "")))
        cond_values.add(_clean_field(row.get("condition", "")))
    benchmark_to_idx = {v: i for i, v in enumerate(sorted(bench_values))}
    condition_to_idx = {v: i for i, v in enumerate(sorted(cond_values))}
    return {
        "benchmark": benchmark_to_idx,
        "condition": condition_to_idx,
        "benchmark_dim": len(benchmark_to_idx),
        "condition_dim": len(condition_to_idx),
        "side_feature_dim": len(benchmark_to_idx) + len(condition_to_idx),
        "representation_version": REPRESENTATION_VERSION,
    }


def encode_side_features(row: dict, vocab: dict):
    """Return a float32 [benchmark_dim + condition_dim] one-hot block for a row.

    Unseen benchmark/condition values produce an all-zero block for that
    field — callers are expected to track unseen counts if observability is
    desired, but encoding never raises.
    """
    import numpy as np

    b_dim = int(vocab["benchmark_dim"])
    c_dim = int(vocab["condition_dim"])
    out = np.zeros(b_dim + c_dim, dtype=np.float32)
    bench = _clean_field(row.get("benchmark", ""))
    cond = _clean_field(row.get("condition", ""))
    b_idx = vocab["benchmark"].get(bench)
    c_idx = vocab["condition"].get(cond)
    if b_idx is not None:
        out[int(b_idx)] = 1.0
    if c_idx is not None:
        out[b_dim + int(c_idx)] = 1.0
    return out
