"""Shared feature construction for item-content models."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

FEATURE_TEXT_VERSION = "benchmark_condition_item_v1"
RAW_ITEM_TEXT_VERSION = "item_content_v1"


def _clean_field(value: Any) -> str:
    """Convert common missing/null sentinels into stable empty strings."""
    if value is None:
        return ""
    if isinstance(value, Real) and math.isnan(float(value)):
        return ""
    return str(value)


def build_item_feature_text(row_or_input: dict, max_chars: int = 4000) -> str:
    """Build deterministic item text from benchmark, condition, and content.

    The item body is truncated before formatting so the header fields are never
    clipped away by a long prompt.
    """
    if max_chars < 0:
        raise ValueError(f"max_chars must be non-negative, got {max_chars}")

    benchmark = _clean_field(row_or_input.get("benchmark", ""))
    condition = _clean_field(row_or_input.get("condition", ""))
    item_content = _clean_field(row_or_input.get("item_content", ""))[:max_chars]
    return f"Benchmark: {benchmark}\nCondition: {condition}\nItem:\n{item_content}"
