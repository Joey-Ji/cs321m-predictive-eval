"""Adaptive labeling hook for v1_kfactor.

Scores uncertain examples highest so the platform reveals labels that are most
informative for the per-subject online adjustment.
"""

from __future__ import annotations

import math


def acquisition_function(input: dict) -> float:
    """Return a finite labeling-priority score; never raise."""
    try:
        from model import ENCODER_OK, _raw_logit, _sigmoid

        if not ENCODER_OK:
            return 0.0
        p = _sigmoid(float(_raw_logit(input)))
        score = 1.0 - 2.0 * abs(p - 0.5)
        if not math.isfinite(score):
            return 0.0
        return float(max(min(score, 1.0), 0.0))
    except Exception:  # noqa: BLE001
        return 0.0
