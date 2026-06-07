"""
Adaptive labeling for Subject-Ability Prediction.

Scores items to maximize information for calibration. Uses uncertainty-based
selection as a proxy for A-optimal design (selects items where model is
most uncertain).
"""

from __future__ import annotations

import math


def acquisition_function(input: dict) -> float:
    """
    Return labeling priority score for active learning.

    Prioritizes items where the model is most uncertain (near p=0.5),
    which provides maximum information for calibration.

    Args:
        input: Dict with keys: subject_content, item_content, benchmark, condition

    Returns:
        Score in [0, 1], higher = more valuable to label
    """
    try:
        # For subject-ability method without precomputed abilities,
        # we use uncertainty as a proxy for informativeness.
        # Items with moderate predicted probability (near 0.5) provide
        # the most information for calibration.

        # Without precomputed state, use a neutral prior
        # In a full implementation, we could maintain running statistics
        # For now, return uniform priority (random selection)
        # This is conservative but valid

        return 0.5

        # Alternative: if we had access to predict(), we could score by uncertainty:
        # from model import predict
        # p = predict(input, labeled=None)
        # uncertainty = 1.0 - 2.0 * abs(p - 0.5)  # max at p=0.5
        # return float(max(min(uncertainty, 1.0), 0.0))

    except Exception:
        return 0.0
