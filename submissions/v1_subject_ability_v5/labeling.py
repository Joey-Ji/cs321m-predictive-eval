"""
Adaptive-labeling acquisition for the subject-ability submission.

We intentionally use a uniform (constant) acquisition score: every candidate
returns the same value, so the platform falls back to uniform-random selection
of the K labels per category. For this method, steering which labels get revealed
did not improve held-out log-loss, so we keep the acquisition channel neutral and
let predict() calibrate online over whatever anchors happen to be revealed.
"""

from __future__ import annotations


def acquisition_function(input: dict) -> float:
    """
    Return a constant acquisition score (uniform-random label selection by design).

    Args:
        input: Dict with keys: benchmark, condition, subject_content, item_content

    Returns:
        A finite float; constant across candidates so the platform selects K
        labels per category uniformly at random.
    """
    return 0.5
