"""
Subject-Ability Prediction with Per-Benchmark Calibration.

Lightweight, CPU-only method for predicting (subject, item) correctness
with adaptive per-category calibration.

Based on conference slide specification: uses Laplace-smoothed subject ability,
Fisher-gated per-category temperature/bias calibration on K=5 anchors.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # logistic sigmoid

# Constants
CLIP_LO, CLIP_HI = 0.02, 0.98
LAPLACE_ALPHA = 1.0
SHRINK_LAMBDA = 1.0
CALIBRATION_RIDGE = 0.01
GLOBAL_TEMPERATURE = 1.0

# Extract subject name from subject_content
NAME_PATTERN = re.compile(
    r"^\s*(?:Name|Subject)\s*:\s*(.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def _extract_subject_id(subject_content: str) -> str:
    """Extract subject ID from subject_content string."""
    match = NAME_PATTERN.search(subject_content)
    if match:
        return match.group(1).strip()
    # Fallback: use first line
    lines = subject_content.strip().split("\n")
    if lines:
        return lines[0].strip()
    return subject_content.strip()


def _compute_ability(observations: list[tuple[str, float]]) -> float:
    """
    Compute subject ability z_i from observations.

    Args:
        observations: List of (item_id, correct) tuples

    Returns:
        Ability logit z_i
    """
    if not observations:
        return 0.0  # Neutral prior

    correct_count = sum(correct for _, correct in observations)
    total_count = len(observations)

    # Laplace-smoothed accuracy
    acc = (correct_count + LAPLACE_ALPHA) / (total_count + 2 * LAPLACE_ALPHA)

    # Convert to logit, clip to avoid infinities
    acc = np.clip(acc, 1e-10, 1 - 1e-10)
    z_i = float(np.log(acc / (1 - acc)))

    return z_i


def _fit_logistic(
    z_values: np.ndarray,
    y_values: np.ndarray,
    rho_init: float = 1.0,
    beta0_init: float = 0.0,
    ridge: float = 0.0,
    rho_target: float = 1.0,
) -> tuple[float, float]:
    """
    Fit logistic regression p = σ(ρ·z + β0).

    Args:
        z_values: Features (subject abilities)
        y_values: Binary labels
        rho_init: Initial ρ (slope)
        beta0_init: Initial β0 (intercept)
        ridge: L2 penalty toward rho_target
        rho_target: Target for ridge regularization

    Returns:
        (rho, beta0) fitted parameters
    """
    if len(z_values) == 0:
        return 1.0, 0.0

    def neg_log_likelihood(params):
        rho, beta0 = params
        logits = rho * z_values + beta0
        probs = expit(logits)

        # NLL
        nll = -np.sum(
            y_values * np.log(probs + 1e-10)
            + (1 - y_values) * np.log(1 - probs + 1e-10)
        )

        # Ridge penalty
        if ridge > 0:
            nll += ridge * (rho - rho_target) ** 2

        return nll

    try:
        result = minimize(
            neg_log_likelihood,
            x0=[rho_init, beta0_init],
            method="L-BFGS-B",
            bounds=[(1e-6, 1e3), (None, None)],
        )
        rho, beta0 = result.x
        return float(rho), float(beta0)
    except Exception:
        return 1.0, 0.0


def _calibrate_category(
    anchors: list[dict[str, Any]],
    abilities: dict[str, float],
    global_rho: float = 1.0,
    global_beta0: float = 0.0,
) -> tuple[float, float]:
    """
    Calibrate temperature and bias for a category using anchors.

    Implements Fisher-gated shrinkage toward global calibration.

    Args:
        anchors: Anchor observations for this category
        abilities: Subject abilities
        global_rho: Global ρ (inverse temperature)
        global_beta0: Global β0

    Returns:
        (rho_final, beta0_final) after shrinkage
    """
    if len(anchors) == 0:
        return global_rho, global_beta0

    # Extract features and labels
    z_values = []
    y_values = []

    for anchor in anchors:
        subject_id = _extract_subject_id(str(anchor.get("subject_content", "")))
        z_i = abilities.get(subject_id, 0.0)
        label = float(anchor.get("label", 0.5))

        z_values.append(z_i)
        y_values.append(label)

    z_values = np.array(z_values)
    y_values = np.array(y_values)

    # Fit logistic regression
    rho_c, beta0_c = _fit_logistic(
        z_values,
        y_values,
        rho_init=global_rho,
        beta0_init=global_beta0,
        ridge=CALIBRATION_RIDGE,
        rho_target=global_rho,
    )

    # Compute Fisher information w.r.t. ρ
    logits = rho_c * z_values + beta0_c
    probs = expit(logits)
    fisher_rho = float(np.sum(probs * (1 - probs) * z_values**2))

    # Fisher-gated shrinkage weight
    w_c = fisher_rho / (fisher_rho + SHRINK_LAMBDA)

    # Shrink ρ toward global
    rho_final = w_c * rho_c + (1 - w_c) * global_rho

    return float(rho_final), float(beta0_c)


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """
    Predict probability that subject answers item correctly.

    Uses revealed anchor labels for per-category calibration.

    Args:
        input: Dict with keys: subject_content, item_content, benchmark, condition
        labeled: Optional list of revealed anchor labels

    Returns:
        Probability in [0, 1] clipped to [CLIP_LO, CLIP_HI]
    """
    try:
        # Extract current input info
        subject_id = _extract_subject_id(str(input.get("subject_content", "")))
        benchmark = str(input.get("benchmark", ""))

        # If no labeled data, return neutral prediction
        if not labeled or len(labeled) == 0:
            # Use neutral ability and calibration
            z_i = 0.0
            prob = expit(z_i)
            return float(np.clip(prob, CLIP_LO, CLIP_HI))

        # Group labeled data by subject and category
        observations_by_subject = defaultdict(list)
        labeled_by_category = defaultdict(list)

        for item in labeled:
            item_subject_id = _extract_subject_id(str(item.get("subject_content", "")))
            item_benchmark = str(item.get("benchmark", ""))
            item_id = str(item.get("item_content", ""))[:50]  # Use prefix as ID
            label = float(item.get("label", 0.5))

            # Track observations per subject
            observations_by_subject[item_subject_id].append((item_id, label))

            # Track labeled items per category
            labeled_by_category[item_benchmark].append(item)

        # Compute subject abilities from labeled data
        abilities = {}
        for subj_id, obs in observations_by_subject.items():
            abilities[subj_id] = _compute_ability(obs)

        # Get current subject ability (use prior if not in labeled)
        z_i = abilities.get(subject_id, 0.0)

        # Fit global calibration on all labeled data
        all_z = []
        all_y = []
        for item in labeled:
            item_subject_id = _extract_subject_id(str(item.get("subject_content", "")))
            item_z = abilities.get(item_subject_id, 0.0)
            item_y = float(item.get("label", 0.5))
            all_z.append(item_z)
            all_y.append(item_y)

        global_rho, global_beta0 = _fit_logistic(
            np.array(all_z),
            np.array(all_y),
            ridge=0.0,
        )

        # Calibrate for current category
        category_anchors = labeled_by_category.get(benchmark, [])
        rho_final, beta0_final = _calibrate_category(
            category_anchors,
            abilities,
            global_rho,
            global_beta0,
        )

        # Predict
        logit = z_i * rho_final + beta0_final
        prob = expit(logit)

        # Clip probability
        prob = float(np.clip(prob, CLIP_LO, CLIP_HI))

        return prob

    except Exception as e:
        # Fallback to neutral prediction on any error
        return 0.5
