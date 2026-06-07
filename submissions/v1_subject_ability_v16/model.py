"""
Subject-Ability Prediction v16: More Aggressive Cross-Category Transfer

V15 results: MLL = -0.6723 (improved from v10's -0.6973)
- Global calibration is working!
- But still too conservative (30% ability, 70% benchmark rate for unlabeled)

V13 changes:
- Increase confidence in ability-based predictions for unlabeled categories
- From 90% to 99% for unlabeled, from slower buildup to faster for labeled
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

# Configuration
CLIP_LO, CLIP_HI = 0.02, 0.98
LAPLACE_ALPHA = 0.5
MIN_OBSERVATIONS = 2

# Prior strength
PRIOR_STRENGTH = 10.0
PRIOR_ABILITY = 0.0

NAME_PATTERN = re.compile(
    r"^\s*(?:Name|Subject)\s*:\s*(.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def _extract_subject_id(subject_content: str) -> str:
    """Extract subject ID from subject_content."""
    match = NAME_PATTERN.search(subject_content)
    if match:
        return match.group(1).strip()
    lines = subject_content.strip().split("\n")
    if lines:
        return lines[0].strip()
    return subject_content.strip()


def _safe_logit(p: float, eps: float = 1e-10) -> float:
    """Numerically stable logit."""
    p = np.clip(p, eps, 1 - eps)
    return float(np.log(p / (1 - p)))


def _update_ability_incremental(
    prior_ability: float,
    prior_strength: float,
    new_observations: list[float],
) -> float:
    """Incrementally update ability using Bayesian approach."""
    if len(new_observations) == 0:
        return prior_ability

    # Convert prior to pseudo-counts
    prior_acc = expit(prior_ability)
    prior_correct = prior_acc * prior_strength
    prior_total = prior_strength

    # Add new observations
    new_correct = sum(new_observations)
    new_total = len(new_observations)

    # Combine with Laplace smoothing
    total_correct = prior_correct + new_correct + LAPLACE_ALPHA
    total_count = prior_total + new_total + 2 * LAPLACE_ALPHA

    acc = total_correct / total_count
    z_i = _safe_logit(acc)

    return float(z_i)


def _fit_calibration(
    z_values: np.ndarray,
    y_values: np.ndarray,
    base_rate: float,
    conservative: bool = True,
) -> tuple[float, float]:
    """
    Fit calibration parameters.

    Args:
        z_values: Subject abilities
        y_values: Labels
        base_rate: Base rate for initialization
        conservative: Whether to use conservative bounds

    Returns:
        (rho, beta0) calibration parameters
    """
    if len(z_values) == 0:
        return 0.8, _safe_logit(base_rate)

    # Start conservative
    rho = 0.8
    beta0 = _safe_logit(base_rate)

    # Fit if we have enough data
    if len(z_values) >= 3:
        try:
            def objective(params):
                r, b = params
                logits = r * z_values + b
                probs = expit(logits)
                # NLL with small ridge
                nll = -np.sum(
                    y_values * np.log(probs + 1e-10)
                    + (1 - y_values) * np.log(1 - probs + 1e-10)
                )
                # Light penalty toward conservative values
                nll += 0.1 * (r - 0.8) ** 2
                return nll

            bounds = [(0.1, 3.0), (None, None)] if conservative else [(0.05, 10.0), (None, None)]

            result = minimize(
                objective,
                x0=[rho, beta0],
                method="L-BFGS-B",
                bounds=bounds,
            )
            if result.success:
                rho, beta0 = result.x
        except Exception:
            pass

    return float(rho), float(beta0)


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """
    Predict with improved cross-category transfer.

    Key change: Always use subject abilities, even for unlabeled categories.
    Use global calibration when no category-specific data available.
    """
    try:
        subject_id = _extract_subject_id(str(input.get("subject_content", "")))
        benchmark = str(input.get("benchmark", ""))

        if not labeled or len(labeled) == 0:
            return 0.5

        # Step 1: Compute subject abilities from ALL labeled data
        global_correct = sum(float(item.get("label", 0.5)) for item in labeled)
        global_total = len(labeled)
        global_rate = global_correct / global_total if global_total > 0 else 0.5

        # Per-benchmark base rates
        benchmark_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        for item in labeled:
            item_benchmark = str(item.get("benchmark", ""))
            label = float(item.get("label", 0.5))
            benchmark_stats[item_benchmark]["correct"] += label
            benchmark_stats[item_benchmark]["total"] += 1

        if benchmark in benchmark_stats:
            bench_stats = benchmark_stats[benchmark]
            benchmark_rate = bench_stats["correct"] / bench_stats["total"] if bench_stats["total"] > 0 else global_rate
        else:
            benchmark_rate = global_rate

        # Compute subject abilities
        observations_by_subject = defaultdict(list)
        for item in labeled:
            item_subject_id = _extract_subject_id(str(item.get("subject_content", "")))
            label = float(item.get("label", 0.5))
            observations_by_subject[item_subject_id].append(label)

        abilities = {}
        for subj_id, obs in observations_by_subject.items():
            abilities[subj_id] = _update_ability_incremental(
                prior_ability=PRIOR_ABILITY,
                prior_strength=PRIOR_STRENGTH,
                new_observations=obs,
            )

        # Get current subject ability
        z_i = abilities.get(subject_id, PRIOR_ABILITY)

        # Step 2: Prepare calibration data
        # Category-specific calibration data
        category_z = []
        category_y = []
        for item in labeled:
            if str(item.get("benchmark", "")) == benchmark:
                item_subject_id = _extract_subject_id(str(item.get("subject_content", "")))
                if item_subject_id in abilities:
                    item_z = abilities[item_subject_id]
                    item_y = float(item.get("label", 0.5))
                    category_z.append(item_z)
                    category_y.append(item_y)

        # Global calibration data (ALL labeled data)
        global_z = []
        global_y = []
        for item in labeled:
            item_subject_id = _extract_subject_id(str(item.get("subject_content", "")))
            if item_subject_id in abilities:
                item_z = abilities[item_subject_id]
                item_y = float(item.get("label", 0.5))
                global_z.append(item_z)
                global_y.append(item_y)

        # NEW: Fit global calibration first
        global_rho, global_beta0 = _fit_calibration(
            np.array(global_z),
            np.array(global_y),
            global_rate,
            conservative=False,  # Can be less conservative for global
        )

        # Fit category-specific calibration if we have data
        if len(category_z) >= 3:
            # Use category-specific calibration
            rho, beta0 = _fit_calibration(
                np.array(category_z),
                np.array(category_y),
                benchmark_rate,
                conservative=True,
            )

            # Mix category and global calibration based on data amount
            category_weight = min(len(category_z) / 10.0, 1.0)
            rho = category_weight * rho + (1 - category_weight) * global_rho
            beta0 = category_weight * beta0 + (1 - category_weight) * global_beta0

            # Use calibrated prediction with moderate mixing
            logit_pred = z_i * rho + beta0
            prob = expit(logit_pred)
            prob = float(np.clip(prob, CLIP_LO, CLIP_HI))

            # Mix with benchmark rate (faster buildup than v12)
            confidence = min(len(category_z) / 12.0, 1.0)  # Faster than v12's 15.0
            prob = confidence * prob + (1 - confidence) * benchmark_rate

        else:
            # NEW: For unlabeled categories, use GLOBAL calibration with subject abilities
            # Instead of ignoring abilities and returning benchmark_rate!
            logit_pred = z_i * global_rho + global_beta0
            prob = expit(logit_pred)
            prob = float(np.clip(prob, CLIP_LO, CLIP_HI))

            # More aggressive: Use 50% ability-based prediction for unlabeled
            confidence = 0.99  # Increased from v15's 0.9
            prob = confidence * prob + (1 - confidence) * benchmark_rate

        # Final clip
        prob = float(np.clip(prob, CLIP_LO, CLIP_HI))

        return prob

    except Exception as e:
        return 0.5
