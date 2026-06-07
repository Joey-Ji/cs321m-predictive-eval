"""
Subject-ability predictor (pure online inference, CPU-only, no pretrained weights).

For each (subject, item, benchmark, condition) query, estimate the subject's
ability from whatever anchor labels are revealed this round, then map ability to
a pass probability via a per-benchmark logistic calibration:

  1. Laplace-smoothed accuracy per subject -> ability logit z_i, shrunk toward the
     group's (trimmed-mean) ability with strength ~1/(1 + n/SHRINK_N0).
  2. Per-benchmark logistic calibration p = sigmoid(rho * z_i + beta0), fit by
     penalized NLL on the in-category anchors.
  3. Blend toward the benchmark base rate until enough in-category anchors are seen.

All state is derived from the `labeled` argument at call time; nothing is loaded
from disk or over the network.
"""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

# Configuration
CLIP_LO, CLIP_HI = 0.02, 0.98     # output probability is clamped to this range
LAPLACE_ALPHA = 0.3               # Laplace smoothing pseudo-count for subject accuracy
MIN_OBSERVATIONS = 2              # min anchors before a subject's ability is trusted
SHRINK_N0 = 4.0                   # pseudo-count for shrinking ability toward the group mean
MIN_FIT_OBSERVATIONS = 4          # min in-category anchors before fitting per-benchmark calibration
CONFIDENCE_SCALE = 8.0            # in-category anchors needed to fully trust the calibrated prob

NAME_PATTERN = re.compile(
    r"^\s*(?:Name|Subject)\s*:\s*(.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def _extract_subject_id(subject_content: str) -> str:
    """Extract a stable subject id from subject_content."""
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


def _compute_ability(observations: list[tuple[str, float]], global_mean: float = 0.0) -> float:
    """
    Compute subject ability (logit) with shrinkage toward the group mean.

    Args:
        observations: List of (item_id, correct) tuples
        global_mean: Global mean ability across subjects

    Returns:
        Ability logit z_i
    """
    if len(observations) < MIN_OBSERVATIONS:
        return global_mean

    correct_count = sum(correct for _, correct in observations)
    total_count = len(observations)

    # Laplace-smoothed accuracy -> ability logit
    acc = (correct_count + LAPLACE_ALPHA) / (total_count + 2 * LAPLACE_ALPHA)
    z_i = _safe_logit(acc)

    # Shrink toward the group mean; shrinkage weakens as observations accumulate
    shrinkage = 1.0 / (1.0 + total_count / SHRINK_N0)
    z_i = (1 - shrinkage) * z_i + shrinkage * global_mean

    return float(z_i)


def _fit_calibration_moderate(
    z_values: np.ndarray,
    y_values: np.ndarray,
    category_base_rate: float,
) -> tuple[float, float]:
    """
    Fit a per-benchmark logistic calibration p = sigmoid(rho * z + beta0).

    Args:
        z_values: Subject abilities (logits) for the in-category anchors
        y_values: Binary pass/fail labels
        category_base_rate: Empirical accuracy for this benchmark

    Returns:
        (rho, beta0) parameters
    """
    if len(z_values) == 0:
        return 1.0, _safe_logit(category_base_rate)

    observed_rate = float(np.mean(y_values))

    # Initial guess
    rho = 0.9
    beta0 = _safe_logit(observed_rate)

    # Only fit when there are enough in-category anchors to be stable
    if len(z_values) >= MIN_FIT_OBSERVATIONS:
        try:
            def objective(params):
                r, b = params
                logits = r * z_values + b
                probs = expit(logits)
                nll = -np.sum(
                    y_values * np.log(probs + 1e-10)
                    + (1 - y_values) * np.log(1 - probs + 1e-10)
                )
                # Ridge penalty keeping the slope near 1.0
                nll += 0.05 * (r - 1.0) ** 2
                return nll

            result = minimize(
                objective,
                x0=[0.9, beta0],
                method="L-BFGS-B",
                bounds=[(0.2, 4.0), (None, None)],
            )
            if result.success:
                rho, beta0 = result.x
        except Exception:
            pass

    return float(rho), float(beta0)


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """
    Predict the probability that the subject passes the item.

    Args:
        input: Dict with keys: subject_content, item_content, benchmark, condition
        labeled: Optional list of revealed anchor labels (same four keys plus "label")

    Returns:
        Probability in [CLIP_LO, CLIP_HI]
    """
    try:
        subject_id = _extract_subject_id(str(input.get("subject_content", "")))
        benchmark = str(input.get("benchmark", ""))

        if not labeled or len(labeled) == 0:
            return 0.5

        # Global statistics
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

        # Group observations by subject
        observations_by_subject = defaultdict(list)
        for item in labeled:
            item_subject_id = _extract_subject_id(str(item.get("subject_content", "")))
            item_id = str(item.get("item_content", ""))[:50]
            label = float(item.get("label", 0.5))
            observations_by_subject[item_subject_id].append((item_id, label))

        # Global mean ability (trimmed mean for robustness)
        all_abilities = []
        for subj_id, obs in observations_by_subject.items():
            if len(obs) >= MIN_OBSERVATIONS:
                correct = sum(label for _, label in obs)
                acc = correct / len(obs)
                all_abilities.append(_safe_logit(acc))

        # Trimmed mean (drop ~10% extremes on each side) once there are enough subjects
        if len(all_abilities) > 4:
            all_abilities_sorted = sorted(all_abilities)
            trim_count = max(1, len(all_abilities) // 10)
            trimmed = all_abilities_sorted[trim_count:-trim_count]
            global_mean_ability = float(np.mean(trimmed))
        elif all_abilities:
            global_mean_ability = float(np.mean(all_abilities))
        else:
            global_mean_ability = 0.0

        # Compute subject abilities
        abilities = {}
        for subj_id, obs in observations_by_subject.items():
            abilities[subj_id] = _compute_ability(obs, global_mean_ability)

        # Current subject ability
        z_i = abilities.get(subject_id, global_mean_ability)

        # Calibration data: in-category anchors only
        category_z = []
        category_y = []
        for item in labeled:
            if str(item.get("benchmark", "")) == benchmark:
                item_subject_id = _extract_subject_id(str(item.get("subject_content", "")))
                item_z = abilities.get(item_subject_id, global_mean_ability)
                item_y = float(item.get("label", 0.5))
                category_z.append(item_z)
                category_y.append(item_y)

        # Fit calibration
        if len(category_z) > 0:
            rho, beta0 = _fit_calibration_moderate(
                np.array(category_z),
                np.array(category_y),
                benchmark_rate,
            )
        else:
            rho = 0.9
            beta0 = _safe_logit(benchmark_rate)

        # Predict
        logit_pred = z_i * rho + beta0
        prob = expit(logit_pred)

        # Blend toward the benchmark base rate until enough in-category anchors are seen
        confidence = min(len(category_z) / CONFIDENCE_SCALE, 1.0)
        prob = confidence * prob + (1 - confidence) * benchmark_rate

        # Final clip
        prob = float(np.clip(prob, CLIP_LO, CLIP_HI))

        return prob

    except Exception:
        return 0.5
