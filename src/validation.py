"""Local validation harness.

Two cold-start splits:
  - item_cold_start_split  — hold out a random fraction of items entirely
  - benchmark_cold_start_split — hold out an entire benchmark's items

Scoring helpers compute the leaderboard metric (mean log-likelihood with light
clipping to avoid log(0)) and AUC-ROC.

Usage:
    from src.validation import (
        item_cold_start_split,
        benchmark_cold_start_split,
        score,
    )

    train, val = item_cold_start_split(rows, val_frac=0.1, seed=0)
    metrics = score(predictions, [r["label"] for r in val])
    # metrics = {"mean_log_likelihood": ..., "auc_roc": ...}
"""

from __future__ import annotations

import math
import random
from typing import Iterable, Sequence

EPS = 1e-6


def _hash_to_unit(s: str, seed: int) -> float:
    """Deterministic [0,1) hash of (s, seed). Used for reproducible item splits."""
    import hashlib

    h = hashlib.sha256(f"{seed}|{s}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def item_cold_start_split(
    rows: Sequence[dict],
    val_frac: float = 0.1,
    seed: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Hold out a random subset of items entirely; train and val never share items."""
    item_ids = {r["item_id"] for r in rows}
    held_out = {iid for iid in item_ids if _hash_to_unit(iid, seed) < val_frac}
    train = [r for r in rows if r["item_id"] not in held_out]
    val = [r for r in rows if r["item_id"] in held_out]
    return train, val


def benchmark_cold_start_split(
    rows: Sequence[dict],
    held_out_benchmarks: Iterable[str],
) -> tuple[list[dict], list[dict]]:
    """Hold out one or more benchmarks entirely. Stress-tests transfer."""
    held = set(held_out_benchmarks)
    train = [r for r in rows if r["benchmark"] not in held]
    val = [r for r in rows if r["benchmark"] in held]
    return train, val


def mean_log_likelihood(predictions: Sequence[float], labels: Sequence[int]) -> float:
    """Mean log-likelihood of binary labels under predicted probabilities.

    Higher is better — matches the leaderboard primary metric. Predictions are
    clipped to [EPS, 1-EPS] to avoid log(0) on miscalibrated outputs.
    """
    if len(predictions) != len(labels):
        raise ValueError(f"len mismatch: {len(predictions)} vs {len(labels)}")
    total = 0.0
    for p, y in zip(predictions, labels):
        p_clipped = min(max(float(p), EPS), 1.0 - EPS)
        total += math.log(p_clipped if y == 1 else 1.0 - p_clipped)
    return total / len(predictions)


def auc_roc(predictions: Sequence[float], labels: Sequence[int]) -> float:
    """ROC-AUC by Mann-Whitney U over rank-sums. O(n log n)."""
    n = len(predictions)
    if n != len(labels):
        raise ValueError(f"len mismatch: {n} vs {len(labels)}")
    pos = sum(1 for y in labels if y == 1)
    neg = n - pos
    if pos == 0 or neg == 0:
        return float("nan")

    indexed = sorted(enumerate(predictions), key=lambda ix: ix[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1

    rank_sum_pos = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (rank_sum_pos - pos * (pos + 1) / 2) / (pos * neg)


def score(predictions: Sequence[float], labels: Sequence[int]) -> dict[str, float]:
    return {
        "mean_log_likelihood": mean_log_likelihood(predictions, labels),
        "auc_roc": auc_roc(predictions, labels),
        "n": len(predictions),
        "p_pos": sum(labels) / len(labels) if labels else float("nan"),
    }
