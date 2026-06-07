"""Advanced statistical and metadata features for Stage 2 K-factor head.

This module computes additional features beyond text embeddings and one-hot
benchmark/condition encoding. Features include:
  - Benchmark-level statistics (difficulty, item count, length distributions)
  - Condition-level statistics (performance deltas, framing effects)
  - Item complexity features (length, token count)
  - Stage 1 parameter distribution features (z distribution per benchmark)

Design principles:
  - All features computed from training data only (no test leakage)
  - Deterministic and reproducible
  - Graceful handling of unseen benchmark/condition values at test time
  - Compatible with existing pipeline (concat to embedding + side features)
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ADVANCED_FEATURE_VERSION = "stage2_advanced_v1"


def _clean_field(value: Any) -> str:
    """Convert common missing/null sentinels into stable empty strings."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def compute_benchmark_statistics(rows: Iterable[dict]) -> dict[str, dict[str, float]]:
    """Compute benchmark-level statistics from training data.

    Args:
        rows: Training rows with keys: benchmark, condition, item_content, label

    Returns:
        Dictionary mapping benchmark -> statistics dict with keys:
          - avg_correct_rate: mean label (difficulty proxy)
          - item_count: number of training items in this benchmark
          - avg_item_length: mean character count
          - std_item_length: std of character count
          - label_entropy: entropy of label distribution
    """
    benchmark_data = defaultdict(lambda: {
        "labels": [],
        "lengths": [],
    })

    for row in rows:
        benchmark = _clean_field(row.get("benchmark", ""))
        if not benchmark:
            continue

        label = row.get("label")
        if label is not None and label in (0, 1, 0.0, 1.0):
            benchmark_data[benchmark]["labels"].append(float(label))

        item_content = _clean_field(row.get("item_content", ""))
        benchmark_data[benchmark]["lengths"].append(len(item_content))

    stats = {}
    for benchmark, data in benchmark_data.items():
        labels = np.array(data["labels"], dtype=np.float32)
        lengths = np.array(data["lengths"], dtype=np.float32)

        if len(labels) == 0:
            continue

        avg_correct = float(labels.mean())
        # Compute label entropy
        if len(labels) > 0:
            p_pos = avg_correct
            p_neg = 1.0 - p_pos
            entropy = 0.0
            if p_pos > 0:
                entropy -= p_pos * np.log2(p_pos)
            if p_neg > 0:
                entropy -= p_neg * np.log2(p_neg)
        else:
            entropy = 0.0

        stats[benchmark] = {
            "avg_correct_rate": avg_correct,
            "item_count": float(len(labels)),
            "avg_item_length": float(lengths.mean()) if len(lengths) > 0 else 0.0,
            "std_item_length": float(lengths.std()) if len(lengths) > 1 else 0.0,
            "label_entropy": float(entropy),
        }

    return stats


def compute_condition_statistics(rows: Iterable[dict]) -> dict[str, dict[str, float]]:
    """Compute condition-level statistics from training data.

    Args:
        rows: Training rows with keys: benchmark, condition, item_content, label

    Returns:
        Dictionary mapping condition -> statistics dict with keys:
          - avg_correct_rate: mean label across all benchmarks
          - item_count: number of training items with this condition
          - benchmark_count: number of unique benchmarks using this condition
    """
    condition_data = defaultdict(lambda: {
        "labels": [],
        "benchmarks": set(),
    })

    for row in rows:
        condition = _clean_field(row.get("condition", ""))
        if not condition:
            condition = "none"  # Normalize empty condition

        label = row.get("label")
        if label is not None and label in (0, 1, 0.0, 1.0):
            condition_data[condition]["labels"].append(float(label))

        benchmark = _clean_field(row.get("benchmark", ""))
        if benchmark:
            condition_data[condition]["benchmarks"].add(benchmark)

    stats = {}
    for condition, data in condition_data.items():
        labels = np.array(data["labels"], dtype=np.float32)

        if len(labels) == 0:
            continue

        stats[condition] = {
            "avg_correct_rate": float(labels.mean()),
            "item_count": float(len(labels)),
            "benchmark_count": float(len(data["benchmarks"])),
        }

    return stats


def compute_benchmark_condition_interaction_statistics(
    rows: Iterable[dict]
) -> dict[tuple[str, str], dict[str, float]]:
    """Compute interaction statistics for (benchmark, condition) pairs.

    Args:
        rows: Training rows with keys: benchmark, condition, item_content, label

    Returns:
        Dictionary mapping (benchmark, condition) -> statistics dict with keys:
          - avg_correct_rate: mean label for this specific combination
          - item_count: number of training items with this combination
    """
    interaction_data = defaultdict(list)

    for row in rows:
        benchmark = _clean_field(row.get("benchmark", ""))
        condition = _clean_field(row.get("condition", ""))
        if not condition:
            condition = "none"

        label = row.get("label")
        if label is not None and label in (0, 1, 0.0, 1.0):
            key = (benchmark, condition)
            interaction_data[key].append(float(label))

    stats = {}
    for key, labels in interaction_data.items():
        labels_arr = np.array(labels, dtype=np.float32)
        stats[key] = {
            "avg_correct_rate": float(labels_arr.mean()),
            "item_count": float(len(labels)),
        }

    return stats


def compute_stage1_parameter_statistics(
    stage1_dir: Path,
) -> dict[str, dict[str, float]]:
    """Compute statistics from Stage 1 fitted item parameters.

    Args:
        stage1_dir: Path to Stage 1 output directory containing item_targets.pt

    Returns:
        Dictionary with global statistics about item parameters:
          - z_mean: global mean of item difficulty (z)
          - z_std: global std of item difficulty
          - v_norm_mean: mean L2 norm of item loading vectors (V)
          - v_norm_std: std of V norms
    """
    import torch

    try:
        item_targets = torch.load(
            stage1_dir / "item_targets.pt",
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        item_targets = torch.load(stage1_dir / "item_targets.pt", map_location="cpu")

    item_z = item_targets["item_z"].detach().cpu().numpy().astype(np.float32)
    item_v = item_targets["item_v"].detach().cpu().numpy().astype(np.float32)

    v_norms = np.linalg.norm(item_v, axis=1)

    stats = {
        "global": {
            "z_mean": float(item_z.mean()),
            "z_std": float(item_z.std()) if len(item_z) > 1 else 0.0,
            "v_norm_mean": float(v_norms.mean()),
            "v_norm_std": float(v_norms.std()) if len(v_norms) > 1 else 0.0,
        }
    }

    return stats


class AdvancedFeatureExtractor:
    """Extract advanced statistical features for Stage 2 training and inference."""

    def __init__(
        self,
        benchmark_stats: dict[str, dict[str, float]],
        condition_stats: dict[str, dict[str, float]],
        interaction_stats: dict[tuple[str, str], dict[str, float]],
        stage1_stats: dict[str, dict[str, float]],
        feature_names: list[str] | None = None,
    ):
        """Initialize with precomputed statistics.

        Args:
            benchmark_stats: Output of compute_benchmark_statistics()
            condition_stats: Output of compute_condition_statistics()
            interaction_stats: Output of compute_benchmark_condition_interaction_statistics()
            stage1_stats: Output of compute_stage1_parameter_statistics()
            feature_names: List of feature names for this configuration
        """
        self.benchmark_stats = benchmark_stats
        self.condition_stats = condition_stats
        self.interaction_stats = interaction_stats
        self.stage1_stats = stage1_stats

        # Compute global fallback values
        self.global_benchmark_defaults = self._compute_global_defaults(benchmark_stats)
        self.global_condition_defaults = self._compute_global_defaults(condition_stats)

        # Feature dimension
        self.feature_dim = self._compute_feature_dim()
        self.feature_names = feature_names or self._generate_feature_names()

    def _compute_global_defaults(self, stats_dict: dict) -> dict[str, float]:
        """Compute global mean for each statistic key (fallback for unseen values)."""
        if not stats_dict:
            return {}

        # Aggregate all values for each statistic key
        keys = set()
        for stat in stats_dict.values():
            keys.update(stat.keys())

        defaults = {}
        for key in keys:
            values = [stat[key] for stat in stats_dict.values() if key in stat]
            defaults[key] = float(np.mean(values)) if values else 0.0

        return defaults

    def _compute_feature_dim(self) -> int:
        """Compute total feature dimension."""
        # Benchmark features: 5 (avg_correct_rate, item_count, avg_item_length, std_item_length, label_entropy)
        # Condition features: 3 (avg_correct_rate, item_count, benchmark_count)
        # Interaction features: 2 (avg_correct_rate, item_count)
        # Stage 1 global features: 4 (z_mean, z_std, v_norm_mean, v_norm_std)
        # Item-level features: 1 (item_length)
        return 5 + 3 + 2 + 4 + 1

    def _generate_feature_names(self) -> list[str]:
        """Generate human-readable feature names."""
        return [
            "bench_avg_correct",
            "bench_item_count",
            "bench_avg_length",
            "bench_std_length",
            "bench_label_entropy",
            "cond_avg_correct",
            "cond_item_count",
            "cond_benchmark_count",
            "interaction_avg_correct",
            "interaction_item_count",
            "stage1_z_mean",
            "stage1_z_std",
            "stage1_v_norm_mean",
            "stage1_v_norm_std",
            "item_length",
        ]

    def extract(self, row: dict) -> np.ndarray:
        """Extract feature vector for a single row.

        Args:
            row: Dictionary with keys: benchmark, condition, item_content

        Returns:
            Float32 array of shape [feature_dim]
        """
        benchmark = _clean_field(row.get("benchmark", ""))
        condition = _clean_field(row.get("condition", ""))
        if not condition:
            condition = "none"
        item_content = _clean_field(row.get("item_content", ""))

        features = []

        # Benchmark features (5 dims)
        bench_stat = self.benchmark_stats.get(benchmark, self.global_benchmark_defaults)
        features.extend([
            bench_stat.get("avg_correct_rate", self.global_benchmark_defaults.get("avg_correct_rate", 0.5)),
            bench_stat.get("item_count", self.global_benchmark_defaults.get("item_count", 0.0)),
            bench_stat.get("avg_item_length", self.global_benchmark_defaults.get("avg_item_length", 0.0)),
            bench_stat.get("std_item_length", self.global_benchmark_defaults.get("std_item_length", 0.0)),
            bench_stat.get("label_entropy", self.global_benchmark_defaults.get("label_entropy", 1.0)),
        ])

        # Condition features (3 dims)
        cond_stat = self.condition_stats.get(condition, self.global_condition_defaults)
        features.extend([
            cond_stat.get("avg_correct_rate", self.global_condition_defaults.get("avg_correct_rate", 0.5)),
            cond_stat.get("item_count", self.global_condition_defaults.get("item_count", 0.0)),
            cond_stat.get("benchmark_count", self.global_condition_defaults.get("benchmark_count", 0.0)),
        ])

        # Interaction features (2 dims)
        interaction_key = (benchmark, condition)
        interaction_stat = self.interaction_stats.get(interaction_key, {})
        features.extend([
            interaction_stat.get("avg_correct_rate", 0.5),  # Fallback to neutral
            interaction_stat.get("item_count", 0.0),
        ])

        # Stage 1 global features (4 dims)
        global_stats = self.stage1_stats.get("global", {})
        features.extend([
            global_stats.get("z_mean", 0.0),
            global_stats.get("z_std", 1.0),
            global_stats.get("v_norm_mean", 1.0),
            global_stats.get("v_norm_std", 0.0),
        ])

        # Item-level features (1 dim)
        features.append(float(len(item_content)))

        return np.array(features, dtype=np.float32)

    def save(self, path: Path) -> None:
        """Save feature extractor state to JSON."""
        state = {
            "version": ADVANCED_FEATURE_VERSION,
            "feature_dim": self.feature_dim,
            "feature_names": self.feature_names,
            "benchmark_stats": self.benchmark_stats,
            "condition_stats": self.condition_stats,
            "interaction_stats": {
                f"{k[0]}::{k[1]}": v for k, v in self.interaction_stats.items()
            },
            "stage1_stats": self.stage1_stats,
            "global_benchmark_defaults": self.global_benchmark_defaults,
            "global_condition_defaults": self.global_condition_defaults,
        }
        path.write_text(json.dumps(state, indent=2))

    @classmethod
    def load(cls, path: Path) -> AdvancedFeatureExtractor:
        """Load feature extractor from JSON."""
        state = json.loads(path.read_text())

        if state.get("version") != ADVANCED_FEATURE_VERSION:
            raise ValueError(
                f"Incompatible feature version: {state.get('version')} "
                f"(expected {ADVANCED_FEATURE_VERSION})"
            )

        # Convert interaction keys back to tuples
        interaction_stats = {
            tuple(k.split("::")): v for k, v in state["interaction_stats"].items()
        }

        extractor = cls(
            benchmark_stats=state["benchmark_stats"],
            condition_stats=state["condition_stats"],
            interaction_stats=interaction_stats,
            stage1_stats=state["stage1_stats"],
            feature_names=state.get("feature_names"),
        )
        extractor.global_benchmark_defaults = state["global_benchmark_defaults"]
        extractor.global_condition_defaults = state["global_condition_defaults"]

        return extractor


def build_advanced_feature_extractor(
    training_rows: Iterable[dict],
    stage1_dir: Path,
) -> AdvancedFeatureExtractor:
    """Build feature extractor from training data and Stage 1 outputs.

    Args:
        training_rows: Iterable of training rows (must be replayable)
        stage1_dir: Path to Stage 1 output directory

    Returns:
        Configured AdvancedFeatureExtractor ready for use
    """
    # Convert to list if needed (we need to iterate multiple times)
    rows_list = list(training_rows)

    benchmark_stats = compute_benchmark_statistics(rows_list)
    condition_stats = compute_condition_statistics(rows_list)
    interaction_stats = compute_benchmark_condition_interaction_statistics(rows_list)
    stage1_stats = compute_stage1_parameter_statistics(stage1_dir)

    return AdvancedFeatureExtractor(
        benchmark_stats=benchmark_stats,
        condition_stats=condition_stats,
        interaction_stats=interaction_stats,
        stage1_stats=stage1_stats,
    )
