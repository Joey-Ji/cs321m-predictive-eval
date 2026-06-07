"""Advanced features v3: Reduced feature set with stronger regularization.

Changes from v2:
- Reduced from 15 features to 6 (keep only the most predictive)
- Remove count and std features (high risk of overfitting)
- Keep only rate-based and length features
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ADVANCED_FEATURE_VERSION = "stage2_advanced_v3"


def _clean_field(value: Any) -> str:
    """Convert common missing/null sentinels into stable empty strings."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def compute_benchmark_statistics(rows: Iterable[dict]) -> dict[str, dict[str, float]]:
    """Compute benchmark-level statistics from training data.

    V3: Only compute avg_correct_rate (most predictive)
    """
    benchmark_data = defaultdict(list)

    for row in rows:
        benchmark = _clean_field(row.get("benchmark", ""))
        if not benchmark:
            continue

        label = row.get("label")
        if label is not None and label in (0, 1, 0.0, 1.0):
            benchmark_data[benchmark].append(float(label))

    stats = {}
    for benchmark, labels in benchmark_data.items():
        if len(labels) > 0:
            stats[benchmark] = {
                "avg_correct_rate": float(np.mean(labels)),
            }

    return stats


def compute_condition_statistics(rows: Iterable[dict]) -> dict[str, dict[str, float]]:
    """Compute condition-level statistics from training data.

    V3: Only compute avg_correct_rate (most predictive)
    """
    condition_data = defaultdict(list)

    for row in rows:
        condition = _clean_field(row.get("condition", ""))
        if not condition:
            condition = "none"

        label = row.get("label")
        if label is not None and label in (0, 1, 0.0, 1.0):
            condition_data[condition].append(float(label))

    stats = {}
    for condition, labels in condition_data.items():
        if len(labels) > 0:
            stats[condition] = {
                "avg_correct_rate": float(np.mean(labels)),
            }

    return stats


def compute_benchmark_condition_interaction_statistics(
    rows: Iterable[dict]
) -> dict[tuple[str, str], dict[str, float]]:
    """Compute interaction statistics for (benchmark, condition) pairs.

    V3: Only compute avg_correct_rate (most predictive)
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
        if len(labels) > 0:
            stats[key] = {
                "avg_correct_rate": float(np.mean(labels)),
            }

    return stats


def compute_stage1_parameter_statistics(
    stage1_dir: Path,
) -> dict[str, dict[str, float]]:
    """Compute statistics from Stage 1 fitted item parameters.

    V3: Only compute z_mean (item difficulty global mean)
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

    stats = {
        "global": {
            "z_mean": float(item_z.mean()),
        }
    }

    return stats


class AdvancedFeatureExtractor:
    """Extract advanced statistical features for Stage 2 training and inference.

    V3: Reduced to 6 features (from 15):
    1. bench_avg_correct
    2. cond_avg_correct
    3. interaction_avg_correct
    4. stage1_z_mean
    5. item_length
    6. item_length_log (log-scaled length for better distribution)
    """

    def __init__(
        self,
        benchmark_stats: dict[str, dict[str, float]],
        condition_stats: dict[str, dict[str, float]],
        interaction_stats: dict[tuple[str, str], dict[str, float]],
        stage1_stats: dict[str, dict[str, float]],
        feature_names: list[str] | None = None,
    ):
        """Initialize with precomputed statistics."""
        self.benchmark_stats = benchmark_stats
        self.condition_stats = condition_stats
        self.interaction_stats = interaction_stats
        self.stage1_stats = stage1_stats

        # Compute global fallback values
        self.global_benchmark_defaults = self._compute_global_defaults(benchmark_stats)
        self.global_condition_defaults = self._compute_global_defaults(condition_stats)

        # Feature dimension
        self.feature_dim = 6  # Reduced from 15
        self.feature_names = feature_names or self._generate_feature_names()

    def _compute_global_defaults(self, stats_dict: dict) -> dict[str, float]:
        """Compute global mean for each statistic key (fallback for unseen values)."""
        if not stats_dict:
            return {}

        keys = set()
        for stat in stats_dict.values():
            keys.update(stat.keys())

        defaults = {}
        for key in keys:
            values = [stat[key] for stat in stats_dict.values() if key in stat]
            defaults[key] = float(np.mean(values)) if values else 0.5  # Default to 0.5 for rates

        return defaults

    def _generate_feature_names(self) -> list[str]:
        """Generate human-readable feature names."""
        return [
            "bench_avg_correct",
            "cond_avg_correct",
            "interaction_avg_correct",
            "stage1_z_mean",
            "item_length",
            "item_length_log",
        ]

    def extract(self, row: dict) -> np.ndarray:
        """Extract feature vector for a single row.

        Args:
            row: Dictionary with keys: benchmark, condition, item_content

        Returns:
            Float32 array of shape [6]
        """
        benchmark = _clean_field(row.get("benchmark", ""))
        condition = _clean_field(row.get("condition", ""))
        if not condition:
            condition = "none"
        item_content = _clean_field(row.get("item_content", ""))

        features = []

        # 1. Benchmark avg_correct_rate
        bench_stat = self.benchmark_stats.get(benchmark, self.global_benchmark_defaults)
        features.append(
            bench_stat.get("avg_correct_rate", self.global_benchmark_defaults.get("avg_correct_rate", 0.5))
        )

        # 2. Condition avg_correct_rate
        cond_stat = self.condition_stats.get(condition, self.global_condition_defaults)
        features.append(
            cond_stat.get("avg_correct_rate", self.global_condition_defaults.get("avg_correct_rate", 0.5))
        )

        # 3. Interaction avg_correct_rate
        interaction_key = (benchmark, condition)
        interaction_stat = self.interaction_stats.get(interaction_key, {})
        features.append(interaction_stat.get("avg_correct_rate", 0.5))

        # 4. Stage 1 z_mean (global item difficulty)
        global_stats = self.stage1_stats.get("global", {})
        features.append(global_stats.get("z_mean", 0.0))

        # 5. Item length (raw)
        item_len = float(len(item_content))
        features.append(item_len)

        # 6. Item length (log-scaled for better distribution)
        features.append(np.log1p(item_len))  # log(1 + length) to handle length=0

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
    """Build feature extractor from training data and Stage 1 outputs."""
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
