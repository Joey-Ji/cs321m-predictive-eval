"""Tests for scalar item residual helpers.

Run:
    python tests/test_item_residual.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from item_residual_utils import (  # noqa: E402
    RuntimeCanonicalizer,
    fit_locked_priors,
    fit_weighted_ridge,
    item_fold_map,
    pearson_corr,
    predict_weighted_ridge,
    prior_probs_for_frame,
)


def _fake_runtime_priors() -> dict:
    sep = "\x1f"
    return {
        "global": 0.5,
        "kappas": {
            "benchmark": 2.0,
            "benchmark_condition": 2.0,
            "subject": 2.0,
            "subject_benchmark": 2.0,
            "subject_category": 2.0,
        },
        "benchmark": {"ai2d_test": 0.6},
        "benchmark_condition": {f"ai2d_test{sep}none": 0.55},
        "subject": {"name: model-a": 0.7},
        "subject_benchmark": {f"name: model-a{sep}ai2d_test": 0.75},
        "subject_category": {f"name: model-a{sep}ai2d_test{sep}none": 0.8},
        "key_sep": sep,
    }


def test_runtime_canonicalizer_aliases() -> None:
    canon = RuntimeCanonicalizer.from_runtime_priors(_fake_runtime_priors())
    assert canon.normalize_benchmark("AI2D-TEST") == "ai2d_test"
    assert canon.normalize_benchmark("ai2d test") == "ai2d_test"
    assert canon.normalize_condition("None") == "none"
    assert canon.normalize_condition("n/a") == "none"


def test_item_fold_map_is_item_disjoint_and_stable() -> None:
    item_ids = ["a", "b", "c", "a", "d", "e"]
    folds = item_fold_map(item_ids, n_folds=5, seed=123)
    assert set(folds) == {"a", "b", "c", "d", "e"}
    assert folds == item_fold_map(list(reversed(item_ids)), n_folds=5, seed=123)
    assert all(0 <= fold < 5 for fold in folds.values())


def test_locked_prior_lookup_falls_through_hierarchy() -> None:
    rows = pd.DataFrame(
        [
            {"subject_key": "s1", "benchmark_key": "b1", "condition_key": "c1", "label": 1},
            {"subject_key": "s1", "benchmark_key": "b1", "condition_key": "c1", "label": 1},
            {"subject_key": "s2", "benchmark_key": "b1", "condition_key": "c1", "label": 0},
            {"subject_key": "s2", "benchmark_key": "b2", "condition_key": "c2", "label": 0},
        ]
    )
    kappas = {
        "benchmark": 2.0,
        "benchmark_condition": 2.0,
        "subject": 2.0,
        "subject_benchmark": 2.0,
        "subject_category": 2.0,
    }
    priors = fit_locked_priors(rows, kappas)
    probs = prior_probs_for_frame(rows, priors)
    assert probs.shape == (4,)
    assert np.isfinite(probs).all()
    assert probs[0] > probs[2]
    unseen = pd.DataFrame(
        [{"subject_key": "unknown", "benchmark_key": "unknown", "condition_key": "unknown", "label": 1}]
    )
    assert float(prior_probs_for_frame(unseen, priors)[0]) == priors.global_p


def test_weighted_ridge_recovers_linear_signal() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 5)).astype(np.float32)
    coef = np.array([0.3, -0.2, 0.1, 0.0, 0.4], dtype=np.float32)
    y = x @ coef + 0.05
    weight = np.linspace(0.5, 2.0, len(x)).astype(np.float32)
    model = fit_weighted_ridge(x, y, weight, alpha=0.01)
    pred = predict_weighted_ridge(x, model)
    assert pearson_corr(pred, y) > 0.999
    assert float(np.sqrt(np.mean((pred - y) ** 2))) < 0.01


def main() -> int:
    tests = [
        test_runtime_canonicalizer_aliases,
        test_item_fold_map_is_item_disjoint_and_stable,
        test_locked_prior_lookup_falls_through_hierarchy,
        test_weighted_ridge_recovers_linear_signal,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

