"""Unit tests for advanced feature extraction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.advanced_features import (
    AdvancedFeatureExtractor,
    compute_benchmark_condition_interaction_statistics,
    compute_benchmark_statistics,
    compute_condition_statistics,
)


def test_benchmark_statistics():
    """Test benchmark-level statistics computation."""
    rows = [
        {"benchmark": "math", "condition": "zero-shot", "item_content": "What is 2+2?", "label": 1},
        {"benchmark": "math", "condition": "few-shot", "item_content": "What is 3+3?", "label": 1},
        {"benchmark": "math", "condition": "zero-shot", "item_content": "What is 100*100?", "label": 0},
        {"benchmark": "coding", "condition": "zero-shot", "item_content": "Write a function", "label": 0},
        {"benchmark": "coding", "condition": "zero-shot", "item_content": "Debug this code", "label": 0},
    ]

    stats = compute_benchmark_statistics(rows)

    assert "math" in stats
    assert "coding" in stats

    # Math: 2 correct out of 3
    assert abs(stats["math"]["avg_correct_rate"] - 2.0/3.0) < 0.01
    assert stats["math"]["item_count"] == 3.0

    # Coding: 0 correct out of 2
    assert stats["coding"]["avg_correct_rate"] == 0.0
    assert stats["coding"]["item_count"] == 2.0

    # Check length statistics
    assert stats["math"]["avg_item_length"] > 0
    assert stats["coding"]["avg_item_length"] > 0

    print("✓ Benchmark statistics test passed")


def test_condition_statistics():
    """Test condition-level statistics computation."""
    rows = [
        {"benchmark": "math", "condition": "zero-shot", "item_content": "Q1", "label": 1},
        {"benchmark": "math", "condition": "zero-shot", "item_content": "Q2", "label": 0},
        {"benchmark": "coding", "condition": "zero-shot", "item_content": "Q3", "label": 1},
        {"benchmark": "math", "condition": "few-shot", "item_content": "Q4", "label": 1},
    ]

    stats = compute_condition_statistics(rows)

    assert "zero-shot" in stats
    assert "few-shot" in stats

    # Zero-shot: 2 correct out of 3
    assert abs(stats["zero-shot"]["avg_correct_rate"] - 2.0/3.0) < 0.01
    assert stats["zero-shot"]["item_count"] == 3.0
    assert stats["zero-shot"]["benchmark_count"] == 2.0  # math and coding

    # Few-shot: 1 correct out of 1
    assert stats["few-shot"]["avg_correct_rate"] == 1.0
    assert stats["few-shot"]["item_count"] == 1.0
    assert stats["few-shot"]["benchmark_count"] == 1.0

    print("✓ Condition statistics test passed")


def test_interaction_statistics():
    """Test benchmark-condition interaction statistics."""
    rows = [
        {"benchmark": "math", "condition": "zero-shot", "label": 1},
        {"benchmark": "math", "condition": "zero-shot", "label": 1},
        {"benchmark": "math", "condition": "few-shot", "label": 0},
        {"benchmark": "coding", "condition": "zero-shot", "label": 0},
    ]

    stats = compute_benchmark_condition_interaction_statistics(rows)

    assert ("math", "zero-shot") in stats
    assert ("math", "few-shot") in stats
    assert ("coding", "zero-shot") in stats

    # Math + zero-shot: 2/2 correct
    assert stats[("math", "zero-shot")]["avg_correct_rate"] == 1.0
    assert stats[("math", "zero-shot")]["item_count"] == 2.0

    # Math + few-shot: 0/1 correct
    assert stats[("math", "few-shot")]["avg_correct_rate"] == 0.0

    print("✓ Interaction statistics test passed")


def test_feature_extractor():
    """Test full feature extraction pipeline."""
    rows = [
        {"benchmark": "math", "condition": "zero-shot", "item_content": "What is 2+2?", "label": 1},
        {"benchmark": "math", "condition": "zero-shot", "item_content": "What is 3+3?", "label": 0},
        {"benchmark": "coding", "condition": "few-shot", "item_content": "Write code", "label": 1},
    ]

    # Build statistics
    bench_stats = compute_benchmark_statistics(rows)
    cond_stats = compute_condition_statistics(rows)
    interaction_stats = compute_benchmark_condition_interaction_statistics(rows)
    stage1_stats = {"global": {"z_mean": 0.0, "z_std": 1.0, "v_norm_mean": 1.0, "v_norm_std": 0.5}}

    extractor = AdvancedFeatureExtractor(
        benchmark_stats=bench_stats,
        condition_stats=cond_stats,
        interaction_stats=interaction_stats,
        stage1_stats=stage1_stats,
    )

    # Test known benchmark/condition
    test_row = {"benchmark": "math", "condition": "zero-shot", "item_content": "Test question?"}
    features = extractor.extract(test_row)

    assert features.shape == (extractor.feature_dim,)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()

    # Check that we get benchmark stats
    assert features[0] == bench_stats["math"]["avg_correct_rate"]  # bench_avg_correct

    # Test unseen benchmark (should use fallback)
    unseen_row = {"benchmark": "unseen", "condition": "unseen", "item_content": "?"}
    features_unseen = extractor.extract(unseen_row)

    assert features_unseen.shape == (extractor.feature_dim,)
    assert np.isfinite(features_unseen).all()

    print(f"✓ Feature extractor test passed (dim={extractor.feature_dim})")
    print(f"  Feature names: {extractor.feature_names}")
    print(f"  Sample features (known): {features[:5]}")
    print(f"  Sample features (unseen): {features_unseen[:5]}")


def test_feature_extractor_serialization(tmp_path):
    """Test saving and loading feature extractor."""
    rows = [
        {"benchmark": "math", "condition": "zero-shot", "item_content": "Q", "label": 1},
    ]

    bench_stats = compute_benchmark_statistics(rows)
    cond_stats = compute_condition_statistics(rows)
    interaction_stats = compute_benchmark_condition_interaction_statistics(rows)
    stage1_stats = {"global": {"z_mean": 0.0, "z_std": 1.0, "v_norm_mean": 1.0, "v_norm_std": 0.5}}

    extractor = AdvancedFeatureExtractor(
        benchmark_stats=bench_stats,
        condition_stats=cond_stats,
        interaction_stats=interaction_stats,
        stage1_stats=stage1_stats,
    )

    # Save
    save_path = tmp_path / "extractor.json"
    extractor.save(save_path)

    # Load
    loaded_extractor = AdvancedFeatureExtractor.load(save_path)

    # Check consistency
    assert loaded_extractor.feature_dim == extractor.feature_dim
    assert loaded_extractor.feature_names == extractor.feature_names

    # Extract features with both and compare
    test_row = {"benchmark": "math", "condition": "zero-shot", "item_content": "Test"}
    features_orig = extractor.extract(test_row)
    features_loaded = loaded_extractor.extract(test_row)

    np.testing.assert_array_almost_equal(features_orig, features_loaded)

    print("✓ Feature extractor serialization test passed")


if __name__ == "__main__":
    import tempfile

    print("Running advanced features tests...\n")

    test_benchmark_statistics()
    test_condition_statistics()
    test_interaction_statistics()
    test_feature_extractor()

    with tempfile.TemporaryDirectory() as tmpdir:
        test_feature_extractor_serialization(Path(tmpdir))

    print("\n✅ All tests passed!")
