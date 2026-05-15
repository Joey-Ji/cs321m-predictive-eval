"""Unit tests for src.features side-feature helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import (
    REPRESENTATION_VERSION,
    build_side_feature_vocab,
    encode_side_features,
)


def _make_vocab() -> dict:
    rows = [
        {"benchmark": "alpha", "condition": "zero-shot"},
        {"benchmark": "beta", "condition": "cot"},
        {"benchmark": "alpha", "condition": "zero-shot"},
        {"benchmark": None, "condition": None},
    ]
    return build_side_feature_vocab(rows)


def test_vocab_shape_and_determinism() -> list[str]:
    errs: list[str] = []
    vocab = _make_vocab()
    if vocab["benchmark_dim"] != 3:
        errs.append(f"benchmark_dim={vocab['benchmark_dim']} expected 3 (alpha, beta, '')")
    if vocab["condition_dim"] != 3:
        errs.append(f"condition_dim={vocab['condition_dim']} expected 3 (cot, zero-shot, '')")
    if vocab["side_feature_dim"] != vocab["benchmark_dim"] + vocab["condition_dim"]:
        errs.append("side_feature_dim must equal benchmark_dim + condition_dim")
    if vocab["representation_version"] != REPRESENTATION_VERSION:
        errs.append(
            f"representation_version={vocab['representation_version']!r} != {REPRESENTATION_VERSION!r}"
        )
    expected_benchmarks = ["", "alpha", "beta"]
    if list(vocab["benchmark"].keys()) != expected_benchmarks:
        errs.append(f"benchmark keys not in sorted order: {list(vocab['benchmark'].keys())}")
    expected_conditions = ["", "cot", "zero-shot"]
    if list(vocab["condition"].keys()) != expected_conditions:
        errs.append(f"condition keys not in sorted order: {list(vocab['condition'].keys())}")
    return errs


def test_seen_keys_one_hot() -> list[str]:
    errs: list[str] = []
    vocab = _make_vocab()
    out = encode_side_features({"benchmark": "alpha", "condition": "cot"}, vocab)
    if out.shape != (vocab["side_feature_dim"],):
        errs.append(f"shape {out.shape} != expected ({vocab['side_feature_dim']},)")
    if float(out.sum()) != 2.0:
        errs.append(f"expected exactly two 1.0 entries, got sum {float(out.sum())}")
    if out.dtype.name != "float32":
        errs.append(f"dtype {out.dtype} != float32")
    b_idx = vocab["benchmark"]["alpha"]
    c_idx = vocab["condition"]["cot"]
    if out[b_idx] != 1.0:
        errs.append("benchmark 'alpha' not set in benchmark block")
    if out[vocab["benchmark_dim"] + c_idx] != 1.0:
        errs.append("condition 'cot' not set in condition block")
    return errs


def test_unseen_keys_zero_block() -> list[str]:
    errs: list[str] = []
    vocab = _make_vocab()
    out = encode_side_features(
        {"benchmark": "unseen-benchmark", "condition": "unseen-condition"}, vocab
    )
    if out.shape != (vocab["side_feature_dim"],):
        errs.append(f"shape {out.shape} != expected ({vocab['side_feature_dim']},)")
    if float(out.sum()) != 0.0:
        errs.append(f"unseen keys should yield all-zero block, got sum {float(out.sum())}")
    return errs


def test_mixed_seen_and_unseen() -> list[str]:
    errs: list[str] = []
    vocab = _make_vocab()
    out = encode_side_features({"benchmark": "alpha", "condition": "unseen"}, vocab)
    if float(out.sum()) != 1.0:
        errs.append(
            f"seen benchmark + unseen condition should yield exactly one 1.0, got sum {float(out.sum())}"
        )
    b_idx = vocab["benchmark"]["alpha"]
    if out[b_idx] != 1.0:
        errs.append("seen benchmark not set")
    return errs


def test_missing_keys_treated_as_empty() -> list[str]:
    errs: list[str] = []
    vocab = _make_vocab()
    out = encode_side_features({}, vocab)
    if float(out.sum()) != 2.0:
        errs.append(
            f"empty row should map to '' in both blocks (sum=2), got {float(out.sum())}"
        )
    empty_b = vocab["benchmark"][""]
    empty_c = vocab["condition"][""]
    if out[empty_b] != 1.0 or out[vocab["benchmark_dim"] + empty_c] != 1.0:
        errs.append("empty fields did not land on '' index in both blocks")
    return errs


def main() -> int:
    all_errs: list[str] = []
    for name, test in [
        ("vocab_shape_and_determinism", test_vocab_shape_and_determinism),
        ("seen_keys_one_hot", test_seen_keys_one_hot),
        ("unseen_keys_zero_block", test_unseen_keys_zero_block),
        ("mixed_seen_and_unseen", test_mixed_seen_and_unseen),
        ("missing_keys_treated_as_empty", test_missing_keys_treated_as_empty),
    ]:
        errs = test()
        if errs:
            for err in errs:
                all_errs.append(f"[{name}] {err}")
        else:
            print(f"PASS {name}")

    if all_errs:
        print(f"\nFAIL - {len(all_errs)} assertion(s):")
        for err in all_errs:
            print(f"  - {err}")
        return 1
    print("\nPASS - side feature unit tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
