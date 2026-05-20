"""Unit tests for the prior-only online intercept update."""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from types import MappingProxyType


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "submissions" / "v1_kfactor" / "model.py"


def _sigmoid(logit: float) -> float:
    if logit >= 0:
        z = math.exp(-logit)
        return 1.0 / (1.0 + z)
    z = math.exp(logit)
    return z / (1.0 + z)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _load_online_namespace(prior_p: float = 0.5) -> MappingProxyType:
    source = MODEL_PATH.read_text()
    prefix = source.split("\nHEAD_META = ", 1)[0]
    namespace: dict = {"__file__": str(MODEL_PATH)}
    previous_hf_home = os.environ.get("HF_HOME")
    with tempfile.TemporaryDirectory(prefix="online-intercept-test-hf-") as tmp:
        os.environ["HF_HOME"] = tmp
        try:
            exec(prefix, namespace)
        finally:
            if previous_hf_home is None:
                os.environ.pop("HF_HOME", None)
            else:
                os.environ["HF_HOME"] = previous_hf_home

    prior_logit = _logit(prior_p)
    namespace["_raw_logit"] = lambda _row: prior_logit
    namespace["_sigmoid"] = _sigmoid
    namespace["ONLINE_INTERCEPT_LAM"] = 50.0
    namespace["ONLINE_INTERCEPT_CLIP"] = 0.15
    namespace["_ONLINE_INTERCEPT_CACHE_KEY"] = None
    namespace["_ONLINE_INTERCEPT_CACHE"] = 0.0
    return MappingProxyType(namespace)


def _row(label: object, idx: int = 0) -> dict:
    return {
        "subject_content": "Name: subject",
        "item_content": f"item {idx}",
        "benchmark": "bench",
        "condition": "cond",
        "label": label,
    }


def test_empty_labeled_delta_is_zero() -> None:
    ns = _load_online_namespace(prior_p=0.37)

    assert ns["_online_intercept_delta"]([]) == 0.0
    assert ns["_online_intercept_delta"](None) == 0.0


def test_one_positive_at_half_prior_uses_newton_formula() -> None:
    ns = _load_online_namespace(prior_p=0.5)

    delta = ns["_online_intercept_delta"]([_row(1)])

    assert math.isclose(delta, 0.5 / 50.25, rel_tol=0.0, abs_tol=1e-12)


def test_clip_threshold_for_low_prior_positive_labels() -> None:
    ns = _load_online_namespace(prior_p=1e-9)

    seven_delta = ns["_online_intercept_delta"]([_row(1, i) for i in range(7)])
    eight_delta = ns["_online_intercept_delta"]([_row(1, i) for i in range(8)])

    assert 0.13 < seven_delta < 0.15
    assert math.isclose(eight_delta, 0.15, rel_tol=0.0, abs_tol=1e-12)


def test_non_binary_labels_are_skipped() -> None:
    ns = _load_online_namespace(prior_p=0.5)

    delta = ns["_online_intercept_delta"]([_row(0.5), _row("1"), _row(None)])

    assert delta == 0.0


def main() -> int:
    tests = [
        test_empty_labeled_delta_is_zero,
        test_one_positive_at_half_prior_uses_newton_formula,
        test_clip_threshold_for_low_prior_positive_labels,
        test_non_binary_labels_are_skipped,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
