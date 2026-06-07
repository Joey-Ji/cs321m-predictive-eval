"""Unit tests for the v1_subject_ability_v5 submission.

Enforces the Codabench predict()/acquisition_function() contract for
``submissions/v1_subject_ability_v5``:

  * predict() returns a NATIVE python float, finite, in [0, 1] (in
    [CLIP_LO, CLIP_HI] for non-empty labeled), deterministic, and handles
    labeled=None / [] / full dicts.
  * acquisition_function() returns a native finite float and never raises.
  * model.py imports with no network access and only stdlib + numpy + scipy.

Runnable two ways:
    uv run python tests/test_v1_subject_ability_v5.py          # script mode
    uv run python -m pytest tests/test_v1_subject_ability_v5.py -q   # pytest
"""

from __future__ import annotations

import importlib.util
import math
import socket
from pathlib import Path
from types import ModuleType
from unittest import mock


ROOT = Path(__file__).resolve().parent.parent
SUBMISSION = ROOT / "submissions" / "v1_subject_ability_v5"
MODEL_PATH = SUBMISSION / "model.py"
LABELING_PATH = SUBMISSION / "labeling.py"


def _load_module(name: str, path: Path) -> ModuleType:
    """Load a submission file in isolation via importlib (no sys.path mutation)."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Loaded once at module import; cheap pure-python modules with no disk/network IO.
MODEL = _load_module("v1_subject_ability_v5_model", MODEL_PATH)
LABELING = _load_module("v1_subject_ability_v5_labeling", LABELING_PATH)

CLIP_LO = MODEL.CLIP_LO
CLIP_HI = MODEL.CLIP_HI
MIN_OBSERVATIONS = MODEL.MIN_OBSERVATIONS


# --------------------------------------------------------------------------- #
# Sample-input fixtures (pure python; no external data files).
# --------------------------------------------------------------------------- #
def _sample_input() -> dict:
    return {
        "benchmark": "bench_0",
        "condition": "zero-shot",
        "subject_content": "Name: Subject-1\nOrganization: Test",
        "item_content": "What is the capital of France? body text.",
    }


def _balanced_labeled() -> list[dict]:
    """Multiple subjects across several benchmarks; full 5-key dicts."""
    rows: list[dict] = []
    for s in range(6):
        for i in range(6):
            rows.append(
                {
                    "benchmark": f"bench_{i % 3}",
                    "condition": "zero-shot",
                    "subject_content": f"Name: Subject-{s}\nOrganization: Test",
                    "item_content": f"Synthetic item {i} body text long enough.",
                    "label": int((s + i) % 2),
                }
            )
    return rows


def _skewed_labeled() -> list[dict]:
    """One benchmark; Subject-0 passes nearly everything, others vary.

    Drives predict() off the 0.5 prior into the calibrated regime.
    """
    rows: list[dict] = []
    for s in range(8):
        pass_rate = 0.95 if s == 0 else (0.1 if s % 2 else 0.5)
        for i in range(8):
            rows.append(
                {
                    "benchmark": "bench_X",
                    "condition": "zero-shot",
                    "subject_content": f"Name: Subject-{s}",
                    "item_content": f"Item {i}",
                    "label": int((i / 8.0) < pass_rate),
                }
            )
    return rows


def _all_same_labeled(label: int) -> list[dict]:
    """30 anchors for one subject, all with the same label (saturates the clip)."""
    return [
        {
            "benchmark": "bench_X",
            "condition": "zero-shot",
            "subject_content": "Name: Subject-0",
            "item_content": f"Item {i}",
            "label": label,
        }
        for i in range(30)
    ]


def _is_finite(x) -> bool:
    return isinstance(x, float) and math.isfinite(x)


# --------------------------------------------------------------------------- #
# P0 tests
# --------------------------------------------------------------------------- #
def test_predict_returns_python_float_not_numpy() -> None:
    # np.float64 passes isinstance(x, float); enforce the exact native type.
    for labeled in (None, [], _balanced_labeled(), _skewed_labeled()):
        result = MODEL.predict(_sample_input(), labeled)
        assert type(result) is float, f"type was {type(result)!r} for labeled len {labeled if labeled is None else len(labeled)}"


def test_predict_in_unit_interval() -> None:
    for labeled in (None, [], _balanced_labeled(), _skewed_labeled()):
        result = MODEL.predict(_sample_input(), labeled)
        assert 0.0 <= result <= 1.0, f"{result} outside [0, 1]"


def test_predict_finite_no_nan_inf() -> None:
    for labeled in (None, [], _balanced_labeled(), _skewed_labeled()):
        result = MODEL.predict(_sample_input(), labeled)
        assert _is_finite(result), f"non-finite result {result!r}"


def test_predict_labeled_none() -> None:
    result = MODEL.predict(_sample_input(), None)
    assert type(result) is float and _is_finite(result)
    assert 0.0 <= result <= 1.0


def test_predict_labeled_empty_list() -> None:
    result = MODEL.predict(_sample_input(), [])
    assert type(result) is float and _is_finite(result)
    assert 0.0 <= result <= 1.0


def test_predict_labeled_with_entries() -> None:
    # Full 5-key dicts (4 content keys + int "label"); skewed so we exercise
    # the real calibration path rather than the empty-labeled early return.
    labeled = _skewed_labeled()
    high = MODEL.predict(
        {**_sample_input(), "benchmark": "bench_X", "subject_content": "Name: Subject-0"},
        labeled,
    )
    assert type(high) is float and _is_finite(high)
    assert CLIP_LO <= high <= CLIP_HI
    # A strong-passing subject should land above the neutral prior.
    assert high > 0.5, f"high-ability subject predicted {high}, expected > 0.5"


# --------------------------------------------------------------------------- #
# P1 tests
# --------------------------------------------------------------------------- #
def test_predict_clamped_to_clip_range() -> None:
    # Non-empty labeled must stay within [CLIP_LO, CLIP_HI]. The all-same
    # fixtures push predict() to the clamp boundaries.
    for label in (0, 1):
        result = MODEL.predict(
            {**_sample_input(), "benchmark": "bench_X", "subject_content": "Name: Subject-0"},
            _all_same_labeled(label),
        )
        assert CLIP_LO <= result <= CLIP_HI, f"label={label} -> {result} outside clip range"
    # Confirm the clamp is actually reachable (otherwise the test is vacuous).
    all_pass = MODEL.predict(
        {**_sample_input(), "benchmark": "bench_X", "subject_content": "Name: Subject-0"},
        _all_same_labeled(1),
    )
    all_fail = MODEL.predict(
        {**_sample_input(), "benchmark": "bench_X", "subject_content": "Name: Subject-0"},
        _all_same_labeled(0),
    )
    assert all_pass == CLIP_HI, f"all-pass did not saturate CLIP_HI: {all_pass}"
    assert all_fail == CLIP_LO, f"all-fail did not saturate CLIP_LO: {all_fail}"


def test_predict_deterministic() -> None:
    inp = _sample_input()
    labeled = _skewed_labeled()
    first = MODEL.predict(inp, labeled)
    for _ in range(5):
        again = MODEL.predict(inp, labeled)
        assert again == first, f"non-deterministic: {again} != {first}"


def test_acquisition_returns_python_float() -> None:
    result = LABELING.acquisition_function(_sample_input())
    assert type(result) is float, f"type was {type(result)!r}"


def test_acquisition_finite_no_nan_inf() -> None:
    result = LABELING.acquisition_function(_sample_input())
    assert _is_finite(result), f"non-finite acquisition score {result!r}"


def test_acquisition_never_raises() -> None:
    degenerate_inputs = [
        {},
        {"benchmark": None, "condition": None, "subject_content": None, "item_content": None},
        {"benchmark": "", "condition": "", "subject_content": "", "item_content": ""},
        {"benchmark": "x"},  # missing keys
        None,  # platform should never pass this, but must not raise
    ]
    for bad in degenerate_inputs:
        result = LABELING.acquisition_function(bad)
        assert type(result) is float and _is_finite(result), f"bad input {bad!r} -> {result!r}"


def test_predict_multi_benchmark_labeled() -> None:
    # Anchors span several benchmarks; predict must use the in-category subset
    # and still return a valid clamped float for the queried benchmark.
    labeled = _balanced_labeled()
    benchmarks = {row["benchmark"] for row in labeled}
    assert len(benchmarks) > 1, "fixture should span multiple benchmarks"
    for benchmark in sorted(benchmarks):
        result = MODEL.predict({**_sample_input(), "benchmark": benchmark}, labeled)
        assert type(result) is float and _is_finite(result)
        assert CLIP_LO <= result <= CLIP_HI, f"benchmark={benchmark} -> {result}"
    # An unseen benchmark falls back to global stats but stays valid.
    unseen = MODEL.predict({**_sample_input(), "benchmark": "bench_unseen"}, labeled)
    assert type(unseen) is float and _is_finite(unseen)
    assert CLIP_LO <= unseen <= CLIP_HI


# --------------------------------------------------------------------------- #
# P2 tests
# --------------------------------------------------------------------------- #
def test_import_no_network() -> None:
    # Re-import model.py AND exercise predict()/acquisition_function() with every
    # network entry point poisoned. Poisoning socket at the lowest level also
    # blocks urllib/http indirectly; urllib.urlopen + http.client are patched too
    # so a future regression adding an outbound call at import OR in a call is caught.
    import http.client
    import urllib.request

    def _boom(*args, **kwargs):
        raise AssertionError("network access in the submission")

    with mock.patch.object(socket, "socket", _boom), mock.patch.object(
        socket, "getaddrinfo", _boom
    ), mock.patch.object(socket, "create_connection", _boom), mock.patch.object(
        urllib.request, "urlopen", _boom
    ), mock.patch.object(http.client, "HTTPConnection", _boom):
        module = _load_module("v1_subject_ability_v5_model_nonet", MODEL_PATH)
        labeling = _load_module("v1_subject_ability_v5_labeling_nonet", LABELING_PATH)
        # Exercise the runtime paths, not just import.
        pred = module.predict(_sample_input(), _skewed_labeled())
        acq = labeling.acquisition_function(_sample_input())
    assert callable(module.predict)
    assert module.CLIP_LO == CLIP_LO and module.CLIP_HI == CLIP_HI
    assert type(pred) is float and _is_finite(pred)
    assert type(acq) is float and _is_finite(acq)


def test_model_self_contained() -> None:
    # model.py must depend only on stdlib + numpy + scipy. Inspect the module
    # source for third-party imports beyond that allowlist.
    import ast
    import sys

    allowed_top_level = {"numpy", "scipy"}
    source = MODEL_PATH.read_text()
    tree = ast.parse(source)
    imported_top: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_top.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                imported_top.add(node.module.split(".")[0])

    stdlib = set(getattr(sys, "stdlib_module_names", set()))
    offenders = {
        name
        for name in imported_top
        if name not in allowed_top_level and name not in stdlib and name != "__future__"
    }
    assert not offenders, f"model.py imports non-allowlisted modules: {sorted(offenders)}"


def test_predict_degenerate_input_values() -> None:
    # Missing keys, None values, empty strings, and bad label types must not
    # raise; predict() guards everything and returns a valid float.
    degenerate_inputs = [
        {},
        {"benchmark": None, "condition": None, "subject_content": None, "item_content": None},
        {"benchmark": "", "condition": "", "subject_content": "", "item_content": ""},
    ]
    degenerate_labeled = [
        None,
        [],
        [{}],
        [
            {
                "benchmark": None,
                "condition": None,
                "subject_content": None,
                "item_content": None,
                "label": None,
            }
        ],
        [{"benchmark": "", "subject_content": "", "item_content": "", "label": "not_a_number"}],
    ]
    for inp in degenerate_inputs:
        for labeled in degenerate_labeled:
            result = MODEL.predict(inp, labeled)
            assert type(result) is float and _is_finite(result), f"inp={inp!r} labeled={labeled!r} -> {result!r}"
            assert 0.0 <= result <= 1.0


def test_predict_below_min_observations() -> None:
    # A subject with fewer than MIN_OBSERVATIONS anchors gets the group mean
    # rather than its own (untrusted) accuracy; result must stay valid/clamped.
    assert MIN_OBSERVATIONS >= 1
    n = MIN_OBSERVATIONS - 1
    labeled = [
        {
            "benchmark": "bench_X",
            "condition": "zero-shot",
            "subject_content": "Name: Subject-0",
            "item_content": f"Item {i}",
            "label": 1,
        }
        for i in range(n)
    ]
    # Pad with other subjects so there is a defined group, still keeping the
    # queried subject below MIN_OBSERVATIONS.
    for s in range(1, 5):
        for i in range(3):
            labeled.append(
                {
                    "benchmark": "bench_X",
                    "condition": "zero-shot",
                    "subject_content": f"Name: Subject-{s}",
                    "item_content": f"Item {i}",
                    "label": int((s + i) % 2),
                }
            )
    result = MODEL.predict(
        {**_sample_input(), "benchmark": "bench_X", "subject_content": "Name: Subject-0"},
        labeled,
    )
    assert type(result) is float and _is_finite(result)
    assert CLIP_LO <= result <= CLIP_HI


# --------------------------------------------------------------------------- #
# Script runner (CI runs tests as scripts, pytest not guaranteed installed).
# --------------------------------------------------------------------------- #
def main() -> int:
    tests = [
        # P0
        test_predict_returns_python_float_not_numpy,
        test_predict_in_unit_interval,
        test_predict_finite_no_nan_inf,
        test_predict_labeled_none,
        test_predict_labeled_empty_list,
        test_predict_labeled_with_entries,
        # P1
        test_predict_clamped_to_clip_range,
        test_predict_deterministic,
        test_acquisition_returns_python_float,
        test_acquisition_finite_no_nan_inf,
        test_acquisition_never_raises,
        test_predict_multi_benchmark_labeled,
        # P2
        test_import_no_network,
        test_model_self_contained,
        test_predict_degenerate_input_values,
        test_predict_below_min_observations,
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - surface failure per test
            failures += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
