"""Canonicalization V2 lookup safety tests.

Run:
    python tests/test_canonicalization_v2.py
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from types import MappingProxyType
from typing import Callable


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "submissions" / "v1_kfactor" / "model.py"


def _find_locked_runtime_priors() -> Path:
    for base in (ROOT, *ROOT.parents):
        candidate = base / "data" / "stage2" / "priors_v1_locked" / "runtime_priors.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("data/stage2/priors_v1_locked/runtime_priors.json not found")


def _load_model_normalizers(priors: dict) -> MappingProxyType:
    source = MODEL_PATH.read_text()
    prefix = source.split("\ndef _build_head", 1)[0]
    namespace: dict = {"__file__": str(MODEL_PATH)}
    previous_hf_home = os.environ.get("HF_HOME")
    with tempfile.TemporaryDirectory(prefix="canon-v2-test-hf-") as tmp:
        os.environ["HF_HOME"] = tmp
        try:
            exec(prefix, namespace)
        finally:
            if previous_hf_home is None:
                os.environ.pop("HF_HOME", None)
            else:
                os.environ["HF_HOME"] = previous_hf_home
    namespace["PRIORS"] = priors
    namespace["PRIOR_KEY_SEP"] = priors["key_sep"]
    namespace["_prior_benchmark_keys"].cache_clear()
    namespace["_prior_condition_keys"].cache_clear()
    namespace["_benchmark_alias_map"].cache_clear()
    namespace["_condition_alias_map"].cache_clear()
    return MappingProxyType(namespace)


def _all_benchmark_keys(priors: dict) -> set[str]:
    sep = priors["key_sep"]
    keys = {str(k) for k in priors["benchmark"]}
    keys.update(str(k).split(sep)[0] for k in priors["benchmark_condition"])
    keys.update(str(k).split(sep)[1] for k in priors["subject_benchmark"])
    keys.update(str(k).split(sep)[1] for k in priors["subject_category"])
    return keys


def _all_condition_keys(priors: dict) -> set[str]:
    sep = priors["key_sep"]
    keys = {str(k).split(sep)[1] for k in priors["benchmark_condition"]}
    keys.update(str(k).split(sep)[2] for k in priors["subject_category"])
    return keys


def _assert_no_collisions(keys: set[str], normalize: Callable[[str], str], label: str) -> None:
    buckets: dict[str, list[str]] = {}
    for key in keys:
        buckets.setdefault(normalize(key), []).append(key)
    collisions = {
        normalized: sorted(set(raw_keys))
        for normalized, raw_keys in buckets.items()
        if len(set(raw_keys)) > 1
    }
    assert not collisions, f"{label} normalization collisions: {collisions}"


def test_subject_prefix_variants_and_collision_safety() -> None:
    priors = json.loads(_find_locked_runtime_priors().read_text())
    ns = _load_model_normalizers(priors)
    normalize_subject = ns["_normalize_subject"]
    subject = next(k for k in sorted(priors["subject"]) if not k.endswith((".", ",", ";")))

    variants = [
        f"Name: {subject}",
        f"name: {subject}",
        f"NAME: {subject}",
        f"Name : {subject}",
        f"Name:{subject}",
        f"Name\uff1a{subject}",
        f"Subject: {subject}",
        f"Model: {subject}",
        f"display_name: {subject}",
        f"- Name: {subject}",
        f"* Name: {subject}",
        f"> Name: {subject}",
        f'"Name: {subject}"',
        f"'Name: {subject}'",
        f"Name: {subject}.",
        f"Name: {subject},",
        f"Name: {subject};",
    ]
    assert {normalize_subject(v) for v in variants} == {subject}
    _assert_no_collisions(set(priors["subject"]), normalize_subject, "subject")


def test_benchmark_alias_variants_and_collision_safety() -> None:
    priors = json.loads(_find_locked_runtime_priors().read_text())
    ns = _load_model_normalizers(priors)
    normalize_benchmark = ns["_normalize_benchmark_key"]

    assert normalize_benchmark("AI2D-TEST") == "ai2d_test"
    assert normalize_benchmark("ai2d test") == "ai2d_test"
    assert normalize_benchmark("MMLU Pro") == "mmlupro"
    assert normalize_benchmark("swe bench") == "swebench"
    _assert_no_collisions(_all_benchmark_keys(priors), normalize_benchmark, "benchmark")


def test_condition_variants_and_collision_safety() -> None:
    priors = json.loads(_find_locked_runtime_priors().read_text())
    ns = _load_model_normalizers(priors)
    normalize_condition = ns["_normalize_condition_key"]
    condition_keys = _all_condition_keys(priors)
    mixed_case = next(c for c in sorted(condition_keys) if c.lower() != c)
    pipe_condition = next(c for c in sorted(condition_keys) if "|" in c)

    assert normalize_condition(mixed_case.lower()) == mixed_case
    assert normalize_condition(mixed_case.upper()) == mixed_case
    assert normalize_condition(pipe_condition.replace("|", " | ")) == pipe_condition
    for variant in ("", "None", "null", "n/a", "na", "-"):
        assert normalize_condition(variant) == "none"
    _assert_no_collisions(condition_keys, normalize_condition, "condition")


def main() -> int:
    tests = [
        test_subject_prefix_variants_and_collision_safety,
        test_benchmark_alias_variants_and_collision_safety,
        test_condition_variants_and_collision_safety,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
