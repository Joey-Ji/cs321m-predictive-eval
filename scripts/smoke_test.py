"""Local CPU smoke test for a submission directory.

Imports model.py and (optionally) labeling.py from a target submission folder,
calls the entry points with synthetic four-field inputs, and verifies that
returns are native Python floats in the expected ranges.

Mirrors what the Codabench platform does at runtime, minus the sandboxing.
Use this before uploading a submission to catch obvious bugs.

Usage:
    python scripts/smoke_test.py submissions/smoke_test
    python scripts/smoke_test.py sample_code_submission
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path


SAMPLE_INPUTS = [
    {
        "benchmark": "mmlu",
        "condition": "zero-shot",
        "subject_content": "Name: GPT-4\nOrganization: OpenAI\nParameters: 1.7T\nReleased: 2023-03-14",
        "item_content": "What is the derivative of sin(x)?",
    },
    {
        "benchmark": "frontiermath2026",
        "condition": "chain-of-thought",
        "subject_content": "Name: Claude-3-Opus\nOrganization: Anthropic",
        "item_content": "Prove that every finite group of order p^2 (p prime) is abelian.",
    },
    {
        "benchmark": "humaneval",
        "condition": "none",
        "subject_content": "Name: Llama-3-70B-Instruct\nOrganization: Meta\nParameters: 70B",
        "item_content": "Write a Python function that returns the n-th Fibonacci number.",
    },
]

SAMPLE_LABELED = [
    dict(SAMPLE_INPUTS[0], label=1),
    dict(SAMPLE_INPUTS[1], label=0),
]


def load_module(submission_dir: Path, name: str):
    target = submission_dir / f"{name}.py"
    if not target.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"smoke_{name}", target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot build spec for {target}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check_predict(module) -> int:
    fails = 0
    print(f"\n[predict] testing {module.__file__}")
    for i, inp in enumerate(SAMPLE_INPUTS):
        t0 = time.perf_counter()
        out = module.predict(inp)
        dt_ms = (time.perf_counter() - t0) * 1000
        print(f"  call {i}: {dt_ms:7.2f} ms  -> {out!r}  (type={type(out).__name__})")

        if not isinstance(out, float):
            print(f"    FAIL: predict() must return native float, got {type(out).__name__}")
            fails += 1
        elif not (0.0 <= out <= 1.0):
            print(f"    FAIL: predict() must return value in [0, 1], got {out}")
            fails += 1
        elif math.isnan(out) or math.isinf(out):
            print(f"    FAIL: predict() returned NaN/inf")
            fails += 1

    print(f"\n[predict + labeled] passing labeled list of {len(SAMPLE_LABELED)} examples")
    out = module.predict(SAMPLE_INPUTS[0], labeled=SAMPLE_LABELED)
    print(f"  -> {out!r}")
    if not isinstance(out, float):
        print(f"  FAIL: predict() with labeled must return native float, got {type(out).__name__}")
        fails += 1

    print(f"\n[predict + labeled=None] explicit None case")
    out = module.predict(SAMPLE_INPUTS[0], labeled=None)
    print(f"  -> {out!r}")
    if not isinstance(out, float):
        print(f"  FAIL: predict() with labeled=None must return native float, got {type(out).__name__}")
        fails += 1

    return fails


def check_acquisition(module) -> int:
    fails = 0
    print(f"\n[acquisition_function] testing {module.__file__}")
    for i, inp in enumerate(SAMPLE_INPUTS):
        out = module.acquisition_function(inp)
        print(f"  call {i}: -> {out!r}  (type={type(out).__name__})")

        if not isinstance(out, float):
            print(f"    FAIL: acquisition_function() must return native float, got {type(out).__name__}")
            fails += 1
        elif math.isnan(out) or math.isinf(out):
            print(f"    FAIL: NaN/inf -> would trigger random fallback for whole round")
            fails += 1
    return fails


def main(submission_dir: Path) -> int:
    if not submission_dir.is_dir():
        print(f"ERROR: {submission_dir} is not a directory")
        return 2

    print(f"Smoke-testing submission: {submission_dir}")
    print(f"Python: {sys.version.split()[0]}")

    sys.path.insert(0, str(submission_dir.resolve()))

    fails = 0
    model = load_module(submission_dir, "model")
    if model is None:
        print(f"ERROR: {submission_dir}/model.py not found (required)")
        return 2
    fails += check_predict(model)

    labeling = load_module(submission_dir, "labeling")
    if labeling is None:
        print("\n[acquisition_function] labeling.py not present (optional) -> platform random fallback")
    else:
        fails += check_acquisition(labeling)

    print("\n" + ("-" * 60))
    if fails == 0:
        print("PASS: submission is shape-correct on CPU.")
        return 0
    print(f"FAIL: {fails} check(s) failed")
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path,
                        help="Directory containing model.py (and optionally labeling.py).")
    args = parser.parse_args()
    sys.exit(main(args.submission_dir))
