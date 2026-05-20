"""In-process smoke test for the Canonicalization V2 submission zip.

Run:
    python scripts/smoke_canon_v2.py --zip submissions/v1_kfactor_canon_v2.zip
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import sys
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType


def _load_model_from_zip(zip_path: Path, tmpdir: Path) -> ModuleType:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmpdir)
    model_path = tmpdir / "model.py"
    if not model_path.exists():
        raise FileNotFoundError(f"model.py not found in {zip_path}")

    os.environ["V1_KFACTOR_DUMMY_ENCODER"] = "1"
    sys.path.insert(0, str(tmpdir))
    try:
        spec = importlib.util.spec_from_file_location("canon_v2_smoke_model", model_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not import {model_path}")
        module = importlib.util.module_from_spec(spec)
        with contextlib.redirect_stdout(sys.stderr):
            spec.loader.exec_module(module)
        return module
    finally:
        try:
            sys.path.remove(str(tmpdir))
        except ValueError:
            pass


def _prior_rows(priors: dict) -> list[tuple[str, str, str]]:
    sep = priors["key_sep"]
    rows: list[tuple[str, str, str]] = []
    for key in priors["subject_category"]:
        parts = str(key).split(sep)
        if len(parts) == 3:
            rows.append((parts[0], parts[1], parts[2]))
    if not rows:
        raise ValueError("runtime_priors.json has no subject_category rows")
    return rows


def _choose(rows: list[tuple[str, str, str]], predicate) -> tuple[str, str, str]:
    for row in rows:
        if predicate(row):
            return row
    raise ValueError("no smoke row matched predicate")


def _input(subject_content: str, benchmark: str, condition: str) -> dict:
    return {
        "subject_content": subject_content,
        "item_content": "Canonicalization V2 smoke item",
        "benchmark": benchmark,
        "condition": condition,
    }


def _predict(model: ModuleType, subject_content: str, benchmark: str, condition: str) -> float:
    return float(model.predict(_input(subject_content, benchmark, condition)))


def _assert_same(label: str, baseline: float, value: float, tol: float = 1e-12) -> None:
    if abs(baseline - value) > tol:
        raise AssertionError(f"{label}: {value:.15f} != baseline {baseline:.15f}")


def run_smoke(zip_path: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="canon-v2-smoke-") as tmp:
        tmpdir = Path(tmp)
        model = _load_model_from_zip(zip_path, tmpdir)
        priors = json.loads((tmpdir / "runtime_priors.json").read_text())
        rows = _prior_rows(priors)

        subject_row = rows[0]
        bench_row = _choose(
            rows,
            lambda row: row[1] in {"ai2d_test", "mathvista_mini", "mmlupro", "swebench"},
        )
        condition_case_row = _choose(rows, lambda row: row[2].lower() != row[2])
        none_row = _choose(rows, lambda row: row[2] == "none")

        subject, benchmark, condition = subject_row
        control_subject = f"Name: {subject}"
        baseline = _predict(model, control_subject, benchmark, condition)
        subject_variants = {
            "control": control_subject,
            "space_before_colon": f"Name : {subject}",
            "fullwidth_colon": f"Name\uff1a{subject}",
            "subject_prefix": f"Subject: {subject}",
            "model_prefix": f"Model: {subject}",
            "display_name_prefix": f"display_name: {subject}",
            "markdown_dash": f"- Name: {subject}",
            "quoted_line": f'"Name: {subject}"',
            "trailing_period": f"Name: {subject}.",
        }
        results: dict[str, float] = {}
        for label, raw_subject in subject_variants.items():
            value = _predict(model, raw_subject, benchmark, condition)
            _assert_same(f"subject_{label}", baseline, value)
            results[f"subject_{label}"] = value

        bench_subject, bench, bench_condition = bench_row
        bench_baseline = _predict(model, f"Name: {bench_subject}", bench, bench_condition)
        bench_variants = {
            "upper": bench.upper(),
            "separator_swap": bench.replace("_", "-"),
            "spaced": bench.replace("_", " "),
        }
        if bench == "mmlupro":
            bench_variants["mmlu_pro"] = "mmlu pro"
        for label, variant in bench_variants.items():
            value = _predict(model, f"Name: {bench_subject}", variant, bench_condition)
            _assert_same(f"benchmark_{label}", bench_baseline, value)
            results[f"benchmark_{label}"] = value

        cond_subject, cond_benchmark, cond = condition_case_row
        cond_baseline = _predict(model, f"Name: {cond_subject}", cond_benchmark, cond)
        for label, variant in {"lower": cond.lower(), "upper": cond.upper()}.items():
            value = _predict(model, f"Name: {cond_subject}", cond_benchmark, variant)
            _assert_same(f"condition_{label}", cond_baseline, value)
            results[f"condition_{label}"] = value

        none_subject, none_benchmark, none_condition = none_row
        none_baseline = _predict(model, f"Name: {none_subject}", none_benchmark, none_condition)
        for label, variant in {"empty": "", "null": "null", "dash": "-"}.items():
            value = _predict(model, f"Name: {none_subject}", none_benchmark, variant)
            _assert_same(f"condition_none_{label}", none_baseline, value)
            results[f"condition_none_{label}"] = value

        cold = _predict(model, "Name: canon-v2-cold-subject", benchmark, condition)
        global_p = float(model._clip_prob(priors["global"]))
        _assert_same("cold_subject_global", global_p, cold)
        results["cold_subject"] = cold

        counters = dict(getattr(model, "_LOOKUP_AUDIT_COUNTERS"))
        return {"zip": str(zip_path), "results": results, "audit_counters": counters}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default="submissions/v1_kfactor_canon_v2.zip", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload = run_smoke(args.zip)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
