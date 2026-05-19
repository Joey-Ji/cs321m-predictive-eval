"""Synthetic lookup audit for v1_kfactor priors-only submissions.

Loads a debug-instrumented submission ZIP, samples dense subject-category prior
cells, and checks whether small Codabench-format input transformations still
resolve to the deepest subject-category prior.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import sys
import tempfile
import unicodedata
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd


Transformation = Callable[[str, str, str, str], tuple[str, str, str] | None]


def _load_model_from_zip(zip_path: Path, tmpdir: Path) -> ModuleType:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmpdir)
    model_path = tmpdir / "model.py"
    if not model_path.exists():
        raise FileNotFoundError(f"model.py not found in {zip_path}")

    os.environ["V1_KFACTOR_DUMMY_ENCODER"] = "1"
    os.environ.pop("V1_KFACTOR_LOOKUP_AUDIT", None)
    sys.path.insert(0, str(tmpdir))
    try:
        spec = importlib.util.spec_from_file_location("lookup_audit_model", model_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not import {model_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        try:
            sys.path.remove(str(tmpdir))
        except ValueError:
            pass


def _display_name_map(subjects_path: Path) -> dict[str, str]:
    subjects = pd.read_parquet(subjects_path, columns=["display_name"])
    out: dict[str, str] = {}
    for display in subjects["display_name"].dropna().astype(str).tolist():
        key = display.strip().lower()
        out.setdefault(key, display)
    return out


def _sample_cells(subject_category_path: Path, n: int, min_count: int, seed: int) -> list[dict[str, Any]]:
    df = pd.read_parquet(
        subject_category_path,
        columns=["subject_key", "benchmark", "condition", "n"],
    )
    dense = df[df["n"] >= min_count].copy()
    if dense.empty:
        raise ValueError(f"no subject_category cells with n >= {min_count}")
    records = dense.to_dict(orient="records")
    rng = random.Random(seed)
    if len(records) > n:
        records = rng.sample(records, n)
    return records


def _pipe_spaced(condition: str) -> str | None:
    if "|" not in condition:
        return None
    return re.sub(r"\s*\|\s*", " | ", condition)


def _transformations() -> dict[str, Transformation]:
    return {
        "control_original": lambda display, benchmark, condition, raw: (raw, benchmark, condition),
        "subject_lowercase_prefix": lambda display, benchmark, condition, raw: (
            f"name: {display}",
            benchmark,
            condition,
        ),
        "subject_uppercase_prefix": lambda display, benchmark, condition, raw: (
            f"NAME: {display}",
            benchmark,
            condition,
        ),
        "subject_no_prefix": lambda display, benchmark, condition, raw: (display, benchmark, condition),
        "subject_leading_whitespace": lambda display, benchmark, condition, raw: (
            f"  Name: {display}",
            benchmark,
            condition,
        ),
        "subject_trailing_newline": lambda display, benchmark, condition, raw: (
            f"Name: {display}\n",
            benchmark,
            condition,
        ),
        "subject_unicode_nfd": lambda display, benchmark, condition, raw: (
            unicodedata.normalize("NFD", raw),
            benchmark,
            condition,
        ),
        "benchmark_titlecase": lambda display, benchmark, condition, raw: (
            raw,
            benchmark.title(),
            condition,
        ),
        "benchmark_trailing_space": lambda display, benchmark, condition, raw: (
            raw,
            f"{benchmark} ",
            condition,
        ),
        "condition_trailing_space": lambda display, benchmark, condition, raw: (
            raw,
            benchmark,
            f"{condition} ",
        ),
        "condition_pipe_spaces": lambda display, benchmark, condition, raw: (
            (raw, benchmark, spaced) if (spaced := _pipe_spaced(condition)) is not None else None
        ),
    }


def _reset_audit(model: ModuleType) -> None:
    counters = getattr(model, "_LOOKUP_AUDIT_COUNTERS")
    for key in counters:
        counters[key] = 0
    getattr(model, "_LOOKUP_AUDIT_SAMPLES").clear()


def _lookup_outcome(model: ModuleType, raw_subject: str, benchmark: str, condition: str) -> tuple[str, str]:
    counters = getattr(model, "_LOOKUP_AUDIT_COUNTERS")
    before = dict(counters)
    subject_key = model._normalize_subject(raw_subject)
    model._prior_values(subject_key, benchmark, condition, raw_subject_content=raw_subject)
    deltas = {
        key: counters[key] - before.get(key, 0)
        for key in counters
        if counters[key] != before.get(key, 0)
    }
    if len(deltas) != 1:
        raise RuntimeError(f"expected one lookup counter increment, got {deltas}")
    return next(iter(deltas)), subject_key


def _examples_append(examples: dict[str, list[dict[str, Any]]], name: str, row: dict[str, Any]) -> None:
    bucket = examples.setdefault(name, [])
    if len(bucket) < 5:
        bucket.append(row)


def run_audit(
    zip_path: Path,
    subject_category_path: Path,
    subjects_path: Path,
    out_path: Path,
    n: int,
    min_count: int,
    seed: int,
) -> dict[str, Any]:
    cells = _sample_cells(subject_category_path, n=n, min_count=min_count, seed=seed)
    display_by_key = _display_name_map(subjects_path)
    transformations = _transformations()
    table: dict[str, Counter[str]] = {name: Counter() for name in transformations}
    examples: dict[str, list[dict[str, Any]]] = {}
    missing_display_keys: list[str] = []

    with tempfile.TemporaryDirectory(prefix="lookup-audit-") as tmp:
        model = _load_model_from_zip(zip_path, Path(tmp))
        _reset_audit(model)

        for cell in cells:
            subject_key = str(cell["subject_key"])
            benchmark = str(cell["benchmark"])
            condition = str(cell["condition"])
            display = display_by_key.get(subject_key)
            if display is None:
                display = subject_key
                if len(missing_display_keys) < 10:
                    missing_display_keys.append(subject_key)
            raw_subject = f"Name: {display}"

            for name, transform in transformations.items():
                variant = transform(display, benchmark, condition, raw_subject)
                if variant is None:
                    table[name]["not_applicable"] += 1
                    continue
                raw_variant, benchmark_variant, condition_variant = variant
                outcome, normalized_subject_key = _lookup_outcome(
                    model,
                    raw_variant,
                    benchmark_variant,
                    condition_variant,
                )
                table[name][outcome] += 1
                if outcome != "hit_subject_category":
                    _examples_append(
                        examples,
                        name,
                        {
                            "expected_subject_key": subject_key,
                            "normalized_subject_key": normalized_subject_key,
                            "benchmark": benchmark,
                            "condition": condition,
                            "raw_subject_content": raw_variant,
                            "variant_benchmark": benchmark_variant,
                            "variant_condition": condition_variant,
                            "outcome": outcome,
                        },
                    )

    payload = {
        "zip": str(zip_path),
        "subject_category_path": str(subject_category_path),
        "subjects_path": str(subjects_path),
        "n_requested": int(n),
        "n_cells": int(len(cells)),
        "min_count": int(min_count),
        "seed": int(seed),
        "missing_display_keys": missing_display_keys,
        "confusion_table": {
            name: dict(sorted(counter.items()))
            for name, counter in table.items()
        },
        "miss_examples": examples,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _format_table(confusion_table: dict[str, dict[str, int]]) -> str:
    outcomes = sorted({outcome for row in confusion_table.values() for outcome in row})
    lines = ["transformation\t" + "\t".join(outcomes)]
    for name, counts in confusion_table.items():
        lines.append(name + "\t" + "\t".join(str(counts.get(outcome, 0)) for outcome in outcomes))
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default="submissions/v1_kfactor_lookup_audit.zip", type=Path)
    parser.add_argument(
        "--subject-category",
        default="data/stage2/priors_v1/subject_category.parquet",
        type=Path,
    )
    parser.add_argument("--subjects", default="data/subjects.parquet", type=Path)
    parser.add_argument("--out", default="reports/lookup_audit_synthetic.json", type=Path)
    parser.add_argument("--n", default=200, type=int)
    parser.add_argument("--min-count", default=20, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload = run_audit(
        zip_path=args.zip,
        subject_category_path=args.subject_category,
        subjects_path=args.subjects,
        out_path=args.out,
        n=args.n,
        min_count=args.min_count,
        seed=args.seed,
    )
    print(_format_table(payload["confusion_table"]))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
