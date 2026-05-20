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
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd


SubjectMutator = Callable[[str], str]
FieldMutator = Callable[[str], str]


@dataclass(frozen=True)
class Transformation:
    field: str
    mutator: str
    fn: Callable[[str, str, str, str], tuple[str, str, str] | None]


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


def _subject_mutators() -> dict[str, SubjectMutator]:
    return {
        "subject_name_prefix": lambda name: f"Name: {name}",
        "subject_lowercase_prefix": lambda name: f"name: {name}",
        "subject_uppercase_prefix": lambda name: f"NAME: {name}",
        "subject_space_before_colon": lambda name: f"Name : {name}",
        "subject_no_space_after_colon": lambda name: f"Name:{name}",
        "subject_trailing_newline": lambda name: f"Name: {name}\n",
        "subject_trailing_space": lambda name: f"Name: {name} ",
        "subject_fullwidth_colon": lambda name: f"Name\uff1a{name}",
        "subject_subject_prefix": lambda name: f"Subject: {name}",
        "subject_model_prefix": lambda name: f"Model: {name}",
        "subject_display_name_prefix": lambda name: f"display_name: {name}",
        "subject_markdown_dash": lambda name: f"- Name: {name}",
        "subject_markdown_star": lambda name: f"* Name: {name}",
        "subject_markdown_quote": lambda name: f"> Name: {name}",
        "subject_double_quoted_line": lambda name: f'"Name: {name}"',
        "subject_single_quoted_line": lambda name: f"'Name: {name}'",
        "subject_no_prefix": lambda name: name,
        "subject_unicode_nfd": lambda name: unicodedata.normalize("NFD", f"Name: {name}"),
        "subject_trailing_period": lambda name: f"Name: {name}.",
        "subject_trailing_comma": lambda name: f"Name: {name},",
        "subject_trailing_semicolon": lambda name: f"Name: {name};",
    }


def _benchmark_mutators() -> dict[str, FieldMutator]:
    return {
        "benchmark_original": lambda b: b,
        "benchmark_stripped": lambda b: b.strip(),
        "benchmark_lower": lambda b: b.lower(),
        "benchmark_upper": lambda b: b.upper(),
        "benchmark_title": lambda b: b.title(),
        "benchmark_surrounding_space": lambda b: f" {b} ",
        "benchmark_hyphen_to_underscore": lambda b: b.replace("-", "_"),
        "benchmark_underscore_to_hyphen": lambda b: b.replace("_", "-"),
        "benchmark_hyphen_to_space": lambda b: b.replace("-", " "),
        "benchmark_underscore_to_space": lambda b: b.replace("_", " "),
        "benchmark_insert_separator": lambda b: _insert_benchmark_separator(b),
        "benchmark_unicode_nfd": lambda b: unicodedata.normalize("NFD", b),
    }


def _condition_mutators(include_synonyms: bool) -> dict[str, FieldMutator]:
    mutators: dict[str, FieldMutator] = {
        "condition_original": lambda c: c,
        "condition_stripped": lambda c: c.strip(),
        "condition_lower": lambda c: c.lower(),
        "condition_upper": lambda c: c.upper() if c else c,
        "condition_pipe_both_spaces": lambda c: c.replace("|", " | "),
        "condition_pipe_no_spaces": lambda c: c.replace("|", "|"),
        "condition_pipe_left_space": lambda c: c.replace("|", " |"),
        "condition_pipe_right_space": lambda c: c.replace("|", "| "),
        "condition_unicode_nfd": lambda c: unicodedata.normalize("NFD", c),
        "condition_none_to_empty": lambda c: "" if c.lower() == "none" else c,
        "condition_empty_to_none": lambda c: "None" if c == "" else c,
        "condition_empty_to_null": lambda c: "null" if c == "" else c,
    }
    if include_synonyms:
        mutators.update(
            {
                "condition_zero_to_digit": lambda c: c.replace("zero-shot", "0-shot"),
                "condition_digit_to_zero": lambda c: c.replace("0-shot", "zero-shot"),
                "condition_cot_to_chain": lambda c: c.replace("cot", "chain-of-thought"),
                "condition_chain_to_cot": lambda c: c.replace("chain-of-thought", "cot"),
            }
        )
    return mutators


def _insert_benchmark_separator(benchmark: str) -> str:
    """Create plausible spaced variants for separator-free benchmark ids."""
    known = {
        "afrimedqa": "afri medqa",
        "agentdojo": "agent dojo",
        "androidworld": "android world",
        "livecodebench": "live code bench",
        "matharena": "math arena",
        "mmlupro": "mmlu pro",
        "rewardbench": "reward bench",
        "swebench": "swe bench",
        "ultrafeedback": "ultra feedback",
    }
    lowered = benchmark.lower()
    return known.get(lowered, benchmark)


def _transformations(include_condition_synonyms: bool) -> list[Transformation]:
    transformations: list[Transformation] = []
    for name, mutator in _subject_mutators().items():
        transformations.append(
            Transformation(
                field="subject",
                mutator=name,
                fn=lambda display, benchmark, condition, raw, mutator=mutator: (
                    mutator(display),
                    benchmark,
                    condition,
                ),
            )
        )
    for name, mutator in _benchmark_mutators().items():
        transformations.append(
            Transformation(
                field="benchmark",
                mutator=name,
                fn=lambda display, benchmark, condition, raw, mutator=mutator: (
                    raw,
                    mutator(benchmark),
                    condition,
                ),
            )
        )
    for name, mutator in _condition_mutators(include_condition_synonyms).items():
        transformations.append(
            Transformation(
                field="condition",
                mutator=name,
                fn=lambda display, benchmark, condition, raw, mutator=mutator: (
                    raw,
                    benchmark,
                    mutator(condition),
                ),
            )
        )
    return transformations


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
    include_condition_synonyms: bool,
) -> dict[str, Any]:
    cells = _sample_cells(subject_category_path, n=n, min_count=min_count, seed=seed)
    display_by_key = _display_name_map(subjects_path)
    transformations = _transformations(include_condition_synonyms)
    table: dict[str, Counter[str]] = {t.mutator: Counter() for t in transformations}
    field_table: dict[str, Counter[str]] = {field: Counter() for field in ("subject", "benchmark", "condition")}
    path_counts: Counter[str] = Counter()
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

            for transformation in transformations:
                variant = transformation.fn(display, benchmark, condition, raw_subject)
                if variant is None:
                    table[transformation.mutator]["not_applicable"] += 1
                    field_table[transformation.field]["not_applicable"] += 1
                    path_counts["not_applicable"] += 1
                    continue
                raw_variant, benchmark_variant, condition_variant = variant
                outcome, normalized_subject_key = _lookup_outcome(
                    model,
                    raw_variant,
                    benchmark_variant,
                    condition_variant,
                )
                table[transformation.mutator][outcome] += 1
                field_table[transformation.field][outcome] += 1
                path_counts[outcome] += 1
                if outcome != "hit_subject_category":
                    _examples_append(
                        examples,
                        transformation.mutator,
                        {
                            "field": transformation.field,
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
        "include_condition_synonyms": bool(include_condition_synonyms),
        "missing_display_keys": missing_display_keys,
        "path_counts": dict(sorted(path_counts.items())),
        "field_confusion_table": {
            name: dict(sorted(counter.items()))
            for name, counter in field_table.items()
        },
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
        default="data/stage2/priors_v1_locked/subject_category.parquet",
        type=Path,
    )
    parser.add_argument("--subjects", default="data/subjects.parquet", type=Path)
    parser.add_argument("--out", default="reports/lookup_audit_synthetic.json", type=Path)
    parser.add_argument("--n", default=200, type=int)
    parser.add_argument("--min-count", default=20, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--include-condition-synonyms",
        action="store_true",
        help="Include aggressive condition synonym mutators; intended only when the runtime ships those aliases.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload = run_audit(
        zip_path=args.zip,
        subject_category_path=args.subject_category,
        subjects_path=args.subjects,
        out_path=args.out,
        n=args.n,
        min_count=args.min_count,
        seed=args.seed,
        include_condition_synonyms=args.include_condition_synonyms,
    )
    print(_format_table(payload["confusion_table"]))
    print("field_confusion")
    print(_format_table(payload["field_confusion_table"]))
    print("path_counts\t" + "\t".join(f"{k}={v}" for k, v in payload["path_counts"].items()))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
