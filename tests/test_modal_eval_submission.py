"""Unit tests for the Modal Codabench-style submission eval helpers.

Run:
    python tests/test_modal_eval_submission.py
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import modal_eval_submission as mes
from src.kfactor import validation_item_ids


def test_category_quota_balances_and_caps() -> list[str]:
    errs: list[str] = []
    groups = {
        ("b", "zero-shot"): [{"i": i} for i in range(5)],
        ("a", "zero-shot"): [{"i": i} for i in range(5)],
        ("c", "cot"): [{"i": 0}],
    }
    quotas = mes._allocate_category_quotas(groups, max_rows=5, max_per_category=3)
    expected = {
        ("a", "zero-shot"): 2,
        ("b", "zero-shot"): 2,
        ("c", "cot"): 1,
    }
    if quotas != expected:
        errs.append(f"quotas={quotas} expected={expected}")
    if sum(quotas.values()) > 5:
        errs.append("quotas exceeded max_rows")
    if any(q > 3 for q in quotas.values()):
        errs.append("quota exceeded max_per_category")
    return errs


def test_labeling_selection_uses_top_k() -> list[str]:
    errs: list[str] = []
    rows = [
        {
            "subject_content": "Name: s",
            "item_content": f"item {i}",
            "benchmark": "bench",
            "condition": "zero-shot",
            "label": i % 2,
        }
        for i in range(4)
    ]
    labeling = ModuleType("labeling")

    def acquisition_function(input: dict) -> float:
        return float(input["item_content"].split()[-1])

    labeling.acquisition_function = acquisition_function
    labeled, method = mes._select_labeled_rows(rows, random.Random(0), k=2, labeling=labeling)
    items = [row["item_content"] for row in labeled]
    if method != "labeling.py":
        errs.append(f"method={method!r} expected labeling.py")
    if items != ["item 3", "item 2"]:
        errs.append(f"selected items={items} expected ['item 3', 'item 2']")
    if any(set(row) != set(mes.INPUT_KEYS) | {"label"} for row in labeled):
        errs.append(f"labeled rows have wrong keys: {labeled}")
    return errs


def test_run_seed_passes_pooled_labeled_list_to_every_predict() -> list[str]:
    """Per competition spec, the full K * m labeled list (union of K labels
    selected per category) must be passed to every predict() call within the
    round, not the per-category slice. This test pins that invariant."""
    errs: list[str] = []
    rows: list[dict] = []
    for benchmark in ("bench_a", "bench_b"):
        for i in range(3):
            rows.append(
                {
                    "subject_content": f"Name: subject {i}",
                    "item_content": f"{benchmark} item {i}",
                    "benchmark": benchmark,
                    "condition": "zero-shot",
                    "label": i % 2,
                }
            )

    calls: list[tuple[str, int, tuple[str, ...]]] = []
    model = ModuleType("model")

    def predict(input: dict, labeled: list[dict] | None = None) -> float:
        labels = labeled or []
        calls.append(
            (
                input["benchmark"],
                id(labels),
                tuple(sorted(row["benchmark"] for row in labels)),
            )
        )
        return 0.75 if input["benchmark"] == "bench_a" else 0.25

    model.predict = predict
    result = mes._run_seed(
        model,
        labeling=None,
        rows=rows,
        seed=7,
        max_rows=6,
        max_per_category=3,
        k=2,
        m_categories=2,
    )
    if int(result["metrics"]["n"]) != 6:
        errs.append(f"n={result['metrics']['n']} expected 6")

    # All 6 predict() calls should have seen the SAME labeled list object
    # (single pool), and that pool should span both benchmarks (k=2 per
    # category * 2 categories = 4 labeled rows total, with both benchmarks
    # represented).
    if len(calls) != 6:
        errs.append(f"call count={len(calls)} expected 6")
    pool_ids = {call[1] for call in calls}
    if len(pool_ids) != 1:
        errs.append(f"labeled list identity not shared across all calls: {pool_ids}")
    pool_benchmark_tuples = {call[2] for call in calls}
    if len(pool_benchmark_tuples) != 1:
        errs.append(f"different labeled pools observed: {pool_benchmark_tuples}")
    pool_benchmarks = next(iter(pool_benchmark_tuples))
    if set(pool_benchmarks) != {"bench_a", "bench_b"}:
        errs.append(f"pool missing categories: {pool_benchmarks}")
    if len(pool_benchmarks) != 4:
        errs.append(f"pool size={len(pool_benchmarks)} expected 4 (k=2 * 2 cats)")
    return errs


def test_load_held_out_item_ids_matches_stage2_logic() -> list[str]:
    errs: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        stage1 = root / "stage1"
        stage2 = root / "stage2"
        sub = root / "sub"
        stage1.mkdir()
        stage2.mkdir()
        sub.mkdir()
        item_to_id = {f"item_{i}": i for i in range(10)}
        (stage1 / "item_to_id.json").write_text(json.dumps(item_to_id))
        (stage2 / "head_meta.json").write_text(json.dumps({"val_frac": 0.3, "seed": 11}))
        held = mes._load_held_out_item_ids(stage1, stage2, sub, val_frac=0.1, seed=0)
        expected = validation_item_ids(list(item_to_id), 0.3, 11)
        if held != expected:
            errs.append(f"held={held} expected={expected}")

        (stage2 / "head_meta.json").write_text(json.dumps({"val_item_ids": ["explicit"]}))
        held = mes._load_held_out_item_ids(stage1, stage2, sub, val_frac=0.1, seed=0)
        if held != {"explicit"}:
            errs.append(f"explicit held={held} expected={{'explicit'}}")
    return errs


def test_load_eval_rows_filters_heldout_and_nonbinary() -> list[str]:
    errs: list[str] = []
    import pyarrow as pa
    import pyarrow.parquet as pq

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "joined.parquet"
        table = pa.table(
            {
                "item_id": ["keep", "drop", "keep"],
                "subject_content": ["Name: a", "Name: b", None],
                "item_content": ["item a", "item b", "item c"],
                "benchmark": ["bench", "bench", "bench"],
                "condition": ["zero-shot", "zero-shot", None],
                "label": [1, 0, 0.5],
            }
        )
        pq.write_table(table, path)
        rows, skipped = mes._load_eval_rows(path, {"keep"})
    if skipped != 1:
        errs.append(f"skipped={skipped} expected 1")
    if len(rows) != 1:
        errs.append(f"len(rows)={len(rows)} expected 1")
    elif rows[0]["item_id"] != "keep" or rows[0]["label"] != 1:
        errs.append(f"wrong loaded row: {rows[0]}")
    return errs


def test_parse_seeds() -> list[str]:
    errs: list[str] = []
    if mes._parse_seeds("0, 2,4") != [0, 2, 4]:
        errs.append("failed to parse comma-separated seeds")
    try:
        mes._parse_seeds(" , ")
    except ValueError:
        pass
    else:
        errs.append("empty seed list did not raise")
    return errs


def main() -> int:
    all_errs: list[str] = []
    tests = [
        ("category_quota_balances_and_caps", test_category_quota_balances_and_caps),
        ("labeling_selection_uses_top_k", test_labeling_selection_uses_top_k),
        ("run_seed_passes_pooled_labeled_list_to_every_predict", test_run_seed_passes_pooled_labeled_list_to_every_predict),
        ("load_held_out_item_ids_matches_stage2_logic", test_load_held_out_item_ids_matches_stage2_logic),
        ("load_eval_rows_filters_heldout_and_nonbinary", test_load_eval_rows_filters_heldout_and_nonbinary),
        ("parse_seeds", test_parse_seeds),
    ]
    for name, test in tests:
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
    print("\nPASS - modal eval submission unit tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
