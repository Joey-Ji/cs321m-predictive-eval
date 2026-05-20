"""Local Codabench-style proxy eval for a packaged submission ZIP.

This mirrors modal_eval_submission.py closely enough to use when Modal is not
available. It imports model.py from the ZIP, samples held-out item-cold rows,
passes a pooled labeled list to predict(), and writes per-seed metrics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import statistics
import sys
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

INPUT_KEYS = ("subject_content", "item_content", "benchmark", "condition")
CATEGORY_KEYS = ("benchmark", "condition")


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _coerce_binary_label(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value if value in (0, 1) else None
    if isinstance(value, float) and math.isfinite(value) and value in (0.0, 1.0):
        return int(value)
    return None


def _input_from_row(row: Mapping[str, Any]) -> dict[str, str]:
    return {key: _clean_str(row.get(key)) for key in INPUT_KEYS}


def _labeled_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    label = _coerce_binary_label(row.get("label"))
    if label is None:
        raise ValueError(f"row has non-binary label: {row.get('label')!r}")
    return {**_input_from_row(row), "label": label}


def _category_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return tuple(_clean_str(row.get(key)) for key in CATEGORY_KEYS)  # type: ignore[return-value]


def _format_category(category: tuple[str, str]) -> str:
    benchmark, condition = category
    return f"{benchmark}::{condition}"


def _group_rows_by_category(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_category_key(row), []).append(row)
    return groups


def _allocate_category_quotas(
    groups: Mapping[tuple[str, str], list[dict[str, Any]]],
    max_rows: int,
    max_per_category: int,
) -> dict[tuple[str, str], int]:
    caps = {category: min(len(rows), max_per_category) for category, rows in groups.items() if rows}
    quotas = {category: 0 for category in caps}
    remaining = min(max_rows, sum(caps.values()))
    active = sorted(caps)
    while remaining > 0 and active:
        next_active: list[tuple[str, str]] = []
        for category in active:
            if remaining <= 0:
                break
            if quotas[category] >= caps[category]:
                continue
            quotas[category] += 1
            remaining -= 1
            if quotas[category] < caps[category]:
                next_active.append(category)
        active = next_active
    return quotas


def _sample_groups(
    groups: Mapping[tuple[str, str], list[dict[str, Any]]],
    rng: random.Random,
    max_rows: int,
    max_per_category: int,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    quotas = _allocate_category_quotas(groups, max_rows, max_per_category)
    sampled: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for category in sorted(quotas):
        quota = quotas[category]
        if quota <= 0:
            continue
        rows = groups[category]
        sampled[category] = list(rows) if len(rows) <= quota else rng.sample(rows, quota)
    return sampled


def _select_labeled_rows(
    rows: list[dict[str, Any]],
    rng: random.Random,
    k: int,
    labeling: ModuleType | None,
) -> tuple[list[dict[str, Any]], str]:
    if not rows or k == 0:
        return [], "none"
    n_reveal = min(k, len(rows))
    acquisition = getattr(labeling, "acquisition_function", None) if labeling is not None else None
    if callable(acquisition):
        try:
            scored = [
                (float(acquisition(_input_from_row(row))), idx, row)
                for idx, row in enumerate(rows)
            ]
            if not all(math.isfinite(score) for score, _, _ in scored):
                raise ValueError("non-finite acquisition score")
            selected = [row for _, _, row in sorted(scored, key=lambda x: (-x[0], x[1]))[:n_reveal]]
            return [_labeled_from_row(row) for row in selected], "labeling.py"
        except Exception as exc:  # noqa: BLE001
            print(f"[local-eval] labeling fallback to random reveal: {exc!r}", flush=True)
    return [_labeled_from_row(row) for row in rng.sample(rows, n_reveal)], "random"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _load_held_out_item_ids(
    stage1_dir: Path,
    stage2_dir: Path,
    submission_dir: Path,
    val_frac: float,
    seed: int,
) -> set[str]:
    from src.kfactor import validation_item_ids

    head_meta = _load_json(stage2_dir / "head_meta.json")
    if not head_meta:
        head_meta = _load_json(submission_dir / "head_meta.json")
    if "val_item_ids" in head_meta:
        held_out = {str(iid) for iid in head_meta["val_item_ids"]}
        if held_out:
            return held_out
    split_val_frac = float(head_meta.get("val_frac", val_frac))
    split_seed = int(head_meta.get("seed", seed))
    item_to_id = _load_json(stage1_dir / "item_to_id.json") or _load_json(submission_dir / "item_to_id.json")
    if not item_to_id:
        raise FileNotFoundError("item_to_id.json not found in stage1 dir or submission ZIP")
    return validation_item_ids([str(iid) for iid in item_to_id], split_val_frac, split_seed)


def _load_eval_rows(joined_path: Path, held_out_item_ids: set[str]) -> tuple[list[dict[str, Any]], int]:
    import pyarrow.parquet as pq

    columns = ["item_id", "subject_content", "item_content", "benchmark", "condition", "label"]
    table = pq.read_table(joined_path, columns=columns)
    rows: list[dict[str, Any]] = []
    skipped_nonbinary = 0
    for row in table.to_pylist():
        item_id = _clean_str(row.get("item_id"))
        if item_id not in held_out_item_ids:
            continue
        label = _coerce_binary_label(row.get("label"))
        if label is None:
            skipped_nonbinary += 1
            continue
        clean_row = {key: _clean_str(row.get(key)) for key in ("item_id", *INPUT_KEYS)}
        clean_row["label"] = label
        rows.append(clean_row)
    return rows, skipped_nonbinary


def _score_predictions(predictions: list[float], labels: list[int]) -> dict[str, Any]:
    from src.validation import auc_roc, mean_log_likelihood

    if not predictions:
        raise ValueError("no predictions to score")
    return {
        "mean_log_likelihood": float(mean_log_likelihood(predictions, labels)),
        "auc_roc": float(auc_roc(predictions, labels)),
        "n": int(len(predictions)),
        "p_pos": float(sum(labels) / len(labels)),
    }


def _checked_predict(model: ModuleType, row: Mapping[str, Any], labeled: list[dict[str, Any]]) -> float:
    pred = model.predict(_input_from_row(row), labeled=labeled)
    if isinstance(pred, bool) or not isinstance(pred, float):
        raise TypeError(f"predict() returned {pred!r} ({type(pred).__name__}), expected Python float")
    if not math.isfinite(pred) or not 0.0 <= pred <= 1.0:
        raise ValueError(f"predict() returned invalid probability {pred!r}")
    return pred


def _run_seed(
    model: ModuleType,
    labeling: ModuleType | None,
    rows: list[dict[str, Any]],
    seed: int,
    max_rows: int,
    max_per_category: int,
    k: int,
    m_categories: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    sampled_groups = _sample_groups(
        _group_rows_by_category(rows),
        rng,
        max_rows=max_rows,
        max_per_category=max_per_category,
    )
    all_categories = sorted(sampled_groups)
    reveal_categories = rng.sample(all_categories, min(m_categories, len(all_categories)))
    labeled_pool: list[dict[str, Any]] = []
    per_category_labeled_counts = {category: 0 for category in all_categories}
    reveal_methods: dict[str, str] = {}
    for category in reveal_categories:
        labeled_for_category, reveal_method = _select_labeled_rows(sampled_groups[category], rng, k, labeling)
        labeled_pool.extend(labeled_for_category)
        per_category_labeled_counts[category] = len(labeled_for_category)
        reveal_methods[_format_category(category)] = reveal_method

    all_predictions: list[float] = []
    all_labels: list[int] = []
    categories: dict[str, dict[str, Any]] = {}
    for category in all_categories:
        category_predictions: list[float] = []
        category_labels: list[int] = []
        for row in sampled_groups[category]:
            category_predictions.append(_checked_predict(model, row, labeled_pool))
            label = _coerce_binary_label(row.get("label"))
            if label is None:
                raise ValueError(f"sampled row has non-binary label: {row.get('label')!r}")
            category_labels.append(label)
        categories[_format_category(category)] = {
            **_score_predictions(category_predictions, category_labels),
            "n_labeled_for_category": float(per_category_labeled_counts[category]),
            "n_labeled_pool": float(len(labeled_pool)),
        }
        all_predictions.extend(category_predictions)
        all_labels.extend(category_labels)
    return {
        "seed": int(seed),
        "metrics": _score_predictions(all_predictions, all_labels),
        "categories": categories,
        "reveal_methods": reveal_methods,
    }


def _mean_std(values: list[float]) -> tuple[float, float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return float("nan"), float("nan")
    if len(finite) == 1:
        return finite[0], 0.0
    return float(statistics.fmean(finite)), float(statistics.stdev(finite))


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    mll_mean, mll_std = _mean_std([r["metrics"]["mean_log_likelihood"] for r in results])
    auc_mean, auc_std = _mean_std([r["metrics"]["auc_roc"] for r in results])
    return {
        "seeds": [int(r["seed"]) for r in results],
        "mean_log_likelihood_mean": mll_mean,
        "mean_log_likelihood_std": mll_std,
        "auc_roc_mean": auc_mean,
        "auc_roc_std": auc_std,
        "n_min": min(int(r["metrics"]["n"]) for r in results),
        "n_max": max(int(r["metrics"]["n"]) for r in results),
    }


def _import_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def main(
    zip_path: Path,
    joined: Path,
    stage1: Path,
    stage2: Path,
    seeds: list[int],
    max_rows: int,
    per_category: int,
    k: int,
    m_categories: int,
    val_frac: float,
    split_seed: int,
    out: Path | None,
) -> None:
    with tempfile.TemporaryDirectory(prefix="item-residual-local-eval-") as tmp_str:
        tmp = Path(tmp_str)
        sub = tmp / "sub"
        sub.mkdir()
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(sub)
        sys.path.insert(0, str(sub))
        model = _import_module(sub / "model.py", "local_eval_submission_model")
        labeling = _import_module(sub / "labeling.py", "local_eval_submission_labeling") if (sub / "labeling.py").exists() else None

        held_out = _load_held_out_item_ids(stage1, stage2, sub, val_frac=val_frac, seed=split_seed)
        rows, skipped_nonbinary = _load_eval_rows(joined, held_out)
        print(
            f"[local-eval] held_out_items={len(held_out):,} rows={len(rows):,} "
            f"skipped_nonbinary={skipped_nonbinary}",
            flush=True,
        )
        groups = _group_rows_by_category(rows)
        precompute = getattr(model, "_precompute_item_residual_embeddings", None)
        if callable(precompute):
            precompute_inputs: list[dict[str, str]] = []
            for seed in seeds:
                sampled = _sample_groups(
                    groups,
                    random.Random(seed),
                    max_rows=max_rows,
                    max_per_category=per_category,
                )
                for category in sorted(sampled):
                    precompute_inputs.extend(_input_from_row(row) for row in sampled[category])
            print(f"[local-eval] precomputing embeddings for {len(precompute_inputs)} sampled rows", flush=True)
            print(f"[local-eval] item residual precompute: {precompute(precompute_inputs, batch_size=32)}", flush=True)
        results = []
        for seed in seeds:
            result = _run_seed(model, labeling, rows, seed, max_rows, per_category, k, m_categories)
            metrics = result["metrics"]
            print(
                f"[local-eval] seed={seed} mll={metrics['mean_log_likelihood']:.6f} "
                f"auc={metrics['auc_roc']:.6f} n={metrics['n']}",
                flush=True,
            )
            results.append(result)
        summary = _summarize(results)
        summary.update(
            {
                "split": "item-cold",
                "zip": str(zip_path),
                "max_rows": int(max_rows),
                "max_per_category": int(per_category),
                "k": int(k),
                "m_categories": int(m_categories),
                "n_held_out_items": int(len(held_out)),
                "n_held_out_rows": int(len(rows)),
                "skipped_nonbinary": int(skipped_nonbinary),
            }
        )
        payload = {"summary": summary, "results": results}
        print(
            "local_eval_submission "
            f"split=item-cold seeds={','.join(str(seed) for seed in seeds)} "
            f"mll={summary['mean_log_likelihood_mean']:.6f}+/-{summary['mean_log_likelihood_std']:.6f} "
            f"auc={summary['auc_roc_mean']:.6f}+/-{summary['auc_roc_std']:.6f}",
            flush=True,
        )
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default="submissions/v1_kfactor.zip", type=Path)
    parser.add_argument("--joined", default="data/joined.parquet", type=Path)
    parser.add_argument("--stage1", default="data/stage1/kfactor_k4", type=Path)
    parser.add_argument("--stage2", default="data/stage2/kfactor_mpnet_mlp_v1", type=Path)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--max-rows", default=5000, type=int)
    parser.add_argument("--per-category", default=1000, type=int)
    parser.add_argument("--k", default=5, type=int)
    parser.add_argument("--m-categories", default=5, type=int)
    parser.add_argument("--val-frac", default=0.1, type=float)
    parser.add_argument("--split-seed", default=0, type=int)
    parser.add_argument("--out", default=None, type=Path)
    args = parser.parse_args()
    main(
        args.zip,
        args.joined,
        args.stage1,
        args.stage2,
        _parse_seeds(args.seeds),
        args.max_rows,
        args.per_category,
        args.k,
        args.m_categories,
        args.val_frac,
        args.split_seed,
        args.out,
    )
