"""Modal Codabench-style MLL eval for a packaged submission.

This is the scoring sibling of modal_smoke_submission.py. It imports model.py
from the submitted ZIP, runs in offline HF mode, samples held-out item-cold rows
from /data/joined.parquet, reveals K labels per benchmark/condition category,
passes the same labeled list to every prediction in that category, and reports
mean log-likelihood plus AUC.

USAGE:
    modal run modal_eval_submission.py
    modal run modal_eval_submission.py --zip submissions/v1_kfactor.zip --seeds 0,1,2
    modal run modal_eval_submission.py --max-rows 5000 --per-category 1000 --k 5
"""

from __future__ import annotations

import importlib.util
import json
import math
import random
import statistics
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import modal

APP_NAME = "eval-comp-submission-eval"
VOLUME_NAME = "eval-comp-data"
HF_CACHE = "/app/hf_cache"
ENCODER = "sentence-transformers/all-mpnet-base-v2"
WORKDIR = "/root/eval_comp"

LOCAL_ROOT = Path(__file__).resolve().parent
INPUT_KEYS = ("subject_content", "item_content", "benchmark", "condition")
CATEGORY_KEYS = ("benchmark", "condition")


def _prefetch_encoder() -> None:
    """Pre-download the encoder into HF_CACHE, matching the smoke harness."""
    import os

    os.environ["HF_HOME"] = HF_CACHE
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(ENCODER, cache_folder=HF_CACHE)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2",
        "sentence-transformers>=3.0",
        "transformers>=4.41",
        "pyarrow>=15",
        "pandas>=2.0",
        "numpy>=1.26",
        "scikit-learn>=1.4",
        "huggingface-hub>=0.24",
    )
    .env({"HF_HOME": HF_CACHE})
    .run_function(_prefetch_encoder)
    .add_local_dir(str(LOCAL_ROOT / "src"), remote_path=f"{WORKDIR}/src")
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
app = modal.App(APP_NAME)


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
    if max_rows <= 0:
        raise ValueError(f"max_rows must be > 0, got {max_rows}")
    if max_per_category <= 0:
        raise ValueError(f"max_per_category must be > 0, got {max_per_category}")

    caps = {
        category: min(len(rows), max_per_category)
        for category, rows in groups.items()
        if rows
    }
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
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if not rows or k == 0:
        return [], "none"

    n_reveal = min(k, len(rows))
    acquisition = getattr(labeling, "acquisition_function", None) if labeling is not None else None
    if callable(acquisition):
        scored: list[tuple[float, int, dict[str, Any]]] = []
        try:
            for idx, row in enumerate(rows):
                score = float(acquisition(_input_from_row(row)))
                if not math.isfinite(score):
                    raise ValueError(f"non-finite acquisition score {score!r}")
                scored.append((score, idx, row))
        except Exception as exc:  # noqa: BLE001
            print(f"[eval] labeling.py fallback to random reveal: {exc!r}", flush=True)
        else:
            selected = [row for _, _, row in sorted(scored, key=lambda x: (-x[0], x[1]))[:n_reveal]]
            return [_labeled_from_row(row) for row in selected], "labeling.py"

    selected = rng.sample(rows, n_reveal)
    return [_labeled_from_row(row) for row in selected], "random"


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
        if not held_out:
            raise ValueError("head_meta.json contains empty val_item_ids")
        return held_out

    split_val_frac = float(head_meta.get("val_frac", val_frac))
    split_seed = int(head_meta.get("seed", seed))
    item_to_id = _load_json(stage1_dir / "item_to_id.json")
    if not item_to_id:
        item_to_id = _load_json(submission_dir / "item_to_id.json")
    if not item_to_id:
        raise FileNotFoundError(
            "item_to_id.json not found in stage1 dir or submission ZIP; "
            "cannot reproduce evaluate_stage2.py item-cold split"
        )
    return validation_item_ids([str(iid) for iid in item_to_id.keys()], split_val_frac, split_seed)


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
) -> dict[str, Any]:
    rng = random.Random(seed)
    groups = _group_rows_by_category(rows)
    sampled_groups = _sample_groups(groups, rng, max_rows=max_rows, max_per_category=max_per_category)
    if not sampled_groups:
        raise ValueError("no sampled eval rows after category grouping")

    all_predictions: list[float] = []
    all_labels: list[int] = []
    categories: dict[str, dict[str, Any]] = {}
    reveal_methods: dict[str, str] = {}

    for category in sorted(sampled_groups):
        category_rows = sampled_groups[category]
        labeled, reveal_method = _select_labeled_rows(category_rows, rng, k=k, labeling=labeling)
        category_predictions: list[float] = []
        category_labels: list[int] = []
        for row in category_rows:
            pred = _checked_predict(model, row, labeled=labeled)
            label = _coerce_binary_label(row.get("label"))
            if label is None:
                raise ValueError(f"sampled row has non-binary label: {row.get('label')!r}")
            category_predictions.append(pred)
            category_labels.append(label)

        category_id = _format_category(category)
        categories[category_id] = {
            **_score_predictions(category_predictions, category_labels),
            "n_labeled": float(len(labeled)),
        }
        reveal_methods[category_id] = reveal_method
        all_predictions.extend(category_predictions)
        all_labels.extend(category_labels)

    return {
        "seed": seed,
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


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite_or_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite_or_none(v) for v in value]
    return value


def _summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    mll_mean, mll_std = _mean_std([r["metrics"]["mean_log_likelihood"] for r in results])
    auc_mean, auc_std = _mean_std([r["metrics"]["auc_roc"] for r in results])
    n_values = [int(r["metrics"]["n"]) for r in results]
    p_pos_mean, p_pos_std = _mean_std([r["metrics"]["p_pos"] for r in results])

    categories = sorted({category for result in results for category in result["categories"]})
    per_category: dict[str, dict[str, float]] = {}
    for category in categories:
        category_results = [result["categories"][category] for result in results if category in result["categories"]]
        cat_mll_mean, cat_mll_std = _mean_std([r["mean_log_likelihood"] for r in category_results])
        cat_auc_mean, cat_auc_std = _mean_std([r["auc_roc"] for r in category_results])
        per_category[category] = {
            "mll_mean": cat_mll_mean,
            "mll_std": cat_mll_std,
            "auc_mean": cat_auc_mean,
            "auc_std": cat_auc_std,
            "n_mean": float(statistics.fmean([r["n"] for r in category_results])),
        }

    return {
        "seeds": [int(r["seed"]) for r in results],
        "mean_log_likelihood_mean": mll_mean,
        "mean_log_likelihood_std": mll_std,
        "auc_roc_mean": auc_mean,
        "auc_roc_std": auc_std,
        "n_min": min(n_values),
        "n_max": max(n_values),
        "p_pos_mean": p_pos_mean,
        "p_pos_std": p_pos_std,
        "per_category": per_category,
    }


def _import_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_seeds(seed_csv: str) -> list[int]:
    seeds = [int(part.strip()) for part in seed_csv.split(",") if part.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


@app.function(image=image, volumes={"/data": volume}, timeout=1800)
def eval_submission(
    zip_bytes: bytes,
    seed_csv: str = "0,1,2",
    max_rows: int = 5000,
    max_per_category: int = 1000,
    k: int = 5,
    val_frac: float = 0.1,
    split_seed: int = 0,
    stage1_path: str = "/data/stage1/kfactor_k4",
    stage2_path: str = "/data/stage2/kfactor_mpnet_linear_v1",
) -> dict[str, Any]:
    import os
    import sys
    import tempfile
    import traceback
    import zipfile
    from pathlib import Path as P

    os.environ["HF_HOME"] = HF_CACHE
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path.insert(0, WORKDIR)

    tmp = P(tempfile.mkdtemp())
    zpath = tmp / "sub.zip"
    zpath.write_bytes(zip_bytes)
    sub = tmp / "sub"
    sub.mkdir()
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(sub)

    print(f"[eval] submission unzipped to {sub}")
    print(f"[eval] HF_HOME={os.environ['HF_HOME']}  OFFLINE=1")
    print(f"[eval] /app/hf_cache contents: {sorted(p.name for p in P(HF_CACHE).iterdir())[:5]}")

    sys.path.insert(0, str(sub))
    try:
        model = _import_module(sub / "model.py", "model")
    except Exception:
        print("[eval] FAIL: model.py raised at import time")
        traceback.print_exc()
        raise
    if not hasattr(model, "predict"):
        raise AttributeError("model.py did not define predict")
    print(f"[eval] model.py imported (ENCODER_OK={getattr(model, 'ENCODER_OK', '?')})")

    labeling = None
    if (sub / "labeling.py").exists():
        try:
            labeling = _import_module(sub / "labeling.py", "submission_labeling")
            print("[eval] labeling.py imported")
        except Exception:
            print("[eval] labeling.py import failed; using random reveal fallback")
            traceback.print_exc()

    held_out = _load_held_out_item_ids(
        stage1_dir=P(stage1_path),
        stage2_dir=P(stage2_path),
        submission_dir=sub,
        val_frac=val_frac,
        seed=split_seed,
    )
    rows, skipped_nonbinary = _load_eval_rows(P("/data/joined.parquet"), held_out)
    if not rows:
        raise ValueError("no binary held-out rows found for item-cold eval")
    groups = _group_rows_by_category(rows)
    print(
        "[eval] held-out "
        f"items={len(held_out)} rows={len(rows)} categories={len(groups)} "
        f"skipped_nonbinary={skipped_nonbinary}"
    )

    seeds = _parse_seeds(seed_csv)
    results = []
    for seed in seeds:
        result = _run_seed(
            model,
            labeling,
            rows,
            seed=seed,
            max_rows=max_rows,
            max_per_category=max_per_category,
            k=k,
        )
        metrics = result["metrics"]
        print(
            "[eval] seed "
            f"{seed}: mll={metrics['mean_log_likelihood']:.6f} "
            f"auc={metrics['auc_roc']:.6f} n={int(metrics['n'])} "
            f"categories={len(result['categories'])}"
        )
        results.append(result)

    summary = _summarize_results(results)
    summary.update(
        {
            "split": "item-cold",
            "max_rows": int(max_rows),
            "max_per_category": int(max_per_category),
            "k": int(k),
            "n_held_out_items": int(len(held_out)),
            "n_held_out_rows": int(len(rows)),
            "n_categories": int(len(groups)),
            "skipped_nonbinary": int(skipped_nonbinary),
        }
    )
    printable = _finite_or_none(summary)
    print(
        "modal_eval_submission "
        f"split=item-cold seeds={','.join(str(s) for s in seeds)} "
        f"mll={summary['mean_log_likelihood_mean']:.6f}+/-{summary['mean_log_likelihood_std']:.6f} "
        f"auc={summary['auc_roc_mean']:.6f}+/-{summary['auc_roc_std']:.6f} "
        f"per_category={json.dumps(printable['per_category'], sort_keys=True)}"
    )
    return printable


@app.local_entrypoint()
def main(
    zip: str = "submissions/v1_kfactor.zip",
    seeds: str = "0,1,2",
    max_rows: int = 5000,
    per_category: int = 1000,
    k: int = 5,
    val_frac: float = 0.1,
    split_seed: int = 0,
    stage1: str = "/data/stage1/kfactor_k4",
    stage2: str = "/data/stage2/kfactor_mpnet_linear_v1",
) -> None:
    path = (LOCAL_ROOT / zip).resolve() if not Path(zip).is_absolute() else Path(zip)
    if not path.exists():
        raise SystemExit(f"submission zip not found: {path}")
    _parse_seeds(seeds)
    print(
        f"[local] uploading {path.name} ({path.stat().st_size / 1024:.1f} KB) "
        f"and running eval (seeds={seeds}, max_rows={max_rows}, per_category={per_category}, k={k})"
    )
    eval_submission.remote(
        path.read_bytes(),
        seeds,
        max_rows,
        per_category,
        k,
        val_frac,
        split_seed,
        stage1,
        stage2,
    )
