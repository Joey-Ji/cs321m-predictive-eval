"""Local fallback for modal_eval_submission.py using the same helper functions.

This is intended for prior-only submissions when Modal upload is unavailable.
It sets V1_KFACTOR_DUMMY_ENCODER=1 by default; the prior-only prediction path
does not use encoder embeddings, but predict() requires ENCODER_OK to be true.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import modal_eval_submission as mes  # noqa: E402


def _find_data_root() -> Path:
    for base in (ROOT, *ROOT.parents):
        candidate = base / "data" / "joined.parquet"
        if candidate.exists():
            return base / "data"
    return ROOT / "data"


DATA_ROOT = _find_data_root()


def _import_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_submission(zip_path: Path, tmpdir: Path, dummy_encoder: bool) -> tuple[ModuleType, ModuleType | None]:
    sub = tmpdir / "sub"
    sub.mkdir()
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(sub)

    previous_dummy = os.environ.get("V1_KFACTOR_DUMMY_ENCODER")
    if dummy_encoder:
        os.environ["V1_KFACTOR_DUMMY_ENCODER"] = "1"
    sys.path.insert(0, str(sub))
    try:
        with contextlib.redirect_stdout(sys.stderr):
            model = _import_module(sub / "model.py", "local_eval_model")
        if not hasattr(model, "predict"):
            raise AttributeError("model.py did not define predict")
        labeling = None
        if (sub / "labeling.py").exists():
            try:
                labeling = _import_module(sub / "labeling.py", "local_eval_labeling")
            except Exception:  # noqa: BLE001
                print("[local-eval] labeling.py import failed; using random reveal fallback", flush=True)
                traceback.print_exc()
        return model, labeling
    finally:
        try:
            sys.path.remove(str(sub))
        except ValueError:
            pass
        if previous_dummy is None:
            os.environ.pop("V1_KFACTOR_DUMMY_ENCODER", None)
        else:
            os.environ["V1_KFACTOR_DUMMY_ENCODER"] = previous_dummy


def run_local_eval(
    zip_path: Path,
    seeds: str,
    max_rows: int,
    per_category: int,
    k: int,
    m_categories: int,
    val_frac: float,
    split_seed: int,
    stage1: Path,
    stage2: Path,
    joined: Path,
    dummy_encoder: bool,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="local-submission-eval-") as tmp:
        tmpdir = Path(tmp)
        model, labeling = _load_submission(zip_path, tmpdir, dummy_encoder=dummy_encoder)
        sub = tmpdir / "sub"
        held_out = mes._load_held_out_item_ids(
            stage1_dir=stage1,
            stage2_dir=stage2,
            submission_dir=sub,
            val_frac=val_frac,
            seed=split_seed,
        )
        rows, skipped_nonbinary = mes._load_eval_rows(joined, held_out)
        groups = mes._group_rows_by_category(rows)
        seed_values = mes._parse_seeds(seeds)
        results = [
            mes._run_seed(
                model,
                labeling,
                rows,
                seed=seed,
                max_rows=max_rows,
                max_per_category=per_category,
                k=k,
                m_categories=m_categories,
            )
            for seed in seed_values
        ]
        summary = mes._summarize_results(results)
        summary.update(
            {
                "split": "item-cold",
                "max_rows": int(max_rows),
                "max_per_category": int(per_category),
                "k": int(k),
                "m_categories": int(m_categories),
                "n_held_out_items": int(len(held_out)),
                "n_held_out_rows": int(len(rows)),
                "n_categories": int(len(groups)),
                "skipped_nonbinary": int(skipped_nonbinary),
                "local_dummy_encoder": bool(dummy_encoder),
                "zip": str(zip_path),
                "stage1": str(stage1),
                "stage2": str(stage2),
                "joined": str(joined),
                "results": results,
            }
        )
        return mes._finite_or_none(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--max-rows", default=5000, type=int)
    parser.add_argument("--per-category", default=1000, type=int)
    parser.add_argument("--k", default=5, type=int)
    parser.add_argument("--m-categories", default=5, type=int)
    parser.add_argument("--val-frac", default=0.1, type=float)
    parser.add_argument("--split-seed", default=0, type=int)
    parser.add_argument("--stage1", default=DATA_ROOT / "stage1/kfactor_k4", type=Path)
    parser.add_argument("--stage2", default=DATA_ROOT / "stage2/kfactor_mpnet_linear_v1", type=Path)
    parser.add_argument("--joined", default=DATA_ROOT / "joined.parquet", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--real-encoder", action="store_true")
    args = parser.parse_args()

    payload = run_local_eval(
        zip_path=args.zip,
        seeds=args.seeds,
        max_rows=args.max_rows,
        per_category=args.per_category,
        k=args.k,
        m_categories=args.m_categories,
        val_frac=args.val_frac,
        split_seed=args.split_seed,
        stage1=args.stage1,
        stage2=args.stage2,
        joined=args.joined,
        dummy_encoder=not args.real_encoder,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        "local_eval_submission "
        f"mll={payload['mean_log_likelihood_mean']:.6f}+/-{payload['mean_log_likelihood_std']:.6f} "
        f"auc={payload['auc_roc_mean']:.6f}+/-{payload['auc_roc_std']:.6f} "
        f"out={args.out}"
    )


if __name__ == "__main__":
    main()
