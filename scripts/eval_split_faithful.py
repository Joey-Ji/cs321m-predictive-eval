"""Split-faithful Lever L proxy evaluation.

This harness fixes the optimistic proxy leak in scripts/evaluate_stage2.py:
the row/item split is chosen before any split-specific fitting, validation
items are never used for the train-only subject refit or priors, and validation
rows are sampled with the same category-balanced quotas as
modal_eval_submission.py.

Historical context: the older optimistic proxy reported roughly -0.528 MLL for
the current MLP base head on all held-out item rows, and around -0.502 on the
Codabench-style sampled proxy after Lever F. This split-faithful harness should
usually be worse; treat sub-0.005 deltas as noise unless they repeat across
seeds.

Example:
    python scripts/eval_split_faithful.py \
      --joined data/joined.parquet \
      --stage1 data/stage1/kfactor_k4 \
      --stage2 data/stage2/kfactor_mpnet_mlp_v1 \
      --emb data/embeddings/mpnet_v1 \
      --residual data/stage2/kfactor_mpnet_residual_v1 \
      --seeds 0,1,2 --max-rows 1500 --per-category 300
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from lever_l_utils import (
    auc_roc,
    mean_log_likelihood,
    subject_category_probs_for_frame,
    tune_priors,
)
from train_kfactor_residual import (
    ResidualMLP,
    _eval_rows_for_split,
    _finite_metrics,
    _load_joined_frame,
    base_logits,
    evaluate_residual,
    load_item_state,
    load_runtime_subject_state,
    refit_subject_state,
    rows_to_arrays,
)


def _mean_std(values: list[float]) -> tuple[float, float]:
    finite = [float(v) for v in values if np.isfinite(float(v))]
    if not finite:
        return float("nan"), float("nan")
    if len(finite) == 1:
        return finite[0], 0.0
    return float(statistics.fmean(finite)), float(statistics.stdev(finite))


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_residual(path: Path, device: str):
    meta_path = path / "head.json"
    weights_path = path / "residual.pt"
    if not meta_path.exists() or not weights_path.exists():
        return None, None, None, None
    meta = json.loads(meta_path.read_text())
    model = ResidualMLP(
        int(meta["input_dim"]),
        hidden=int(meta["hidden"]),
        layers=int(meta.get("layers", 2)),
        dropout=float(meta.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(_torch_load(weights_path))
    model.eval()
    mean = np.asarray(meta["feature_mean"], dtype=np.float32)
    std = np.asarray(meta["feature_std"], dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return model, mean, std, meta


def run_seed(
    seed: int,
    df,
    item_ids: list[str],
    runtime_subjects,
    item_state,
    residual_model,
    residual_mean,
    residual_std,
    residual_seed: int | None,
    allow_cross_seed_residual: bool,
    val_frac: float,
    max_rows: int,
    per_category: int,
    subject_epochs: int,
    subject_lr: float,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    held_out, train_df, val_df = _eval_rows_for_split(
        df,
        item_ids,
        val_frac=val_frac,
        seed=seed,
        max_rows=max_rows,
        per_category=per_category,
    )
    priors, kappas = tune_priors(train_df, val_df)
    prior_probs = subject_category_probs_for_frame(val_df, priors)
    labels = val_df["label"].to_numpy(dtype=np.int8)
    prior_metrics = {
        "mean_log_likelihood": mean_log_likelihood(prior_probs, labels),
        "auc_roc": auc_roc(prior_probs, labels),
    }

    split_subjects = refit_subject_state(
        train_df,
        runtime_subjects,
        item_state,
        epochs=subject_epochs,
        batch_size=batch_size,
        lr=subject_lr,
        seed=seed,
        device=device,
    )
    val_rows = rows_to_arrays(val_df, split_subjects, item_state, priors)
    base_probs = 1.0 / (1.0 + np.exp(-base_logits(val_rows, split_subjects, item_state).astype(np.float64)))
    base_metrics_flat = _finite_metrics("base", base_probs, val_rows.labels.astype(np.int8))
    base_metrics = {
        "mean_log_likelihood": base_metrics_flat["base_mll"],
        "auc_roc": base_metrics_flat["base_auc"],
    }

    residual_metrics = None
    if residual_model is not None and (
        allow_cross_seed_residual or residual_seed is None or int(residual_seed) == int(seed)
    ):
        residual_metrics = evaluate_residual(
            residual_model,
            val_rows,
            split_subjects,
            item_state,
            residual_mean,
            residual_std,
            batch_size=batch_size,
            device=device,
        )
    elif residual_model is not None:
        print(
            f"seed={seed} residual skipped: artifact seed={residual_seed} "
            "would make this split leaky",
            flush=True,
        )

    result = {
        "seed": seed,
        "n_held_out_items": len(held_out),
        "n_train_rows": len(train_df),
        "n_val_rows": len(val_df),
        "kappas": kappas,
        "base": base_metrics,
        "priors_only": prior_metrics,
        "priors_residual": residual_metrics,
    }
    print(
        f"seed={seed} "
        f"base_mll={base_metrics['mean_log_likelihood']:.6f} base_auc={base_metrics['auc_roc']:.6f} "
        f"priors_mll={prior_metrics['mean_log_likelihood']:.6f} priors_auc={prior_metrics['auc_roc']:.6f}"
        + (
            f" residual_mll={residual_metrics['mean_log_likelihood']:.6f} "
            f"residual_auc={residual_metrics['auc_roc']:.6f}"
            if residual_metrics is not None
            else " residual=missing"
        ),
        flush=True,
    )
    return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"seeds": [int(r["seed"]) for r in results]}
    for label in ("base", "priors_only", "priors_residual"):
        present = [r[label] for r in results if r.get(label) is not None]
        if not present:
            continue
        mll_mean, mll_std = _mean_std([r["mean_log_likelihood"] for r in present])
        auc_mean, auc_std = _mean_std([r["auc_roc"] for r in present])
        summary[label] = {
            "mll_mean": mll_mean,
            "mll_std": mll_std,
            "auc_mean": auc_mean,
            "auc_std": auc_std,
        }
    return summary


def parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def main(
    joined: Path,
    stage1: Path,
    stage2: Path,
    emb: Path,
    residual: Path,
    seeds: list[int],
    val_frac: float,
    max_rows: int,
    per_category: int,
    subject_epochs: int,
    subject_lr: float,
    batch_size: int,
    device: str,
    out: Path | None,
    allow_cross_seed_residual: bool,
) -> None:
    print("loading data and frozen item state", flush=True)
    df = _load_joined_frame(joined)
    item_ids = [str(iid) for iid in json.loads((emb / "item_id_order.json").read_text())]
    item_state = load_item_state(stage2, emb, joined_path=joined)
    runtime_subjects = load_runtime_subject_state(stage1)
    residual_model, residual_mean, residual_std, residual_meta = load_residual(residual, device)
    residual_seed = None if residual_meta is None else int(residual_meta.get("seed", -1))
    if residual_model is None:
        print(f"residual artifacts not found under {residual}; reporting base and priors only", flush=True)
    else:
        print(
            f"loaded residual input_dim={residual_meta['input_dim']} seed={residual_seed} from {residual}",
            flush=True,
        )

    results = [
        run_seed(
            seed,
            df,
            item_ids,
            runtime_subjects,
            item_state,
            residual_model,
            residual_mean,
            residual_std,
            residual_seed,
            allow_cross_seed_residual,
            val_frac=val_frac,
            max_rows=max_rows,
            per_category=per_category,
            subject_epochs=subject_epochs,
            subject_lr=subject_lr,
            batch_size=batch_size,
            device=device,
        )
        for seed in seeds
    ]
    summary = summarize(results)
    payload = {"summary": summary, "results": results}
    print(json.dumps(summary, indent=2), flush=True)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--joined", default="data/joined.parquet", type=Path)
    parser.add_argument("--stage1", default="data/stage1/kfactor_k4", type=Path)
    parser.add_argument("--stage2", default="data/stage2/kfactor_mpnet_mlp_v1", type=Path)
    parser.add_argument("--emb", default="data/embeddings/mpnet_v1", type=Path)
    parser.add_argument("--residual", default="data/stage2/kfactor_mpnet_residual_v1", type=Path)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--max-rows", type=int, default=1500)
    parser.add_argument("--per-category", type=int, default=300)
    parser.add_argument("--subject-epochs", type=int, default=3)
    parser.add_argument("--subject-lr", type=float, default=5e-2)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None, type=Path)
    parser.add_argument(
        "--allow-cross-seed-residual",
        action="store_true",
        help="Evaluate one residual artifact on other split seeds. This is leaky for split-faithful reporting.",
    )
    args = parser.parse_args()
    main(
        args.joined,
        args.stage1,
        args.stage2,
        args.emb,
        args.residual,
        parse_seeds(args.seeds),
        args.val_frac,
        args.max_rows,
        args.per_category,
        args.subject_epochs,
        args.subject_lr,
        args.batch_size,
        args.device,
        args.out,
        args.allow_cross_seed_residual,
    )
