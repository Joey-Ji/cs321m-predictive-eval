"""Split-faithful JE-IRT proxy against the PRIOR_ONLY baseline."""

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

from je_irt_utils import load_je_irt_artifacts, predict_je_irt_probs
from lever_l_utils import auc_roc, mean_log_likelihood, split_faithful_eval_rows, subject_category_probs_for_frame, tune_priors
from train_kfactor_residual import _load_joined_frame


def parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def _mean_std(values: list[float]) -> tuple[float, float]:
    finite = [float(v) for v in values if np.isfinite(float(v))]
    if not finite:
        return float("nan"), float("nan")
    if len(finite) == 1:
        return finite[0], 0.0
    return float(statistics.fmean(finite)), float(statistics.stdev(finite))


def _run_seed(
    seed: int,
    df,
    item_ids: list[str],
    item_embeddings: np.ndarray,
    item_to_row: dict[str, int],
    je_irt_dir: Path,
    val_frac: float,
    max_rows: int,
    per_category: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    model, subject_to_id, _ = load_je_irt_artifacts(je_irt_dir, device=device)
    held_out, train_df, val_df = split_faithful_eval_rows(
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
    je_probs = predict_je_irt_probs(
        model,
        subject_to_id,
        item_embeddings,
        item_to_row,
        val_df,
        prior_probs,
        batch_size=batch_size,
        device=device,
    )
    prior_mll = mean_log_likelihood(prior_probs, labels)
    je_mll = mean_log_likelihood(je_probs, labels)
    return {
        "seed": int(seed),
        "n_held_out_items": int(len(held_out)),
        "n_train_rows": int(len(train_df)),
        "n_val_rows": int(len(val_df)),
        "kappas": kappas,
        "prior_only_mll": float(prior_mll),
        "je_irt_mll": float(je_mll),
        "delta": float(je_mll - prior_mll),
        "prior_only_auc": float(auc_roc(prior_probs, labels)),
        "auc": float(auc_roc(je_probs, labels)),
        "fallback_rows": int(
            (
                val_df["subject_id"].map(subject_to_id).isna()
                | val_df["item_id"].map(item_to_row).isna()
            ).sum()
        ),
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    prior_mll, prior_std = _mean_std([r["prior_only_mll"] for r in results])
    je_mll, je_std = _mean_std([r["je_irt_mll"] for r in results])
    delta, delta_std = _mean_std([r["delta"] for r in results])
    auc, auc_std = _mean_std([r["auc"] for r in results])
    return {
        "prior_only_mll": prior_mll,
        "prior_only_mll_std": prior_std,
        "je_irt_mll": je_mll,
        "je_irt_mll_std": je_std,
        "delta": delta,
        "delta_std": delta_std,
        "auc": auc,
        "auc_std": auc_std,
        "seeds": [int(r["seed"]) for r in results],
        "n_seeds": len(results),
        "seed_results": results,
    }


def main(
    joined: Path,
    emb: Path,
    je_irt_dir: Path,
    seeds: list[int],
    val_frac: float,
    max_rows: int,
    per_category: int,
    batch_size: int,
    device: str,
) -> None:
    print("loading data and JE-IRT artifacts", file=sys.stderr, flush=True)
    df = _load_joined_frame(joined)
    item_ids = [str(iid) for iid in json.loads((emb / "item_id_order.json").read_text())]
    item_to_row = {item_id: idx for idx, item_id in enumerate(item_ids)}
    item_embeddings = np.load(emb / "item_embeddings.npy").astype(np.float32)
    results = [
        _run_seed(
            seed,
            df,
            item_ids,
            item_embeddings,
            item_to_row,
            je_irt_dir,
            val_frac,
            max_rows,
            per_category,
            batch_size,
            device,
        )
        for seed in seeds
    ]
    summary = summarize(results)
    (je_irt_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--joined", default="data/joined.parquet", type=Path)
    parser.add_argument("--emb", default="data/embeddings/mpnet_v1", type=Path)
    parser.add_argument("--je-irt", required=True, type=Path)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--max-rows", type=int, default=1500)
    parser.add_argument("--per-category", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(
        args.joined,
        args.emb,
        args.je_irt,
        parse_seeds(args.seeds),
        args.val_frac,
        args.max_rows,
        args.per_category,
        args.batch_size,
        args.device,
    )
