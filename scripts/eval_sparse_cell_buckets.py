"""Bucket split-faithful proxy MLL by train-time subject/category count.

This is a diagnostic companion to eval_split_faithful.py. It evaluates the
two frozen base signals used by the existing proxy:

* ``priors_only``: the split-train empirical-Bayes subject/category prior.
* ``kfactor_base``: the K-factor MLP base logit after split-train subject refit.

For every sampled validation row it computes ``n_train_cell``, the number of
training rows with the same ``(subject_key, benchmark, condition)``, then
reports per-bucket mean log-likelihood.
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
    sigmoid,
    subject_category_probs_for_frame,
    tune_priors,
)
from train_kfactor_residual import (
    _eval_rows_for_split,
    _load_joined_frame,
    base_logits,
    load_item_state,
    load_runtime_subject_state,
    refit_subject_state,
    rows_to_arrays,
)

BUCKETS = (
    ("count<5", 0, 5),
    ("5<=count<20", 5, 20),
    ("count>=20", 20, None),
)


def _mean_std(values: list[float]) -> tuple[float, float]:
    finite = [float(v) for v in values if np.isfinite(float(v))]
    if not finite:
        return float("nan"), float("nan")
    if len(finite) == 1:
        return finite[0], 0.0
    return float(statistics.fmean(finite)), float(statistics.stdev(finite))


def _row_mll(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    probs = np.clip(probs.astype(np.float64), 1e-6, 1.0 - 1e-6)
    labels = labels.astype(np.float64)
    return labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)


def _cell_counts(train_df, val_df) -> np.ndarray:
    counts = (
        train_df.groupby(["subject_key", "benchmark", "condition"], sort=False)
        .size()
        .rename("n_train_cell")
    )
    joined = val_df[["subject_key", "benchmark", "condition"]].join(
        counts,
        on=["subject_key", "benchmark", "condition"],
    )
    return joined["n_train_cell"].fillna(0).to_numpy(dtype=np.int64)


def _bucket_mask(counts: np.ndarray, low: int, high: int | None) -> np.ndarray:
    mask = counts >= low
    if high is not None:
        mask &= counts < high
    return mask


def _metrics_for_probs(probs: np.ndarray, labels: np.ndarray, row_mll: np.ndarray) -> dict[str, Any]:
    labels_i = labels.astype(np.int8)
    return {
        "mean_log_likelihood": mean_log_likelihood(probs, labels_i),
        "auc_roc": auc_roc(probs, labels_i),
        "n": int(len(labels_i)),
        "p_pos": float(labels_i.mean()) if len(labels_i) else float("nan"),
        "row_mll_mean": float(row_mll.mean()) if len(row_mll) else float("nan"),
    }


def _bucket_metrics(probs: np.ndarray, labels: np.ndarray, counts: np.ndarray) -> dict[str, Any]:
    row_mll = _row_mll(probs, labels)
    out: dict[str, Any] = {}
    for name, low, high in BUCKETS:
        mask = _bucket_mask(counts, low, high)
        if not mask.any():
            out[name] = {
                "mean_log_likelihood": float("nan"),
                "auc_roc": float("nan"),
                "n": 0,
                "p_pos": float("nan"),
                "row_mll_mean": float("nan"),
            }
            continue
        out[name] = _metrics_for_probs(probs[mask], labels[mask], row_mll[mask])
    return out


def run_seed(
    seed: int,
    df,
    item_ids: list[str],
    runtime_subjects,
    item_state,
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
    counts = _cell_counts(train_df, val_df)
    labels = val_df["label"].to_numpy(dtype=np.int8)

    priors, kappas = tune_priors(train_df, val_df)
    prior_probs = subject_category_probs_for_frame(val_df, priors)

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
    base_probs = sigmoid(base_logits(val_rows, split_subjects, item_state).astype(np.float64))

    result = {
        "seed": int(seed),
        "n_held_out_items": int(len(held_out)),
        "n_train_rows": int(len(train_df)),
        "n_val_rows": int(len(val_df)),
        "kappas": kappas,
        "buckets": {
            name: int(_bucket_mask(counts, low, high).sum())
            for name, low, high in BUCKETS
        },
        "priors_only": {
            "overall": _metrics_for_probs(prior_probs, labels, _row_mll(prior_probs, labels)),
            "buckets": _bucket_metrics(prior_probs, labels, counts),
        },
        "kfactor_base": {
            "overall": _metrics_for_probs(base_probs, labels, _row_mll(base_probs, labels)),
            "buckets": _bucket_metrics(base_probs, labels, counts),
        },
    }
    print(
        f"seed={seed} "
        f"priors_mll={result['priors_only']['overall']['mean_log_likelihood']:.6f} "
        f"kfactor_mll={result['kfactor_base']['overall']['mean_log_likelihood']:.6f} "
        f"buckets={result['buckets']}",
        flush=True,
    )
    return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"seeds": [int(r["seed"]) for r in results]}
    for signal in ("priors_only", "kfactor_base"):
        signal_summary: dict[str, Any] = {}
        overall = [r[signal]["overall"]["mean_log_likelihood"] for r in results]
        m, s = _mean_std(overall)
        signal_summary["overall"] = {"mll_mean": m, "mll_std": s}
        bucket_summary: dict[str, Any] = {}
        for name, _, _ in BUCKETS:
            values = [
                r[signal]["buckets"][name]["mean_log_likelihood"]
                for r in results
                if r[signal]["buckets"][name]["n"] > 0
            ]
            counts = [int(r[signal]["buckets"][name]["n"]) for r in results]
            bm, bs = _mean_std(values)
            bucket_summary[name] = {
                "mll_mean": bm,
                "mll_std": bs,
                "n_mean": float(statistics.fmean(counts)) if counts else 0.0,
            }
        signal_summary["buckets"] = bucket_summary
        summary[signal] = signal_summary

    deltas: dict[str, Any] = {}
    overall_delta = [
        r["kfactor_base"]["overall"]["mean_log_likelihood"]
        - r["priors_only"]["overall"]["mean_log_likelihood"]
        for r in results
    ]
    dm, ds = _mean_std(overall_delta)
    deltas["overall"] = {"mll_mean": dm, "mll_std": ds}
    for name, _, _ in BUCKETS:
        values = []
        for r in results:
            if (
                r["kfactor_base"]["buckets"][name]["n"] > 0
                and r["priors_only"]["buckets"][name]["n"] > 0
            ):
                values.append(
                    r["kfactor_base"]["buckets"][name]["mean_log_likelihood"]
                    - r["priors_only"]["buckets"][name]["mean_log_likelihood"]
                )
        bm, bs = _mean_std(values)
        deltas[name] = {"mll_mean": bm, "mll_std": bs}
    summary["delta_kfactor_minus_priors"] = deltas
    return summary


def parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def main(args: argparse.Namespace) -> None:
    print("loading joined rows and frozen item state", flush=True)
    df = _load_joined_frame(args.joined)
    item_ids = [str(iid) for iid in json.loads((args.emb / "item_id_order.json").read_text())]
    item_state = load_item_state(args.stage2, args.emb)
    runtime_subjects = load_runtime_subject_state(args.stage1)
    results = [
        run_seed(
            seed,
            df,
            item_ids,
            runtime_subjects,
            item_state,
            val_frac=args.val_frac,
            max_rows=args.max_rows,
            per_category=args.per_category,
            subject_epochs=args.subject_epochs,
            subject_lr=args.subject_lr,
            batch_size=args.batch_size,
            device=args.device,
        )
        for seed in parse_seeds(args.seeds)
    ]
    payload = {"summary": summarize(results), "results": results}
    print(json.dumps(payload["summary"], indent=2), flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--joined", default="data/joined.parquet", type=Path)
    parser.add_argument("--stage1", default="data/stage1/kfactor_k4", type=Path)
    parser.add_argument("--stage2", default="data/stage2/kfactor_mpnet_mlp_v1", type=Path)
    parser.add_argument("--emb", default="data/embeddings/mpnet_v1", type=Path)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--max-rows", type=int, default=1500)
    parser.add_argument("--per-category", type=int, default=300)
    parser.add_argument("--subject-epochs", type=int, default=3)
    parser.add_argument("--subject-lr", type=float, default=5e-2)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None, type=Path)
    main(parser.parse_args())
