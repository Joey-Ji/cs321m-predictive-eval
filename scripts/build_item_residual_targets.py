"""Build OOF scalar item-difficulty residual targets.

Each item target is a shrunk Newton update against an out-of-fold locked prior:

    delta_i = sum(y - p_prior) / (sum(p_prior * (1 - p_prior)) + lambda_item)

The fold split is item-disjoint. The prior for every validation row is fit on
training rows whose item_id is in a different fold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from item_residual_utils import (  # noqa: E402
    fit_locked_priors,
    item_fold_map,
    load_joined_canonical,
    locked_kappas,
    prior_probs_for_frame,
    sha256_file,
)


def _finite_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Item Residual Phase 0",
        "",
        f"- joined rows: {summary['n_rows']}",
        f"- items: {summary['n_items']}",
        f"- folds: {summary['n_folds']} seed={summary['seed']}",
        f"- lambda_item: {summary['lambda_item']}",
        f"- kappas: `{json.dumps(summary['kappas'], sort_keys=True)}`",
        "",
        "## Target Distribution",
        "",
        f"- mean(delta): {summary['delta_stats']['mean']:.6f}",
        f"- std(delta): {summary['delta_stats']['std']:.6f}",
        f"- min/max(delta): {summary['delta_stats']['min']:.6f} / {summary['delta_stats']['max']:.6f}",
        f"- |delta| > 0.25: {summary['tail_counts']['abs_gt_0_25']}",
        f"- clipped at +/-0.50: {summary['tail_counts']['abs_ge_0_50']}",
        "",
        "## Decision",
        "",
        f"- proceed_to_phase1: `{summary['proceed_to_phase1']}`",
        f"- reason: {summary['decision_reason']}",
        "",
        "## Leakage Spot Check",
        "",
    ]
    for check in summary["leakage_spot_check"]:
        lines.append(
            f"- item `{check['item_id']}` fold={check['fold']} "
            f"train_contains_item={check['train_contains_item']}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main(
    joined: Path,
    runtime_priors: Path,
    out_dir: Path,
    n_folds: int,
    seed: int,
    lambda_item: float,
    target_clip: float,
    report_json: Path,
    report_md: Path,
) -> None:
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading canonical joined rows from {joined}", flush=True)
    df, runtime, _ = load_joined_canonical(joined, runtime_priors)
    kappas = locked_kappas(runtime)
    fold_by_item = item_fold_map(df["item_id"].astype(str).unique().tolist(), n_folds=n_folds, seed=seed)
    df["item_residual_fold"] = df["item_id"].map(fold_by_item).astype("int8")
    print(
        f"loaded rows={len(df):,} items={df['item_id'].nunique():,} "
        f"subjects={df['subject_key'].nunique():,} folds={n_folds}",
        flush=True,
    )

    item_parts: list[pd.DataFrame] = []
    row_parts: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, Any]] = []
    for fold in range(n_folds):
        train_df = df[df["item_residual_fold"] != fold]
        val_df = df[df["item_residual_fold"] == fold].copy()
        if len(train_df) == 0 or len(val_df) == 0:
            raise ValueError(f"fold {fold} produced empty train or validation rows")
        print(
            f"fold={fold} train_rows={len(train_df):,} val_rows={len(val_df):,} "
            f"val_items={val_df['item_id'].nunique():,}",
            flush=True,
        )
        priors = fit_locked_priors(train_df, kappas)
        prior_p = prior_probs_for_frame(val_df, priors)
        h = prior_p * (1.0 - prior_p)
        residual_frame = val_df[
            ["item_id", "item_content", "benchmark_key", "condition_key", "label", "row_index"]
        ].copy()
        residual_frame["prior_p"] = prior_p.astype(np.float32)
        residual_frame["grad"] = residual_frame["label"].to_numpy(dtype=np.float64) - prior_p
        residual_frame["hess"] = h.astype(np.float64)

        grouped = residual_frame.groupby("item_id", observed=True).agg(
            item_content=("item_content", "first"),
            benchmark=("benchmark_key", "first"),
            condition=("condition_key", "first"),
            n_rows=("label", "size"),
            label_sum=("label", "sum"),
            grad_sum=("grad", "sum"),
            hess_sum=("hess", "sum"),
            benchmark_nunique=("benchmark_key", "nunique"),
            condition_nunique=("condition_key", "nunique"),
        )
        grouped = grouped.reset_index()
        grouped["fold"] = fold
        grouped["delta_raw"] = grouped["grad_sum"] / (grouped["hess_sum"] + float(lambda_item))
        grouped["delta"] = grouped["delta_raw"].clip(-float(target_clip), float(target_clip))
        grouped["weight"] = grouped["hess_sum"].astype(np.float32)
        item_parts.append(grouped)
        row_parts.append(
            residual_frame[["row_index", "item_id", "label", "prior_p"]].assign(fold=fold)
        )
        fold_summaries.append(
            {
                "fold": fold,
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "val_items": int(grouped["item_id"].nunique()),
                "delta_mean": float(grouped["delta"].mean()),
                "delta_std": float(grouped["delta"].std(ddof=0)),
            }
        )

    targets = pd.concat(item_parts, ignore_index=True)
    row_priors = pd.concat(row_parts, ignore_index=True)
    target_path = out_dir / "oof_item_targets.parquet"
    row_path = out_dir / "oof_row_priors.parquet"
    targets.to_parquet(target_path, index=False)
    row_priors.to_parquet(row_path, index=False)

    delta = targets["delta"].to_numpy(dtype=np.float64)
    delta_raw = targets["delta_raw"].to_numpy(dtype=np.float64)
    ambiguous = targets[(targets["benchmark_nunique"] > 1) | (targets["condition_nunique"] > 1)]
    leakage_spot_check = []
    for item_id in targets["item_id"].astype(str).sort_values().head(8).tolist():
        fold = int(fold_by_item[item_id])
        train_contains = bool(df[(df["item_residual_fold"] != fold) & (df["item_id"] == item_id)].shape[0])
        leakage_spot_check.append(
            {"item_id": item_id, "fold": fold, "train_contains_item": train_contains}
        )

    std = float(np.std(delta, ddof=0))
    proceed = bool(std >= 0.05)
    reason = (
        "target std is large enough to attempt a content model"
        if proceed
        else "target std < 0.05; residual target has no usable spread"
    )
    summary = {
        "phase": "phase0_oof_targets",
        "joined": str(joined),
        "joined_sha256": sha256_file(joined),
        "runtime_priors": str(runtime_priors),
        "runtime_priors_sha256": sha256_file(runtime_priors),
        "target_path": str(target_path),
        "row_prior_path": str(row_path),
        "n_rows": int(len(df)),
        "n_items": int(targets["item_id"].nunique()),
        "n_folds": int(n_folds),
        "seed": int(seed),
        "lambda_item": float(lambda_item),
        "target_clip": float(target_clip),
        "kappas": kappas,
        "folds": fold_summaries,
        "delta_stats": _finite_stats(delta),
        "delta_raw_stats": _finite_stats(delta_raw),
        "tail_counts": {
            "abs_gt_0_25": int((np.abs(delta) > 0.25).sum()),
            "abs_ge_0_50": int((np.abs(delta) >= float(target_clip)).sum()),
            "raw_abs_gt_0_50": int((np.abs(delta_raw) > float(target_clip)).sum()),
        },
        "ambiguous_item_feature_rows": int(len(ambiguous)),
        "leakage_spot_check": leakage_spot_check,
        "proceed_to_phase1": proceed,
        "decision_reason": reason,
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_markdown(summary, report_md)
    print(json.dumps(summary["delta_stats"], indent=2), flush=True)
    print(f"wrote {target_path}", flush=True)
    print(f"wrote {row_path}", flush=True)
    print(f"phase0 proceed_to_phase1={proceed}: {reason}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--joined", default="data/joined.parquet", type=Path)
    parser.add_argument("--runtime-priors", default="data/stage2/priors_v1_locked/runtime_priors.json", type=Path)
    parser.add_argument("--out-dir", default="data/stage2/item_residual_v1", type=Path)
    parser.add_argument("--folds", default=5, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--lambda-item", default=10.0, type=float)
    parser.add_argument("--target-clip", default=0.50, type=float)
    parser.add_argument("--report-json", default="reports/item_residual_phase0.json", type=Path)
    parser.add_argument("--report-md", default="reports/item_residual_phase0.md", type=Path)
    args = parser.parse_args()
    main(
        args.joined,
        args.runtime_priors,
        args.out_dir,
        args.folds,
        args.seed,
        args.lambda_item,
        args.target_clip,
        args.report_json,
        args.report_md,
    )

