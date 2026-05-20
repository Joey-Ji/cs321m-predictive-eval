"""Train a scalar item-difficulty residual from OOF item targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from item_residual_utils import (  # noqa: E402
    bce_mll_from_prior_delta,
    fit_weighted_ridge,
    pearson_corr,
    predict_weighted_ridge,
    weighted_rmse,
)


WEIGHT_GRID_DEFAULT = (0.0, 0.1, 0.2, 0.35, 0.5)
ALPHA_GRID_DEFAULT = (0.1, 1.0, 10.0, 100.0, 1000.0)


def _parse_float_csv(value: str) -> list[float]:
    out = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not out:
        raise ValueError("expected at least one float")
    return out


def _load_side_meta(path: Path) -> dict[str, Any]:
    meta = json.loads(path.read_text())
    for key in ("benchmark", "condition", "benchmark_dim", "condition_dim", "side_feature_dim"):
        if key not in meta:
            raise KeyError(f"{path} missing {key}")
    return meta


def load_item_features(targets, emb_dir: Path, side_meta_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Return [item embedding, canonical benchmark one-hot, condition one-hot]."""

    item_order = [str(v) for v in json.loads((emb_dir / "item_id_order.json").read_text())]
    item_to_row = {item_id: idx for idx, item_id in enumerate(item_order)}
    embeddings = np.load(emb_dir / "item_embeddings.npy").astype(np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(item_order):
        raise ValueError(f"bad embedding matrix shape {embeddings.shape} for {len(item_order)} item ids")

    item_indices = targets["item_id"].astype(str).map(item_to_row)
    missing = targets[item_indices.isna()]["item_id"].astype(str).head(10).tolist()
    if missing:
        raise KeyError(f"targets contain item ids missing from embeddings, e.g. {missing}")
    emb = embeddings[item_indices.to_numpy(dtype=np.int64)]

    side_meta = _load_side_meta(side_meta_path)
    b_dim = int(side_meta["benchmark_dim"])
    c_dim = int(side_meta["condition_dim"])
    side = np.zeros((len(targets), b_dim + c_dim), dtype=np.float32)
    row_ids = np.arange(len(targets), dtype=np.int64)
    b_idx = targets["benchmark"].astype(str).map(side_meta["benchmark"])
    b_keep = b_idx.notna().to_numpy()
    side[row_ids[b_keep], b_idx[b_keep].to_numpy(dtype=np.int64)] = 1.0
    c_idx = targets["condition"].astype(str).map(side_meta["condition"])
    c_keep = c_idx.notna().to_numpy()
    side[row_ids[c_keep], b_dim + c_idx[c_keep].to_numpy(dtype=np.int64)] = 1.0

    features = np.concatenate([emb, side], axis=1).astype(np.float32)
    feature_meta = {
        "input_dim": int(features.shape[1]),
        "embedding_dim": int(embeddings.shape[1]),
        "side_feature_dim": int(side.shape[1]),
        "benchmark_dim": b_dim,
        "condition_dim": c_dim,
        "benchmark_unseen_count": int((~b_keep).sum()),
        "condition_unseen_count": int((~c_keep).sum()),
        "feature_order": [
            f"item_embedding_0..{embeddings.shape[1] - 1}",
            f"benchmark_one_hot_0..{b_dim - 1}",
            f"condition_one_hot_0..{c_dim - 1}",
        ],
    }
    return features, feature_meta


def _evaluate_holdout(
    targets,
    row_priors,
    pred_delta: np.ndarray,
    runtime_clip: float,
    weights: list[float],
) -> tuple[dict[str, float], list[dict[str, float]]]:
    holdout = targets[["item_id", "delta", "weight"]].copy()
    holdout["pred_delta"] = pred_delta.astype(np.float32)
    clipped = np.clip(pred_delta.astype(np.float64), -float(runtime_clip), float(runtime_clip))
    holdout["pred_delta_runtime"] = clipped.astype(np.float32)
    item_metrics = {
        "weighted_rmse": weighted_rmse(pred_delta, holdout["delta"].to_numpy(dtype=np.float64), holdout["weight"].to_numpy(dtype=np.float64)),
        "pearson": pearson_corr(pred_delta, holdout["delta"].to_numpy(dtype=np.float64)),
        "pred_delta_mean": float(np.mean(pred_delta)),
        "pred_delta_std": float(np.std(pred_delta, ddof=0)),
        "pred_delta_runtime_abs_gt_clip_count": int((np.abs(pred_delta) > float(runtime_clip)).sum()),
    }

    rows = row_priors.merge(holdout[["item_id", "pred_delta_runtime"]], on="item_id", how="inner")
    labels = rows["label"].to_numpy(dtype=np.int8)
    prior_p = rows["prior_p"].to_numpy(dtype=np.float64)
    delta = rows["pred_delta_runtime"].to_numpy(dtype=np.float64)
    prior_mll = bce_mll_from_prior_delta(labels, prior_p, delta, 0.0)
    table = []
    for weight_w in weights:
        mll = bce_mll_from_prior_delta(labels, prior_p, delta, weight_w)
        table.append(
            {
                "weight_w": float(weight_w),
                "row_mll": float(mll),
                "row_mll_gain": float(mll - prior_mll),
                "n_rows": int(len(rows)),
            }
        )
    return item_metrics, table


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Item Residual Phase 1",
        "",
        f"- model: {summary['model_type']}",
        f"- holdout_fold: {summary['holdout_fold']}",
        f"- min_gain: {summary['min_gain']}",
        f"- proceed_to_full_train: `{summary['proceed_to_full_train']}`",
        f"- reason: {summary['decision_reason']}",
        "",
        "## Holdout Table",
        "",
        "| alpha | w | row MLL | gain | RMSE | corr |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["holdout_table"]:
        lines.append(
            f"| {row['alpha']:.6g} | {row['weight_w']:.2f} | {row['row_mll']:.6f} | "
            f"{row['row_mll_gain']:.6f} | {row['weighted_rmse']:.6f} | {row['pearson']:.6f} |"
        )
    if summary.get("artifact_dir"):
        lines.extend(
            [
                "",
                "## Artifact",
                "",
                f"- artifact_dir: `{summary['artifact_dir']}`",
                f"- selected_alpha: {summary['selected_alpha']}",
                f"- selected_weight_w: {summary['selected_weight_w']}",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _save_artifact(
    out_dir: Path,
    model: dict[str, Any],
    meta: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "coef": torch.tensor(np.asarray(model["coef"], dtype=np.float32)),
            "intercept": torch.tensor(float(model["intercept"]), dtype=torch.float32),
        },
        out_dir / "item_residual_model.pt",
    )
    meta = {
        **meta,
        "feature_mean": np.asarray(model["feature_mean"], dtype=np.float32).astype(float).tolist(),
        "feature_std": np.asarray(model["feature_std"], dtype=np.float32).astype(float).tolist(),
    }
    (out_dir / "item_residual_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def main(
    targets_path: Path,
    row_priors_path: Path,
    emb_dir: Path,
    side_meta_path: Path,
    out_dir: Path,
    holdout_fold: int,
    alphas: list[float],
    weights: list[float],
    min_gain: float,
    runtime_clip: float,
    report_json: Path,
    report_md: Path,
    force_full: bool,
) -> None:
    import pandas as pd

    targets = pd.read_parquet(targets_path)
    row_priors = pd.read_parquet(row_priors_path)
    features, feature_meta = load_item_features(targets, emb_dir, side_meta_path)
    y = targets["delta"].to_numpy(dtype=np.float64)
    sample_weight = targets["weight"].to_numpy(dtype=np.float64)
    folds = targets["fold"].to_numpy(dtype=np.int64)
    train_mask = folds != int(holdout_fold)
    holdout_mask = folds == int(holdout_fold)
    if int(holdout_mask.sum()) == 0 or int(train_mask.sum()) == 0:
        raise ValueError(f"holdout_fold={holdout_fold} produced empty train or holdout split")

    holdout_rows = row_priors[row_priors["fold"].astype("int64") == int(holdout_fold)].copy()
    holdout_table: list[dict[str, float]] = []
    model_by_alpha: dict[float, dict[str, Any]] = {}
    for alpha in alphas:
        model = fit_weighted_ridge(features[train_mask], y[train_mask], sample_weight[train_mask], alpha=alpha)
        model_by_alpha[float(alpha)] = model
        pred = predict_weighted_ridge(features[holdout_mask], model)
        item_metrics, bce_table = _evaluate_holdout(
            targets[holdout_mask].reset_index(drop=True),
            holdout_rows,
            pred,
            runtime_clip=runtime_clip,
            weights=weights,
        )
        for row in bce_table:
            holdout_table.append(
                {
                    "alpha": float(alpha),
                    **row,
                    **item_metrics,
                }
            )

    best = max(holdout_table, key=lambda row: (row["row_mll_gain"], row["row_mll"]))
    proceed = bool(best["row_mll_gain"] >= float(min_gain))
    reason = (
        f"best holdout gain {best['row_mll_gain']:.6f} >= {min_gain:.6f}"
        if proceed
        else f"best holdout gain {best['row_mll_gain']:.6f} < {min_gain:.6f}; residual model has no usable signal"
    )

    artifact_dir = None
    full_meta = None
    if proceed or force_full:
        selected_alpha = float(best["alpha"])
        full_model = fit_weighted_ridge(features, y, sample_weight, alpha=selected_alpha)
        full_pred = predict_weighted_ridge(features, full_model)
        full_meta = {
            "model_type": "weighted_ridge_item_delta_v1",
            "ridge_alpha": selected_alpha,
            "weight_w": float(best["weight_w"]),
            "runtime_delta_clip": float(runtime_clip),
            "target_delta_clip": 0.50,
            "input_dim": int(feature_meta["input_dim"]),
            "embedding_dim": int(feature_meta["embedding_dim"]),
            "side_feature_dim": int(feature_meta["side_feature_dim"]),
            "benchmark_dim": int(feature_meta["benchmark_dim"]),
            "condition_dim": int(feature_meta["condition_dim"]),
            "feature_order": feature_meta["feature_order"],
            "source_targets": str(targets_path),
            "source_row_priors": str(row_priors_path),
            "holdout_fold": int(holdout_fold),
            "holdout_best": best,
            "full_train_metrics": {
                "weighted_rmse": weighted_rmse(full_pred, y, sample_weight),
                "pearson": pearson_corr(full_pred, y),
                "pred_delta_mean": float(np.mean(full_pred)),
                "pred_delta_std": float(np.std(full_pred, ddof=0)),
            },
        }
        _save_artifact(out_dir, full_model, full_meta)
        artifact_dir = str(out_dir)

    summary = {
        "phase": "phase1_ridge_training",
        "model_type": "weighted_ridge_item_delta_v1",
        "targets_path": str(targets_path),
        "row_priors_path": str(row_priors_path),
        "n_items": int(len(targets)),
        "n_train_items": int(train_mask.sum()),
        "n_holdout_items": int(holdout_mask.sum()),
        "holdout_fold": int(holdout_fold),
        "alphas": [float(v) for v in alphas],
        "weights": [float(v) for v in weights],
        "runtime_clip": float(runtime_clip),
        "min_gain": float(min_gain),
        "feature_meta": feature_meta,
        "holdout_table": holdout_table,
        "best": best,
        "proceed_to_full_train": proceed,
        "force_full": bool(force_full),
        "decision_reason": reason,
        "artifact_dir": artifact_dir,
        "selected_alpha": None if full_meta is None else full_meta["ridge_alpha"],
        "selected_weight_w": None if full_meta is None else full_meta["weight_w"],
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_markdown(summary, report_md)
    print(json.dumps({"best": best, "proceed_to_full_train": proceed, "reason": reason}, indent=2), flush=True)
    if artifact_dir:
        print(f"wrote runtime artifact to {artifact_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="data/stage2/item_residual_v1/oof_item_targets.parquet", type=Path)
    parser.add_argument("--row-priors", default="data/stage2/item_residual_v1/oof_row_priors.parquet", type=Path)
    parser.add_argument("--emb", default="data/embeddings/mpnet_v1", type=Path)
    parser.add_argument("--side-meta", default="data/stage2/kfactor_mpnet_mlp_v1/side_feature_meta.json", type=Path)
    parser.add_argument("--out-dir", default="data/stage2/item_residual_v1", type=Path)
    parser.add_argument("--holdout-fold", default=0, type=int)
    parser.add_argument("--alphas", default=",".join(str(v) for v in ALPHA_GRID_DEFAULT))
    parser.add_argument("--weights", default=",".join(str(v) for v in WEIGHT_GRID_DEFAULT))
    parser.add_argument("--min-gain", default=0.003, type=float)
    parser.add_argument("--runtime-clip", default=0.25, type=float)
    parser.add_argument("--report-json", default="reports/item_residual_phase1.json", type=Path)
    parser.add_argument("--report-md", default="reports/item_residual_phase1.md", type=Path)
    parser.add_argument(
        "--force-full",
        action="store_true",
        help="Write a full-data artifact even if the Phase 1 holdout stop gate fails.",
    )
    args = parser.parse_args()
    main(
        args.targets,
        args.row_priors,
        args.emb,
        args.side_meta,
        args.out_dir,
        args.holdout_fold,
        _parse_float_csv(args.alphas),
        _parse_float_csv(args.weights),
        args.min_gain,
        args.runtime_clip,
        args.report_json,
        args.report_md,
        args.force_full,
    )

