"""Fast local proxy for the scalar item residual using cached item embeddings.

This is a fallback for the Modal package evaluator when serial predict-time
encoding is too slow. It uses the same corrected item-cold sampling logic and
the same runtime composition, but reads data/embeddings/mpnet_v1 for held-out
training items instead of calling SentenceTransformer one item at a time.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from item_residual_utils import (  # noqa: E402
    LockedPriorTables,
    RuntimeCanonicalizer,
    bce_mll_from_prior_delta,
    load_joined_canonical,
    locked_kappas,
    logit_prob,
    predict_weighted_ridge,
    sigmoid,
)
from modal_eval_submission import _group_rows_by_category, _load_held_out_item_ids, _sample_groups  # noqa: E402
from src.validation import auc_roc, mean_log_likelihood  # noqa: E402
from train_item_residual import load_item_features  # noqa: E402

CLIP_LO = 0.02
CLIP_HI = 0.98


def _parse_seeds(value: str) -> list[int]:
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


def _runtime_priors_to_tables(runtime: dict[str, Any]) -> LockedPriorTables:
    return LockedPriorTables(
        global_p=float(runtime["global"]),
        benchmark={str(k): float(v) for k, v in runtime.get("benchmark", {}).items()},
        benchmark_condition={str(k): float(v) for k, v in runtime.get("benchmark_condition", {}).items()},
        subject={str(k): float(v) for k, v in runtime.get("subject", {}).items()},
        subject_benchmark={str(k): float(v) for k, v in runtime.get("subject_benchmark", {}).items()},
        subject_category={str(k): float(v) for k, v in runtime.get("subject_category", {}).items()},
        kappas=locked_kappas(runtime),
        key_sep=str(runtime.get("key_sep", "\x1f")),
    )


def _load_item_residual_model(artifact_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    meta = json.loads((artifact_dir / "item_residual_meta.json").read_text())
    try:
        state = torch.load(artifact_dir / "item_residual_model.pt", map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(artifact_dir / "item_residual_model.pt", map_location="cpu")
    model = {
        "coef": state["coef"].detach().cpu().numpy().astype(np.float32),
        "intercept": float(state["intercept"].detach().cpu()),
        "feature_mean": np.asarray(meta["feature_mean"], dtype=np.float32),
        "feature_std": np.asarray(meta["feature_std"], dtype=np.float32),
    }
    return model, meta


def _score(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "mean_log_likelihood": float(mean_log_likelihood(probs.tolist(), labels.astype(int).tolist())),
        "auc_roc": float(auc_roc(probs.tolist(), labels.astype(int).tolist())),
        "n": float(len(labels)),
        "p_pos": float(labels.mean()),
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    summary = payload["summary"]
    lines = [
        "# Item Residual Phase 3 Cached Proxy",
        "",
        f"- baseline MLL: {summary['baseline']['mll_mean']:.6f} +/- {summary['baseline']['mll_std']:.6f}",
        f"- candidate MLL: {summary['candidate']['mll_mean']:.6f} +/- {summary['candidate']['mll_std']:.6f}",
        f"- gain: {summary['gain']['mll_mean']:.6f} +/- {summary['gain']['mll_std']:.6f}",
        f"- sign consistency: {summary['sign_consistency']['improved_seeds']}/{summary['sign_consistency']['n_seeds']}",
        f"- decision: `{summary['decision']['pass']}` - {summary['decision']['reason']}",
        "",
        "| seed | baseline MLL | candidate MLL | gain |",
        "|---:|---:|---:|---:|",
    ]
    for row in payload["results"]:
        lines.append(
            f"| {row['seed']} | {row['baseline']['mean_log_likelihood']:.6f} | "
            f"{row['candidate']['mean_log_likelihood']:.6f} | {row['gain_mll']:.6f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main(
    joined: Path,
    runtime_priors: Path,
    artifact_dir: Path,
    emb_dir: Path,
    side_meta: Path,
    stage1: Path,
    stage2: Path,
    seeds: list[int],
    max_rows: int,
    per_category: int,
    val_frac: float,
    split_seed: int,
    min_gain: float,
    out: Path,
    report_md: Path,
) -> None:
    import pandas as pd

    print("loading canonical rows and locked priors", flush=True)
    df, runtime, _ = load_joined_canonical(joined, runtime_priors)
    canonicalizer = RuntimeCanonicalizer.from_runtime_priors(runtime)
    priors = _runtime_priors_to_tables(runtime)
    held_out = _load_held_out_item_ids(stage1, stage2, Path("__unused_submission_dir__"), val_frac, split_seed)
    val_df = df[df["item_id"].isin(held_out)].copy()
    if val_df.empty:
        raise ValueError("no held-out rows for cached proxy")
    rows = val_df[
        [
            "row_index",
            "item_id",
            "subject_content",
            "item_content",
            "benchmark",
            "condition",
            "subject_key",
            "benchmark_key",
            "condition_key",
            "label",
        ]
    ].to_dict(orient="records")
    groups = _group_rows_by_category(rows)
    residual_model, residual_meta = _load_item_residual_model(artifact_dir)
    weight_w = float(residual_meta["weight_w"])
    runtime_clip = float(residual_meta.get("runtime_delta_clip", 0.25))

    results = []
    for seed in seeds:
        sampled_groups = _sample_groups(groups, random.Random(seed), max_rows=max_rows, max_per_category=per_category)
        sampled_rows = [row for category in sorted(sampled_groups) for row in sampled_groups[category]]
        sample_df = pd.DataFrame(sampled_rows)
        # Re-apply canonicalization to mirror predict-time behavior on raw fields.
        sample_df["benchmark"] = sample_df["benchmark"].map(canonicalizer.normalize_benchmark)
        sample_df["condition"] = sample_df["condition"].map(canonicalizer.normalize_condition)
        sample_df["benchmark_key"] = sample_df["benchmark"]
        sample_df["condition_key"] = sample_df["condition"]

        prior_p = np.empty(len(sample_df), dtype=np.float64)
        from item_residual_utils import prior_probs_for_frame

        prior_p[:] = prior_probs_for_frame(sample_df, priors)
        feature_df = sample_df[["item_id", "item_content", "benchmark", "condition"]].copy()
        features, _ = load_item_features(feature_df, emb_dir, side_meta)
        delta = np.clip(predict_weighted_ridge(features, residual_model), -runtime_clip, runtime_clip)
        labels = sample_df["label"].to_numpy(dtype=np.int8)
        baseline_probs = np.clip(prior_p.astype(np.float64), CLIP_LO, CLIP_HI)
        candidate_probs = np.clip(
            np.asarray(sigmoid(logit_prob(prior_p) + weight_w * delta), dtype=np.float64),
            CLIP_LO,
            CLIP_HI,
        )
        baseline = _score(baseline_probs, labels)
        candidate = _score(candidate_probs, labels)
        result = {
            "seed": int(seed),
            "baseline": baseline,
            "candidate": candidate,
            "gain_mll": float(candidate["mean_log_likelihood"] - baseline["mean_log_likelihood"]),
            "n_unique_items": int(sample_df["item_id"].nunique()),
        }
        print(
            f"seed={seed} baseline_mll={baseline['mean_log_likelihood']:.6f} "
            f"candidate_mll={candidate['mean_log_likelihood']:.6f} "
            f"gain={result['gain_mll']:.6f}",
            flush=True,
        )
        results.append(result)

    baseline_mean, baseline_std = _mean_std([r["baseline"]["mean_log_likelihood"] for r in results])
    candidate_mean, candidate_std = _mean_std([r["candidate"]["mean_log_likelihood"] for r in results])
    gain_mean, gain_std = _mean_std([r["gain_mll"] for r in results])
    improved = sum(1 for r in results if r["gain_mll"] > 0.0)
    passed = bool(gain_mean >= float(min_gain) and improved >= 7)
    reason = (
        f"gain {gain_mean:.6f} >= {min_gain:.6f} and {improved}/{len(results)} seeds improved"
        if passed
        else f"gain {gain_mean:.6f} or sign consistency {improved}/{len(results)} did not meet gate"
    )
    payload = {
        "summary": {
            "baseline": {"mll_mean": baseline_mean, "mll_std": baseline_std},
            "candidate": {"mll_mean": candidate_mean, "mll_std": candidate_std},
            "gain": {"mll_mean": gain_mean, "mll_std": gain_std},
            "sign_consistency": {"improved_seeds": improved, "n_seeds": len(results)},
            "decision": {"pass": passed, "reason": reason, "min_gain": float(min_gain)},
            "weight_w": weight_w,
            "runtime_delta_clip": runtime_clip,
            "max_rows": int(max_rows),
            "max_per_category": int(per_category),
            "n_held_out_items": int(len(held_out)),
            "n_held_out_rows": int(len(val_df)),
        },
        "results": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_markdown(payload, report_md)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--joined", default="data/joined.parquet", type=Path)
    parser.add_argument("--runtime-priors", default="data/stage2/priors_v1_locked/runtime_priors.json", type=Path)
    parser.add_argument("--artifact-dir", default="data/stage2/item_residual_v1", type=Path)
    parser.add_argument("--emb", default="data/embeddings/mpnet_v1", type=Path)
    parser.add_argument("--side-meta", default="data/stage2/kfactor_mpnet_mlp_v1/side_feature_meta.json", type=Path)
    parser.add_argument("--stage1", default="data/stage1/kfactor_k4", type=Path)
    parser.add_argument("--stage2", default="data/stage2/kfactor_mpnet_linear_v1", type=Path)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--max-rows", default=5000, type=int)
    parser.add_argument("--per-category", default=1000, type=int)
    parser.add_argument("--val-frac", default=0.1, type=float)
    parser.add_argument("--split-seed", default=0, type=int)
    parser.add_argument("--min-gain", default=0.005, type=float)
    parser.add_argument("--out", default="reports/item_residual_cached_proxy.json", type=Path)
    parser.add_argument("--report-md", default="reports/item_residual_phase3.md", type=Path)
    args = parser.parse_args()
    main(
        args.joined,
        args.runtime_priors,
        args.artifact_dir,
        args.emb,
        args.side_meta,
        args.stage1,
        args.stage2,
        _parse_seeds(args.seeds),
        args.max_rows,
        args.per_category,
        args.val_frac,
        args.split_seed,
        args.min_gain,
        args.out,
        args.report_md,
    )
