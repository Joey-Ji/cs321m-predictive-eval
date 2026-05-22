"""Train a HistGradientBoostingClassifier on engineered features.

Saves artifacts that the runtime in submissions/v1_kfactor/model.py loads.

Features at predict() time:
  benchmark (cat, vocab from training)
  condition (cat, vocab from training)
  item_content_len, item_has_numbers, item_has_code (cheap from item text)
  subject_mean_correct, subject_n_obs (lookup table built from training)
  prior_logit (from existing PRIOR_ONLY scoring)
  pca_0..pca_{PCA_DIM-1} (mpnet embedding -> PCA at predict() time)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier

from lever_l_utils import (
    auc_roc,
    mean_log_likelihood,
    split_faithful_eval_rows,
    subject_category_probs_for_frame,
    tune_priors,
)
from train_kfactor_residual import _load_joined_frame


EPS = 1e-6
PCA_DIM = 32


def to_logit(p):
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def build_features(
    frame,
    prior_probs,
    benchmark_to_id,
    condition_to_id,
    subject_mean,
    subject_n,
    item_pca,
    item_to_row,
):
    item_idx = frame["item_id"].map(item_to_row).fillna(-1).astype("int64").to_numpy()
    pca_block = np.zeros((len(frame), PCA_DIM), dtype=np.float32)
    mask = item_idx >= 0
    pca_block[mask] = item_pca[item_idx[mask]]
    base = pd.DataFrame({
        "benchmark": frame["benchmark"].map(benchmark_to_id).fillna(-1).astype("int64"),
        "condition": frame["condition"].map(condition_to_id).fillna(-1).astype("int64"),
        "item_content_len": frame["item_content"].str.len().fillna(0).astype("float64"),
        "item_has_numbers": frame["item_content"].str.contains(r"\d", regex=True, na=False).astype("float64"),
        "item_has_code": frame["item_content"].str.contains(r"```|def |class ", regex=True, na=False).astype("float64"),
        "subject_mean_correct": frame["subject_key"].map(subject_mean).fillna(0.5).astype("float64"),
        "subject_n_obs": frame["subject_key"].map(subject_n).fillna(0).astype("float64"),
        "prior_logit": to_logit(prior_probs).astype("float64"),
    }).reset_index(drop=True)
    pca_df = pd.DataFrame(pca_block, columns=[f"pca_{i}" for i in range(PCA_DIM)]).reset_index(drop=True)
    return pd.concat([base, pca_df], axis=1)


def main(
    joined: Path,
    emb_dir: Path,
    out: Path,
    eval_seed: int,
    max_iter: int,
    learning_rate: float,
    max_leaf_nodes: int,
    min_samples_leaf: int,
    l2_regularization: float,
    full_data: bool,
):
    out.mkdir(parents=True, exist_ok=True)
    print(f"loading joined parquet ...", flush=True)
    df = _load_joined_frame(joined)
    item_id_order = [str(iid) for iid in json.loads((emb_dir / "item_id_order.json").read_text())]
    item_to_row = {iid: i for i, iid in enumerate(item_id_order)}
    item_emb = np.load(emb_dir / "item_embeddings.npy").astype(np.float32)
    enc_meta = json.loads((emb_dir / "encoder_meta.json").read_text())
    print(f"rows={len(df)} items={len(item_id_order)} emb_dim={item_emb.shape[1]}", flush=True)

    print(f"fitting PCA-{PCA_DIM} ...", flush=True)
    pca = PCA(n_components=PCA_DIM, random_state=0)
    item_pca = pca.fit_transform(item_emb).astype(np.float32)
    print(f"PCA explained_variance={pca.explained_variance_ratio_.sum():.3f}", flush=True)

    benchmark_to_id = {b: i for i, b in enumerate(sorted(df["benchmark"].unique()))}
    condition_to_id = {c: i for i, c in enumerate(sorted(df["condition"].unique()))}

    # Held-out validation to compute honest val mll (only when not full_data)
    if full_data:
        train_df = df.copy()
        val_df = df.iloc[:0].copy()
    else:
        held_out, train_df, val_df = split_faithful_eval_rows(
            df, item_id_order, val_frac=0.1, seed=eval_seed, max_rows=1500, per_category=300,
        )

    priors, kappas = tune_priors(train_df, val_df if len(val_df) else train_df)
    prior_probs_train = subject_category_probs_for_frame(train_df, priors)

    subject_mean = train_df.groupby("subject_key")["label"].mean()
    subject_n = train_df.groupby("subject_key").size()

    X_train = build_features(
        train_df, prior_probs_train, benchmark_to_id, condition_to_id,
        subject_mean, subject_n, item_pca, item_to_row,
    )
    y_train = train_df["label"].to_numpy(dtype=np.float64)
    print(f"training GBM on {len(y_train):,} rows, {X_train.shape[1]} features ...", flush=True)
    t0 = time.time()
    gbm = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        categorical_features=[0, 1],
        random_state=0,
        early_stopping=False,
    )
    gbm.fit(X_train, y_train)
    print(f"trained in {time.time() - t0:.0f}s, n_iter={gbm.n_iter_}", flush=True)

    val_mll = None
    val_auc = None
    if len(val_df):
        prior_probs_val = subject_category_probs_for_frame(val_df, priors)
        X_val = build_features(
            val_df, prior_probs_val, benchmark_to_id, condition_to_id,
            subject_mean, subject_n, item_pca, item_to_row,
        )
        y_val = val_df["label"].to_numpy(dtype=np.float64)
        p_val = gbm.predict_proba(X_val)[:, 1]
        val_mll = mean_log_likelihood(p_val, y_val)
        val_auc = auc_roc(p_val, y_val)
        prior_mll = mean_log_likelihood(prior_probs_val, y_val)
        print(f"val_mll={val_mll:.4f}  prior_mll={prior_mll:.4f}  delta={val_mll - prior_mll:+.4f}  val_auc={val_auc:.4f}", flush=True)

    # Save artifacts
    (out / "model.pkl").write_bytes(pickle.dumps(gbm, protocol=4))
    (out / "pca.pkl").write_bytes(pickle.dumps(pca, protocol=4))
    np.save(out / "item_pca.npy", item_pca)  # precomputed for known items
    feature_meta = {
        "pca_dim": PCA_DIM,
        "encoder": enc_meta.get("encoder"),
        "encoder_dim": int(item_emb.shape[1]),
        "max_chars": int(enc_meta.get("max_chars", 4000)),
        "benchmark_to_id": benchmark_to_id,
        "condition_to_id": condition_to_id,
        "subject_mean_correct": {str(k): float(v) for k, v in subject_mean.items()},
        "subject_n_obs": {str(k): int(v) for k, v in subject_n.items()},
        "item_to_row": item_to_row,  # for known-item PCA lookup at predict
    }
    (out / "feature_meta.json").write_text(json.dumps(feature_meta) + "\n")
    config = {
        "model": "hist_gradient_boosting",
        "pca_dim": PCA_DIM,
        "max_iter": int(max_iter),
        "learning_rate": float(learning_rate),
        "max_leaf_nodes": int(max_leaf_nodes),
        "min_samples_leaf": int(min_samples_leaf),
        "l2_regularization": float(l2_regularization),
        "full_data": bool(full_data),
        "eval_seed": int(eval_seed),
        "val_mll": (None if val_mll is None else float(val_mll)),
        "val_auc": (None if val_auc is None else float(val_auc)),
        "n_train_rows": int(len(y_train)),
        "n_iter_ran": int(gbm.n_iter_),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(f"wrote GBM artifacts to {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--joined", default="data/joined.parquet", type=Path)
    parser.add_argument("--emb", default="data/embeddings/mpnet_v1", type=Path)
    parser.add_argument("--out", default="data/stage2/gbm_v1", type=Path)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=63)
    parser.add_argument("--min-samples-leaf", type=int, default=50)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--full-data", action="store_true")
    args = parser.parse_args()
    main(
        args.joined, args.emb, args.out, args.eval_seed,
        args.max_iter, args.lr, args.max_leaf_nodes, args.min_samples_leaf, args.l2,
        args.full_data,
    )
