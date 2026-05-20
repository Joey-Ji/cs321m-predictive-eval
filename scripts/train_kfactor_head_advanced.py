"""Train K-factor head with advanced statistical features.

This is an enhanced version of train_kfactor_head.py that adds:
  - Benchmark-level statistics (difficulty, item count, length distributions)
  - Condition-level statistics (performance, benchmark coverage)
  - Benchmark-condition interaction statistics
  - Stage 1 parameter distribution features
  - Item-level features (length, complexity)

Usage:
    python scripts/train_kfactor_head_advanced.py \
      --joined data/joined.parquet \
      --stage1 data/stage1/kfactor_k4 \
      --emb data/embeddings/mpnet_v1 \
      --out data/stage2/kfactor_mpnet_advanced_v1 \
      --head-type mlp \
      --hidden 256 \
      --epochs 30 \
      --lr 0.001
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.advanced_features import build_advanced_feature_extractor
from src.features import EMBEDDING_REPRESENTATION_VERSION, REPRESENTATION_VERSION
from src.kfactor import load_item_targets, validation_item_ids


def build_head(in_dim: int, out_dim: int, head_type: str, hidden: int):
    import torch.nn as nn

    if head_type == "linear":
        return nn.Linear(in_dim, out_dim)
    if head_type == "mlp":
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
        )
    raise ValueError(f"unsupported head type: {head_type}")


def _finite_float(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite metric: {value}")
    return value


def _per_dim_r2(y_true, y_pred) -> list[float]:
    values: list[float] = []
    for dim in range(y_true.shape[1]):
        true_d = y_true[:, dim]
        pred_d = y_pred[:, dim]
        ss_tot = float(((true_d - true_d.mean()) ** 2).sum())
        ss_res = float(((true_d - pred_d) ** 2).sum())
        values.append(float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 0.0)
    return values


def load_joined_data(joined_path: Path) -> list[dict]:
    """Load joined.parquet and return list of rows as dicts."""
    import pyarrow.parquet as pq

    table = pq.read_table(
        joined_path,
        columns=["item_id", "benchmark", "condition", "item_content", "label"],
    )
    return table.to_pylist()


def main(
    joined_path: Path,
    stage1_dir: Path,
    emb_dir: Path,
    out_dir: Path,
    head_type: str,
    hidden: int,
    epochs: int,
    lr: float,
    val_frac: float,
    seed: int,
) -> None:
    import numpy as np
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading training data...")
    joined_rows = load_joined_data(joined_path)
    print(f"  Loaded {len(joined_rows):,} training rows")

    print("Building advanced feature extractor...")
    feature_extractor = build_advanced_feature_extractor(
        training_rows=joined_rows,
        stage1_dir=stage1_dir,
    )
    print(f"  Feature dimension: {feature_extractor.feature_dim}")
    print(f"  Feature names: {', '.join(feature_extractor.feature_names)}")

    # Save feature extractor for inference time
    feature_extractor.save(out_dir / "advanced_feature_extractor.json")

    print("Loading Stage 1 targets...")
    item_targets = load_item_targets(stage1_dir / "item_targets.pt")
    item_to_id = json.loads((stage1_dir / "item_to_id.json").read_text())
    item_v = item_targets["item_v"].detach().cpu().numpy().astype(np.float32)
    item_z = item_targets["item_z"].detach().cpu().numpy().astype(np.float32)
    if item_v.ndim != 2 or item_z.ndim != 1 or item_v.shape[0] != item_z.shape[0]:
        raise ValueError(f"invalid item target shapes: item_v={item_v.shape} item_z={item_z.shape}")

    print("Loading embeddings and side features...")
    item_id_order = [str(iid) for iid in json.loads((emb_dir / "item_id_order.json").read_text())]
    embeddings = np.load(emb_dir / "item_embeddings.npy").astype(np.float32)
    enc_meta = json.loads((emb_dir / "encoder_meta.json").read_text())
    if enc_meta.get("representation_version") != EMBEDDING_REPRESENTATION_VERSION:
        raise ValueError(
            f"K-factor head requires embeddings with representation_version={EMBEDDING_REPRESENTATION_VERSION!r}; "
            f"got {enc_meta.get('representation_version')!r}"
        )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(item_id_order):
        raise ValueError(f"embedding shape {embeddings.shape} inconsistent with {len(item_id_order)} item IDs")

    side_features = np.load(emb_dir / "item_side_features.npy").astype(np.float32)
    vocab = json.loads((emb_dir / "side_feature_meta.json").read_text())
    side_feature_dim = int(vocab["side_feature_dim"])
    if side_features.ndim != 2 or side_features.shape[0] != embeddings.shape[0]:
        raise ValueError(
            f"side feature shape {side_features.shape} inconsistent with embeddings {embeddings.shape}"
        )
    if side_features.shape[1] != side_feature_dim:
        raise ValueError(
            f"side feature dim {side_features.shape[1]} != vocab side_feature_dim {side_feature_dim}"
        )

    # Build item_id -> row mapping from joined data
    print("Mapping items to joined rows for feature extraction...")
    item_to_joined_row = {}
    for row in joined_rows:
        item_id = str(row.get("item_id", ""))
        if item_id and item_id not in item_to_joined_row:
            item_to_joined_row[item_id] = row

    print("Aligning embeddings with Stage 1 targets and extracting advanced features...")
    rows: list[tuple[int, np.ndarray, str, np.ndarray]] = []
    for row_idx, item_id in enumerate(item_id_order):
        target_idx = item_to_id.get(item_id)
        if target_idx is None:
            continue

        # Extract advanced features for this item
        joined_row = item_to_joined_row.get(item_id)
        if joined_row is None:
            # Item not in joined data, skip (shouldn't happen for training items)
            continue

        advanced_features = feature_extractor.extract(joined_row)

        y_row = np.concatenate([item_v[target_idx], np.array([item_z[target_idx]], dtype=np.float32)])
        rows.append((row_idx, y_row.astype(np.float32), item_id, advanced_features))

    if not rows:
        raise ValueError("no overlapping item IDs between embeddings and Stage 1 item targets")

    print(f"  Aligned {len(rows):,} items")

    # Construct feature matrices
    X_emb = np.stack([embeddings[row_idx] for row_idx, _, _, _ in rows]).astype(np.float32)
    X_side = np.stack([side_features[row_idx] for row_idx, _, _, _ in rows]).astype(np.float32)
    X_advanced = np.stack([adv_feat for _, _, _, adv_feat in rows]).astype(np.float32)

    # Concatenate: [embedding | side_features | advanced_features]
    X = np.concatenate([X_emb, X_side, X_advanced], axis=1).astype(np.float32)
    embedding_dim = int(X_emb.shape[1])
    advanced_feature_dim = int(X_advanced.shape[1])

    y = np.stack([target for _, target, _, _ in rows]).astype(np.float32)
    iids = [item_id for _, _, item_id, _ in rows]

    print("Splitting train/validation...")
    held_out = validation_item_ids(iids, val_frac=val_frac, seed=seed)
    is_val = np.array([iid in held_out for iid in iids], dtype=bool)
    if is_val.all() or (~is_val).sum() == 0:
        raise ValueError("validation split left no training items")

    X_train, y_train = X[~is_val], y[~is_val]
    X_val, y_val = X[is_val], y[is_val]
    if len(X_val) == 0:
        raise ValueError("validation split left no validation items; increase --val-frac")

    # Standardize targets
    target_mean = y_train.mean(axis=0)
    target_std = y_train.std(axis=0)
    target_std = np.where(target_std < 1e-6, 1.0, target_std).astype(np.float32)
    y_train_std = ((y_train - target_mean) / target_std).astype(np.float32)
    y_val_std = ((y_val - target_mean) / target_std).astype(np.float32)

    in_dim = int(X.shape[1])
    out_dim = int(y.shape[1])
    k = out_dim - 1
    target_order = [f"v_{i}" for i in range(k)] + ["z"]

    print(f"\nFeature dimensions:")
    print(f"  Embedding:        {embedding_dim}")
    print(f"  Side features:    {side_feature_dim}")
    print(f"  Advanced features: {advanced_feature_dim}")
    print(f"  Total input:      {in_dim}")
    print(f"  Output (k+1):     {out_dim}")
    print(f"\nDataset split:")
    print(f"  Train: {len(X_train):,} items")
    print(f"  Val:   {len(X_val):,} items")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTraining {head_type} head for {epochs} epochs on {device}...")

    head = build_head(in_dim, out_dim, head_type, hidden).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    Xt = torch.from_numpy(X_train).to(device)
    yt = torch.from_numpy(y_train_std).to(device)
    Xv = torch.from_numpy(X_val).to(device)
    yv = torch.from_numpy(y_val_std).to(device)

    best_val = float("inf")
    for epoch in range(epochs):
        head.train()
        opt.zero_grad()
        pred = head(Xt)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()

        if epoch == 0 or (epoch + 1) % max(1, epochs // 10) == 0:
            head.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(head(Xv), yv))
            best_val = min(best_val, val_loss)
            print(
                f"  epoch {epoch + 1:>4d}/{epochs} "
                f"train_mse_std={float(loss.detach()):.5f} val_mse_std={val_loss:.5f}"
            )

    print("\nComputing final metrics...")
    head.eval()
    with torch.no_grad():
        val_pred_std = head(Xv).cpu().numpy()
    val_pred = (val_pred_std * target_std + target_mean).astype(np.float32)
    val_mse_per_dim = ((y_val - val_pred) ** 2).mean(axis=0)
    val_r2_per_dim = _per_dim_r2(y_val, val_pred)

    metrics = {
        "target_val_mse": _finite_float(float(val_mse_per_dim.mean())),
        "target_val_mse_per_dim": [_finite_float(x) for x in val_mse_per_dim.tolist()],
        "target_val_r2_per_dim": [_finite_float(x) for x in val_r2_per_dim],
        "target_val_mse_standardized": _finite_float(float(((y_val_std - val_pred_std) ** 2).mean())),
        "best_val_mse_standardized": _finite_float(best_val),
        "n_aligned": int(len(rows)),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
    }

    print(f"\nValidation metrics:")
    print(f"  MSE (standardized): {metrics['target_val_mse_standardized']:.5f}")
    print(f"  MSE (original):     {metrics['target_val_mse']:.5f}")
    print(f"  R² per dimension:   {[f'{r:.3f}' for r in val_r2_per_dim]}")

    print("\nSaving outputs...")
    torch.save(head.state_dict(), out_dir / "head.pt")
    (out_dir / "head_meta.json").write_text(
        json.dumps(
            {
                "head_type": head_type,
                "hidden": hidden,
                "in_dim": in_dim,
                "embedding_dim": embedding_dim,
                "side_feature_dim": side_feature_dim,
                "advanced_feature_dim": advanced_feature_dim,
                "advanced_feature_names": feature_extractor.feature_names,
                "out_dim": out_dim,
                "k": k,
                "target_order": target_order,
                "encoder": enc_meta.get("encoder"),
                "encoder_dim": enc_meta.get("dim"),
                "representation_version": REPRESENTATION_VERSION,
                "embedding_representation_version": enc_meta.get("representation_version"),
                "max_chars": enc_meta.get("max_chars", 4000),
                "val_frac": val_frac,
                "seed": seed,
                "split": "item-cold",
                "val_item_ids": sorted(held_out),
                "epochs": epochs,
                "lr": lr,
                "uses_advanced_features": True,
            },
            indent=2,
        )
    )
    (out_dir / "side_feature_meta.json").write_text(json.dumps(vocab, indent=2))
    (out_dir / "target_scaler.json").write_text(
        json.dumps(
            {
                "target_order": target_order,
                "mean": target_mean.astype(float).tolist(),
                "std": target_std.astype(float).tolist(),
            },
            indent=2,
        )
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nDone! Outputs saved to {out_dir}")
    print(f"  head.pt")
    print(f"  head_meta.json")
    print(f"  advanced_feature_extractor.json")
    print(f"  metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joined", type=Path, required=True, help="Path to joined.parquet")
    parser.add_argument("--stage1", type=Path, required=True, help="Stage 1 output directory")
    parser.add_argument("--emb", type=Path, required=True, help="Embeddings directory")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--head-type", default="mlp", choices=["linear", "mlp"])
    parser.add_argument("--hidden", type=int, default=256, help="MLP hidden dimension")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    main(
        args.joined,
        args.stage1,
        args.emb,
        args.out,
        args.head_type,
        args.hidden,
        args.epochs,
        args.lr,
        args.val_frac,
        args.seed,
    )
