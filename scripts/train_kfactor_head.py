"""Train a cold-start head from item embeddings to K-factor item parameters."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def main(
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

    item_targets = load_item_targets(stage1_dir / "item_targets.pt")
    item_to_id = json.loads((stage1_dir / "item_to_id.json").read_text())
    item_v = item_targets["item_v"].detach().cpu().numpy().astype(np.float32)
    item_z = item_targets["item_z"].detach().cpu().numpy().astype(np.float32)
    if item_v.ndim != 2 or item_z.ndim != 1 or item_v.shape[0] != item_z.shape[0]:
        raise ValueError(f"invalid item target shapes: item_v={item_v.shape} item_z={item_z.shape}")

    item_id_order = [str(iid) for iid in json.loads((emb_dir / "item_id_order.json").read_text())]
    embeddings = np.load(emb_dir / "item_embeddings.npy").astype(np.float32)
    enc_meta = json.loads((emb_dir / "encoder_meta.json").read_text())
    if embeddings.ndim != 2 or embeddings.shape[0] != len(item_id_order):
        raise ValueError(f"embedding shape {embeddings.shape} inconsistent with {len(item_id_order)} item IDs")

    rows: list[tuple[int, np.ndarray, str]] = []
    for row_idx, item_id in enumerate(item_id_order):
        target_idx = item_to_id.get(item_id)
        if target_idx is None:
            continue
        y_row = np.concatenate([item_v[target_idx], np.array([item_z[target_idx]], dtype=np.float32)])
        rows.append((row_idx, y_row.astype(np.float32), item_id))
    if not rows:
        raise ValueError("no overlapping item IDs between embeddings and Stage 1 item targets")

    X = np.stack([embeddings[row_idx] for row_idx, _, _ in rows]).astype(np.float32)
    y = np.stack([target for _, target, _ in rows]).astype(np.float32)
    iids = [item_id for _, _, item_id in rows]
    held_out = validation_item_ids(iids, val_frac=val_frac, seed=seed)
    is_val = np.array([iid in held_out for iid in iids], dtype=bool)
    if is_val.all() or (~is_val).sum() == 0:
        raise ValueError("validation split left no training items")

    X_train, y_train = X[~is_val], y[~is_val]
    X_val, y_val = X[is_val], y[is_val]
    if len(X_val) == 0:
        raise ValueError("validation split left no validation items; increase --val-frac")

    target_mean = y_train.mean(axis=0)
    target_std = y_train.std(axis=0)
    target_std = np.where(target_std < 1e-6, 1.0, target_std).astype(np.float32)
    y_train_std = ((y_train - target_mean) / target_std).astype(np.float32)
    y_val_std = ((y_val - target_mean) / target_std).astype(np.float32)

    in_dim = int(X.shape[1])
    out_dim = int(y.shape[1])
    k = out_dim - 1
    target_order = [f"v_{i}" for i in range(k)] + ["z"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = build_head(in_dim, out_dim, head_type, hidden).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    Xt = torch.from_numpy(X_train).to(device)
    yt = torch.from_numpy(y_train_std).to(device)
    Xv = torch.from_numpy(X_val).to(device)
    yv = torch.from_numpy(y_val_std).to(device)

    print(f"Aligned {len(rows):,} items between Stage 1 targets and embeddings")
    print(f"  train={len(X_train):,} val={len(X_val):,} in_dim={in_dim} out_dim={out_dim} k={k}")
    print(f"Training {head_type} head for {epochs} epochs on {device} ...")
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

    torch.save(head.state_dict(), out_dir / "head.pt")
    (out_dir / "head_meta.json").write_text(
        json.dumps(
            {
                "head_type": head_type,
                "hidden": hidden,
                "in_dim": in_dim,
                "out_dim": out_dim,
                "k": k,
                "target_order": target_order,
                "encoder": enc_meta.get("encoder"),
                "encoder_dim": enc_meta.get("dim"),
                "feature_text_version": enc_meta.get("feature_text_version"),
                "val_frac": val_frac,
                "seed": seed,
                "split": "item-cold",
                "val_item_ids": sorted(held_out),
                "epochs": epochs,
                "lr": lr,
            },
            indent=2,
        )
    )
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
    print(f"Done. Wrote head artifacts to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", default="data/stage1/kfactor_k4", type=Path)
    parser.add_argument("--emb", default="data/embeddings/mpnet_v1", type=Path)
    parser.add_argument("--out", default="data/stage2/kfactor_mpnet_linear_v1", type=Path)
    parser.add_argument("--head", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.stage1, args.emb, args.out, args.head, args.hidden, args.epochs, args.lr, args.val_frac, args.seed)
