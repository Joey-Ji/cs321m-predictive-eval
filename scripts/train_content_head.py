"""Stage 2a (part 2): train a regressor from item embeddings to IRT difficulty (b).

Reads:
  data/irt/b.pt, data/irt/item_to_id.json   — Stage 1 output
  data/embeddings/item_embeddings.npy, data/embeddings/item_id_order.json   — Stage 2a output

Writes:
  data/head/head.pt          — head state_dict
  data/head/head_meta.json   — architecture, encoder, validation metrics

Default head = linear probe (matches Truong et al.).
Upgrade = 2-layer MLP (--head mlp).

Validation = item-cold-start split (held-out items never seen during head training).

Usage:
    python scripts/train_content_head.py
    python scripts/train_content_head.py --head mlp --hidden 256 --epochs 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import RAW_ITEM_TEXT_VERSION


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
    raise ValueError(head_type)


def main(
    irt_dir: Path,
    emb_dir: Path,
    out_dir: Path,
    head_type: str,
    hidden: int,
    targets: str,
    epochs: int,
    lr: float,
    val_frac: float,
    seed: int,
) -> None:
    import numpy as np
    import torch
    import torch.nn as nn

    from src.validation import _hash_to_unit

    out_dir.mkdir(parents=True, exist_ok=True)

    item_to_id = json.loads((irt_dir / "item_to_id.json").read_text())
    b = torch.load(irt_dir / "b.pt").numpy()  # [n_items]
    log_a = None
    if targets == "b+log_a":
        if not (irt_dir / "log_a.pt").exists():
            raise FileNotFoundError(f"--targets b+log_a requires {irt_dir}/log_a.pt (run fit_irt.py with --model 2pl)")
        log_a = torch.load(irt_dir / "log_a.pt").numpy()
        if float(np.abs(log_a).sum()) == 0.0:
            print("WARN: log_a.pt is all zeros (1PL Stage 1). Predicting it is degenerate; "
                  "consider re-fitting Stage 1 with --model 2pl, or use --targets b.")

    item_id_order = json.loads((emb_dir / "item_id_order.json").read_text())
    embeddings = np.load(emb_dir / "item_embeddings.npy")
    enc_meta = json.loads((emb_dir / "encoder_meta.json").read_text())
    if enc_meta.get("feature_text_version") not in (None, RAW_ITEM_TEXT_VERSION):
        raise ValueError(
            f"IRT content head expects raw item text embeddings ({RAW_ITEM_TEXT_VERSION}); "
            f"got {enc_meta.get('feature_text_version')!r}"
        )

    rows = []
    for row_idx, iid in enumerate(item_id_order):
        if iid in item_to_id:
            j = item_to_id[iid]
            tgt = (b[j],) if log_a is None else (b[j], log_a[j])
            rows.append((row_idx, tgt, iid))
    print(f"Aligned {len(rows):,} items between IRT and embedding sets   targets={targets}")

    X = np.stack([embeddings[r[0]] for r in rows])
    y = np.array([r[1] for r in rows], dtype=np.float32)  # [N, out_dim]
    iids = [r[2] for r in rows]
    out_dim = y.shape[1]

    is_val = np.array([_hash_to_unit(iid, seed) < val_frac for iid in iids])
    X_train, y_train = X[~is_val], y[~is_val]
    X_val, y_val = X[is_val], y[is_val]
    print(f"  train: {len(X_train):,}   val (item-cold-start): {len(X_val):,}   out_dim: {out_dim}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = build_head(X.shape[1], out_dim, head_type, hidden).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    Xt = torch.from_numpy(X_train).to(device)
    yt = torch.from_numpy(y_train).to(device)
    Xv = torch.from_numpy(X_val).to(device)
    yv = torch.from_numpy(y_val).to(device)

    print(f"Training {head_type} head ({out_dim}-d output) for {epochs} epochs ...")
    best_val = float("inf")
    for ep in range(epochs):
        head.train()
        opt.zero_grad()
        loss = loss_fn(head(Xt), yt)
        loss.backward()
        opt.step()
        if (ep + 1) % max(1, epochs // 10) == 0 or ep == 0:
            head.eval()
            with torch.no_grad():
                vl = float(loss_fn(head(Xv), yv))
            best_val = min(best_val, vl)
            print(f"  epoch {ep + 1:>4d}/{epochs}  train_mse={float(loss):.4f}  val_mse={vl:.4f}")

    head.eval()
    with torch.no_grad():
        val_pred = head(Xv).cpu().numpy()
        val_true = y_val
        per_dim_mse = ((val_true - val_pred) ** 2).mean(axis=0).tolist()
        per_dim_r2 = []
        for d in range(out_dim):
            t = val_true[:, d]
            p = val_pred[:, d]
            ss_tot = float(((t - t.mean()) ** 2).sum())
            ss_res = float(((t - p) ** 2).sum())
            per_dim_r2.append(1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))

    torch.save(head.state_dict(), out_dir / "head.pt")
    (out_dir / "head_meta.json").write_text(
        json.dumps(
            {
                "head_type": head_type,
                "hidden": hidden,
                "in_dim": X.shape[1],
                "out_dim": out_dim,
                "targets": targets,                 # "b" or "b+log_a"
                "target_order": ["b"] if out_dim == 1 else ["b", "log_a"],
                "encoder": enc_meta["encoder"],
                "feature_text_version": enc_meta.get("feature_text_version", RAW_ITEM_TEXT_VERSION),
                "n_train": int(len(X_train)),
                "n_val": int(len(X_val)),
                "val_mse": float(np.mean(per_dim_mse)),
                "val_mse_per_dim": per_dim_mse,
                "val_r2_per_dim": per_dim_r2,
                "epochs": epochs,
                "lr": lr,
                "val_frac": val_frac,
                "seed": seed,
            },
            indent=2,
        )
    )
    print(f"Done. per-dim val_mse={per_dim_mse}  per-dim val_r2={per_dim_r2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--irt", default="data/irt", type=Path)
    parser.add_argument("--emb", default="data/embeddings", type=Path)
    parser.add_argument("--out", default="data/head", type=Path)
    parser.add_argument("--head", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--targets", choices=("b", "b+log_a"), default="b",
                        help="b = predict difficulty only (1PL/Rasch). "
                             "b+log_a = predict both targets jointly (2PL).")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.irt, args.emb, args.out, args.head, args.hidden, args.targets, args.epochs, args.lr, args.val_frac, args.seed)
