"""Stage 1: fit Rasch (1PL) or 2PL IRT on the training response matrix.

Reads data/joined.parquet (output of download_data.py).
Writes:
  data/irt/theta.pt           — tensor [n_subjects], subject ability
  data/irt/b.pt               — tensor [n_items], item difficulty
  data/irt/log_a.pt           — tensor [n_items], log-discrimination (2PL only; zeros for 1PL)
  data/irt/subject_to_id.json — str -> int (normalized name lookup)
  data/irt/item_to_id.json    — str -> int
  data/irt/fit_log.json       — final loss, AUC on training, hyperparams

Usage:
    python scripts/fit_irt.py --model 1pl --epochs 200
    python scripts/fit_irt.py --model 2pl --epochs 300 --reg 1e-3
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

NAME_LINE = re.compile(r"^\s*Name:\s*(.+?)\s*$", re.MULTILINE)


def normalize_subject(subject_content: str) -> str:
    """Extract a stable lookup key from subject_content (which is text, not a dict)."""
    if not subject_content:
        return ""
    m = NAME_LINE.search(subject_content)
    return m.group(1).strip().lower() if m else subject_content.strip().lower()


class IRT:
    """Pure-PyTorch IRT implementation. 1PL or 2PL."""

    def __init__(self, n_subjects: int, n_items: int, model: str = "1pl"):
        import torch

        self.model = model
        self.theta = torch.nn.Parameter(torch.zeros(n_subjects))
        self.b = torch.nn.Parameter(torch.zeros(n_items))
        if model == "2pl":
            self.log_a = torch.nn.Parameter(torch.zeros(n_items))
        else:
            self.log_a = None

    def parameters(self):
        ps = [self.theta, self.b]
        if self.log_a is not None:
            ps.append(self.log_a)
        return ps

    def forward(self, s_ids, i_ids):
        import torch

        diff = self.theta[s_ids] - self.b[i_ids]
        if self.log_a is not None:
            return torch.sigmoid(self.log_a[i_ids].exp() * diff)
        return torch.sigmoid(diff)


def main(joined_path: Path, out_dir: Path, model: str, epochs: int, batch: int, lr: float, reg: float) -> None:
    import numpy as np
    import pyarrow.parquet as pq
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {joined_path} ...")
    table = pq.read_table(joined_path)
    n_rows = table.num_rows
    print(f"  rows: {n_rows:,}")

    subj_contents = table.column("subject_content").to_pylist()
    item_ids_raw = table.column("item_id").to_pylist()
    labels_raw = table.column("label").to_pylist()

    subject_keys = [normalize_subject(s) for s in subj_contents]
    subject_to_id = {s: i for i, s in enumerate(sorted(set(subject_keys)))}
    item_to_id = {it: i for i, it in enumerate(sorted(set(item_ids_raw)))}
    print(f"  subjects: {len(subject_to_id):,}")
    print(f"  items   : {len(item_to_id):,}")

    s_ids = np.fromiter((subject_to_id[s] for s in subject_keys), dtype=np.int64)
    i_ids = np.fromiter((item_to_id[it] for it in item_ids_raw), dtype=np.int64)
    y = np.fromiter((float(l) for l in labels_raw), dtype=np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")

    irt = IRT(len(subject_to_id), len(item_to_id), model=model)
    for p in irt.parameters():
        p.data = p.data.to(device)
    opt = torch.optim.Adam(irt.parameters(), lr=lr)

    s_t = torch.from_numpy(s_ids).to(device)
    i_t = torch.from_numpy(i_ids).to(device)
    y_t = torch.from_numpy(y).to(device)

    n = n_rows
    print(f"Training {model.upper()} for {epochs} epochs, batch={batch}, lr={lr}, reg={reg} ...")
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch):
            idx = perm[start : start + batch]
            p = irt.forward(s_t[idx], i_t[idx]).clamp(1e-6, 1 - 1e-6)
            yb = y_t[idx]
            bce = -(yb * p.log() + (1 - yb) * (1 - p).log()).mean()
            penalty = sum(pp.pow(2).mean() for pp in irt.parameters()) * reg
            loss = bce + penalty
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(bce)
            n_batches += 1
        if (ep + 1) % max(1, epochs // 10) == 0 or ep == 0:
            print(f"  epoch {ep + 1:>4d}/{epochs}  bce={total_loss / n_batches:.4f}")

    print("Saving ...")
    torch.save(irt.theta.detach().cpu(), out_dir / "theta.pt")
    torch.save(irt.b.detach().cpu(), out_dir / "b.pt")
    if irt.log_a is not None:
        torch.save(irt.log_a.detach().cpu(), out_dir / "log_a.pt")
    else:
        torch.save(torch.zeros(len(item_to_id)), out_dir / "log_a.pt")

    (out_dir / "subject_to_id.json").write_text(json.dumps(subject_to_id, indent=2))
    (out_dir / "item_to_id.json").write_text(json.dumps(item_to_id, indent=2))

    log = {
        "model": model,
        "n_subjects": len(subject_to_id),
        "n_items": len(item_to_id),
        "n_rows": n_rows,
        "epochs": epochs,
        "lr": lr,
        "reg": reg,
        "final_bce": total_loss / n_batches,
    }
    (out_dir / "fit_log.json").write_text(json.dumps(log, indent=2))
    print(f"Done. Outputs in {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--joined", default="data/joined.parquet", type=Path)
    parser.add_argument("--out", default="data/irt", type=Path)
    parser.add_argument("--model", choices=("1pl", "2pl"), default="1pl")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--reg", type=float, default=1e-4)
    args = parser.parse_args()
    main(args.joined, args.out, args.model, args.epochs, args.batch, args.lr, args.reg)
