"""Fit a K-factor logistic IRT model on the public training responses.

The model is:

    P(y_ij = 1) = sigmoid(U_i @ V_j + Z_j)

where U_i is a K-dimensional subject/model capability vector, V_j is a
K-dimensional item loading vector, and Z_j is an item difficulty/bias term.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


@dataclass
class FitConfig:
    joined_path: str
    out_dir: str
    k: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    val_frac: float
    seed: int
    device: str


class KFactorIRT(nn.Module):
    def __init__(self, n_subjects: int, n_items: int, k: int, z_init: float) -> None:
        super().__init__()
        self.subject_u = nn.Embedding(n_subjects, k)
        self.item_v = nn.Embedding(n_items, k)
        self.item_z = nn.Embedding(n_items, 1)

        nn.init.normal_(self.subject_u.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.item_v.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.item_z.weight, z_init)

    def forward(self, subjects: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        u = self.subject_u(subjects)
        v = self.item_v(items)
        z = self.item_z(items).squeeze(-1)
        return (u * v).sum(dim=1) + z


def _hash_to_unit(value: str, seed: int) -> float:
    import hashlib

    h = hashlib.sha256(f"{seed}|{value}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def _load_rows(joined_path: Path) -> tuple[list[str], list[str], np.ndarray]:
    if not joined_path.exists():
        raise FileNotFoundError(
            f"{joined_path} does not exist. Run scripts/download_data.py first."
        )

    table = pq.read_table(joined_path, columns=["subject_id", "item_id", "label"])
    subjects = table.column("subject_id").to_pylist()
    items = table.column("item_id").to_pylist()
    labels = np.asarray(table.column("label").to_pylist(), dtype=np.float32)

    binary_mask = np.logical_or(labels == 0.0, labels == 1.0)
    if not binary_mask.all():
        subjects = [s for s, keep in zip(subjects, binary_mask) if keep]
        items = [i for i, keep in zip(items, binary_mask) if keep]
        labels = labels[binary_mask]

    return subjects, items, labels


def _encode(values: list[str]) -> tuple[np.ndarray, list[str]]:
    ids = sorted(set(values))
    lookup = {value: idx for idx, value in enumerate(ids)}
    encoded = np.fromiter((lookup[value] for value in values), dtype=np.int64)
    return encoded, ids


def _make_row_split(
    raw_subjects: list[str],
    raw_items: list[str],
    val_frac: float,
    seed: int,
) -> np.ndarray:
    if val_frac <= 0.0:
        return np.zeros(len(raw_subjects), dtype=bool)
    return np.asarray(
        [
            _hash_to_unit(f"{row_idx}|{subject_id}|{item_id}", seed) < val_frac
            for row_idx, (subject_id, item_id) in enumerate(zip(raw_subjects, raw_items))
        ],
        dtype=bool,
    )


def _loss_on_loader(
    model: KFactorIRT,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for subjects, items, labels in loader:
            subjects = subjects.to(device)
            items = items.to(device)
            labels = labels.to(device)
            logits = model(subjects, items)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * len(labels)
            total_n += len(labels)
    return total_loss / max(total_n, 1)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fit(config: FitConfig) -> dict[str, object]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    joined_path = Path(config.joined_path)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_subjects, raw_items, labels = _load_rows(joined_path)
    subject_idx, subject_ids = _encode(raw_subjects)
    item_idx, item_ids = _encode(raw_items)

    is_val = _make_row_split(raw_subjects, raw_items, config.val_frac, config.seed)
    if is_val.all() or (~is_val).all():
        raise ValueError(
            f"Bad validation split: train={(~is_val).sum()} val={is_val.sum()}"
        )

    train_ds = TensorDataset(
        torch.from_numpy(subject_idx[~is_val]),
        torch.from_numpy(item_idx[~is_val]),
        torch.from_numpy(labels[~is_val]),
    )
    val_ds = TensorDataset(
        torch.from_numpy(subject_idx[is_val]),
        torch.from_numpy(item_idx[is_val]),
        torch.from_numpy(labels[is_val]),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=config.device == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=config.device == "cuda",
    )

    device = torch.device(config.device)
    train_rate = float(labels[~is_val].mean())
    z_init = math.log(train_rate / (1.0 - train_rate))
    model = KFactorIRT(len(subject_ids), len(item_ids), config.k, z_init).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    history = []
    best_val_loss = float("inf")
    best_state = None
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{config.epochs}", leave=False)
        for subjects, items, batch_labels in progress:
            subjects = subjects.to(device)
            items = items.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(subjects, items)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item()) * len(batch_labels)
            train_n += len(batch_labels)
            progress.set_postfix(loss=train_loss_sum / train_n)

        train_loss = train_loss_sum / train_n
        val_loss = _loss_on_loader(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_log_loss": train_loss, "val_log_loss": val_loss})
        print(
            f"epoch={epoch:02d} train_log_loss={train_loss:.6f} "
            f"val_log_loss={val_loss:.6f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    subject_u = model.subject_u.weight.detach().cpu().numpy()
    item_v = model.item_v.weight.detach().cpu().numpy()
    item_z = model.item_z.weight.detach().cpu().numpy().reshape(-1)

    _write_csv(
        out_dir / "subject_capabilities.csv",
        [
            {"subject_id": sid, **{f"u_{d}": float(subject_u[i, d]) for d in range(config.k)}}
            for i, sid in enumerate(subject_ids)
        ],
        ["subject_id", *[f"u_{d}" for d in range(config.k)]],
    )
    _write_csv(
        out_dir / "item_parameters.csv",
        [
            {
                "item_id": iid,
                **{f"v_{d}": float(item_v[i, d]) for d in range(config.k)},
                "z": float(item_z[i]),
            }
            for i, iid in enumerate(item_ids)
        ],
        ["item_id", *[f"v_{d}" for d in range(config.k)], "z"],
    )

    state_path = out_dir / "model_state.pt"
    torch.save(
        {
            "config": asdict(config),
            "subject_ids": subject_ids,
            "item_ids": item_ids,
            "state_dict": model.state_dict(),
        },
        state_path,
    )

    summary = {
        "config": asdict(config),
        "n_rows": int(len(labels)),
        "n_train_rows": int((~is_val).sum()),
        "n_val_rows": int(is_val.sum()),
        "n_subjects": len(subject_ids),
        "n_items": len(item_ids),
        "train_positive_rate": train_rate,
        "val_positive_rate": float(labels[is_val].mean()),
        "best_val_log_loss": best_val_loss,
        "best_val_mean_log_likelihood": -best_val_loss,
        "history": history,
        "outputs": {
            "subject_capabilities": str(out_dir / "subject_capabilities.csv"),
            "item_parameters": str(out_dir / "item_parameters.csv"),
            "model_state": str(state_path),
        },
    }
    with (out_dir / "fit_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joined", default="data/joined.parquet", type=Path)
    parser.add_argument("--out", default="stage_1/k_factor_irt/outputs/k_factor_irt", type=Path)
    parser.add_argument("--k", default=4, type=int)
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("--batch-size", default=8192, type=int)
    parser.add_argument("--lr", default=3e-2, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--val-frac", default=0.1, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = FitConfig(
        joined_path=str(args.joined),
        out_dir=str(args.out),
        k=args.k,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_frac=args.val_frac,
        seed=args.seed,
        device=args.device,
    )
    summary = fit(config)
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
