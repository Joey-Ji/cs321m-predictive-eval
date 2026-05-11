"""Fit a K-factor logistic IRT model on the public training responses.

The model is:

    P(y_ij = 1) = sigmoid(S_i + U_i @ V_j + Z_j)

where S_i is a subject/model bias, U_i is a K-dimensional subject/model
capability vector, V_j is a K-dimensional item loading vector, and Z_j is an
item difficulty/bias term.
"""

from __future__ import annotations

import argparse
import csv
import json
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
    warmup_epochs: float
    warmup_start_factor: float
    lr_factor: float
    lr_patience: int
    min_lr: float
    weight_decay: float
    val_frac: float
    smoothing: float
    seed: int
    device: str


class KFactorIRT(nn.Module):
    def __init__(
        self,
        n_subjects: int,
        n_items: int,
        k: int,
        subject_bias_init: np.ndarray,
        item_bias_init: np.ndarray,
    ) -> None:
        super().__init__()
        self.subject_bias = nn.Embedding(n_subjects, 1, sparse=True)
        self.subject_u = nn.Embedding(n_subjects, k, sparse=True)
        self.item_v = nn.Embedding(n_items, k, sparse=True)
        self.item_z = nn.Embedding(n_items, 1, sparse=True)

        nn.init.normal_(self.subject_u.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.item_v.weight, mean=0.0, std=0.1)
        with torch.no_grad():
            self.subject_bias.weight.copy_(
                torch.as_tensor(subject_bias_init, dtype=torch.float32).view(-1, 1)
            )
            self.item_z.weight.copy_(
                torch.as_tensor(item_bias_init, dtype=torch.float32).view(-1, 1)
            )

    def forward(self, subjects: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        s = self.subject_bias(subjects).squeeze(-1)
        u = self.subject_u(subjects)
        v = self.item_v(items)
        z = self.item_z(items).squeeze(-1)
        return s + (u * v).sum(dim=1) + z

    def batch_l2(self, subjects: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        s = self.subject_bias(subjects).squeeze(-1)
        u = self.subject_u(subjects)
        v = self.item_v(items)
        z = self.item_z(items).squeeze(-1)
        return (
            s.square().mean()
            + u.square().mean()
            + v.square().mean()
            + z.square().mean()
        )


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


def _safe_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1.0 - 1e-4)
    return np.log(p / (1.0 - p))


def _marginal_initializers(
    subject_idx: np.ndarray,
    item_idx: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    n_subjects: int,
    n_items: int,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    train_subjects = subject_idx[train_mask]
    train_items = item_idx[train_mask]
    train_labels = labels[train_mask]

    global_rate = float(train_labels.mean())
    global_logit = float(_safe_logit(np.asarray([global_rate]))[0])

    subject_counts = np.bincount(train_subjects, minlength=n_subjects).astype(np.float64)
    subject_sums = np.bincount(
        train_subjects, weights=train_labels, minlength=n_subjects
    ).astype(np.float64)
    item_counts = np.bincount(train_items, minlength=n_items).astype(np.float64)
    item_sums = np.bincount(train_items, weights=train_labels, minlength=n_items).astype(
        np.float64
    )

    subject_rates = (subject_sums + smoothing * global_rate) / (
        subject_counts + smoothing
    )
    item_rates = (item_sums + smoothing * global_rate) / (item_counts + smoothing)

    subject_bias_init = _safe_logit(subject_rates) - global_logit
    item_bias_init = _safe_logit(item_rates)
    return (
        subject_bias_init.astype(np.float32),
        item_bias_init.astype(np.float32),
        global_rate,
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
    if is_val.all():
        raise ValueError(
            f"Bad validation split: train={(~is_val).sum()} val={is_val.sum()}"
        )
    has_val = bool(is_val.any())
    train_mask = ~is_val

    train_ds = TensorDataset(
        torch.from_numpy(subject_idx[train_mask]),
        torch.from_numpy(item_idx[train_mask]),
        torch.from_numpy(labels[train_mask]),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=config.device == "cuda",
    )
    val_loader = None
    if has_val:
        val_ds = TensorDataset(
            torch.from_numpy(subject_idx[is_val]),
            torch.from_numpy(item_idx[is_val]),
            torch.from_numpy(labels[is_val]),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=config.device == "cuda",
        )

    device = torch.device(config.device)
    subject_bias_init, item_bias_init, train_rate = _marginal_initializers(
        subject_idx=subject_idx,
        item_idx=item_idx,
        labels=labels,
        train_mask=train_mask,
        n_subjects=len(subject_ids),
        n_items=len(item_ids),
        smoothing=config.smoothing,
    )
    model = KFactorIRT(
        len(subject_ids),
        len(item_ids),
        config.k,
        subject_bias_init=subject_bias_init,
        item_bias_init=item_bias_init,
    ).to(device)
    initial_lr = (
        config.lr * config.warmup_start_factor
        if config.warmup_epochs > 0.0
        else config.lr
    )
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_factor,
        patience=config.lr_patience,
        min_lr=config.min_lr,
    )
    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    history = []
    best_val_loss = float("inf")
    best_state = None
    warmup_steps = int(config.warmup_epochs * len(train_loader))
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{config.epochs}", leave=False)
        for batch_idx, (subjects, items, batch_labels) in enumerate(progress, start=1):
            global_step = (epoch - 1) * len(train_loader) + batch_idx
            if warmup_steps > 0 and global_step <= warmup_steps:
                alpha = global_step / warmup_steps
                lr = config.lr * (
                    config.warmup_start_factor
                    + alpha * (1.0 - config.warmup_start_factor)
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr

            subjects = subjects.to(device)
            items = items.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(subjects, items)
            bce_loss = criterion(logits, batch_labels)
            loss = bce_loss
            if config.weight_decay > 0.0:
                loss = loss + config.weight_decay * model.batch_l2(subjects, items)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(bce_loss.item()) * len(batch_labels)
            train_n += len(batch_labels)
            progress.set_postfix(loss=train_loss_sum / train_n, lr=optimizer.param_groups[0]["lr"])

        train_loss = train_loss_sum / train_n
        val_loss = (
            _loss_on_loader(model, val_loader, criterion, device)
            if val_loader is not None
            else None
        )
        monitor_loss = val_loss if val_loss is not None else train_loss
        if epoch > config.warmup_epochs:
            scheduler.step(monitor_loss)
        history_row = {
            "epoch": epoch,
            "train_log_loss": train_loss,
            "val_log_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(history_row)
        if val_loss is None:
            print(f"epoch={epoch:02d} train_log_loss={train_loss:.6f}")
        else:
            print(
                f"epoch={epoch:02d} train_log_loss={train_loss:.6f} "
                f"val_log_loss={val_loss:.6f}"
            )
        if monitor_loss < best_val_loss:
            best_val_loss = monitor_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    subject_bias = model.subject_bias.weight.detach().cpu().numpy().reshape(-1)
    subject_u = model.subject_u.weight.detach().cpu().numpy()
    item_v = model.item_v.weight.detach().cpu().numpy()
    item_z = model.item_z.weight.detach().cpu().numpy().reshape(-1)

    _write_csv(
        out_dir / "subject_capabilities.csv",
        [
            {
                "subject_id": sid,
                "subject_bias": float(subject_bias[i]),
                **{f"u_{d}": float(subject_u[i, d]) for d in range(config.k)},
            }
            for i, sid in enumerate(subject_ids)
        ],
        ["subject_id", "subject_bias", *[f"u_{d}" for d in range(config.k)]],
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
        "n_train_rows": int(train_mask.sum()),
        "n_val_rows": int(is_val.sum()),
        "n_subjects": len(subject_ids),
        "n_items": len(item_ids),
        "train_positive_rate": train_rate,
        "val_positive_rate": float(labels[is_val].mean()) if has_val else None,
        "best_monitor_log_loss": best_val_loss,
        "best_monitor_mean_log_likelihood": -best_val_loss,
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
    parser.add_argument(
        "--warmup-epochs",
        default=0.0,
        type=float,
        help="Linearly ramp LR up to --lr over this many epochs.",
    )
    parser.add_argument(
        "--warmup-start-factor",
        default=0.1,
        type=float,
        help="Initial LR as a fraction of --lr when warmup is enabled.",
    )
    parser.add_argument("--lr-factor", default=0.5, type=float)
    parser.add_argument("--lr-patience", default=2, type=int)
    parser.add_argument("--min-lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--val-frac", default=0.1, type=float)
    parser.add_argument(
        "--smoothing",
        default=20.0,
        type=float,
        help="Pseudo-count strength for subject/item marginal initialization.",
    )
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
        warmup_epochs=args.warmup_epochs,
        warmup_start_factor=args.warmup_start_factor,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        val_frac=args.val_frac,
        smoothing=args.smoothing,
        seed=args.seed,
        device=args.device,
    )
    summary = fit(config)
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
