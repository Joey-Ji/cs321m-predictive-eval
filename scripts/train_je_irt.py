"""Train a JE-IRT scoring head on frozen cached item embeddings."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn

from je_irt_utils import JEIRTModel
from lever_l_utils import auc_roc, mean_log_likelihood, split_faithful_eval_rows
from train_kfactor_residual import _load_joined_frame


def _finite_float(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite metric: {value}")
    return value


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def _set_deterministic(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def _subject_mapping(df) -> dict[str, int]:
    subjects = sorted(str(subject_key) for subject_key in df["subject_key"].dropna().unique().tolist())
    return {subject_key: idx for idx, subject_key in enumerate(subjects)}


def _arrays_for_rows(df, subject_to_id: dict[str, int], item_to_row: dict[str, int]) -> tuple[np.ndarray, ...]:
    subject_idx = df["subject_key"].map(subject_to_id).fillna(-1).to_numpy(dtype=np.int64)
    item_idx = df["item_id"].map(item_to_row).fillna(-1).to_numpy(dtype=np.int64)
    labels = df["label"].to_numpy(dtype=np.float32)
    keep = (subject_idx >= 0) & (item_idx >= 0)
    return subject_idx[keep], item_idx[keep], labels[keep]


def _evaluate(
    model: JEIRTModel,
    subject_idx: np.ndarray,
    item_idx: np.ndarray,
    labels: np.ndarray,
    item_embeddings_t: torch.Tensor,
    batch_size: int,
    device: str,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    total = 0
    probs: list[np.ndarray] = []
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            stop = min(start + batch_size, len(labels))
            s = torch.from_numpy(subject_idx[start:stop]).to(device)
            i = torch.from_numpy(item_idx[start:stop]).to(device)
            y = torch.from_numpy(labels[start:stop]).to(device)
            logits = model(s, item_embeddings_t[i])
            loss = loss_fn(logits, y)
            loss_sum += float(loss.detach())
            total += len(y)
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    if total == 0:
        raise ValueError("no rows available for evaluation")
    prob_np = np.concatenate(probs, axis=0).astype(np.float64)
    label_np = labels.astype(np.int8)
    return {
        "bce": loss_sum / total,
        "mean_log_likelihood": mean_log_likelihood(prob_np, label_np),
        "auc_roc": auc_roc(prob_np, label_np),
    }


def _train_epoch(
    model: JEIRTModel,
    opt: torch.optim.Optimizer,
    subject_idx: np.ndarray,
    item_idx: np.ndarray,
    labels: np.ndarray,
    item_embeddings_t: torch.Tensor,
    batch_size: int,
    seed: int,
    device: str,
) -> float:
    model.train()
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    order = torch.randperm(len(labels), generator=torch.Generator().manual_seed(seed))
    total_loss = 0.0
    total = 0
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size].numpy()
        s = torch.from_numpy(subject_idx[idx]).to(device)
        i = torch.from_numpy(item_idx[idx]).to(device)
        y = torch.from_numpy(labels[idx]).to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(s, item_embeddings_t[i])
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        total_loss += float(loss.detach())
        total += len(y)
    if total == 0:
        raise ValueError("no rows available for training")
    return -total_loss / total


def main(
    joined: Path,
    emb: Path,
    out: Path,
    epochs: int,
    lr: float,
    val_frac: float,
    seed: int,
    hidden: int,
    dim: int,
    batch: int,
    weight_decay: float,
    dropout: float,
    max_rows: int,
    val_rows: int,
    per_category: int,
    patience: int,
    full_data: bool = False,
) -> None:
    _set_deterministic(seed)
    out.mkdir(parents=True, exist_ok=True)

    print("loading joined rows", flush=True)
    df = _load_joined_frame(joined)
    subject_to_id = _subject_mapping(df)
    if not subject_to_id:
        raise ValueError("no normalized subject keys found in joined data")

    item_id_order = [str(iid) for iid in json.loads((emb / "item_id_order.json").read_text())]
    item_to_row = {item_id: idx for idx, item_id in enumerate(item_id_order)}
    item_embeddings = np.load(emb / "item_embeddings.npy").astype(np.float32)
    enc_meta = json.loads((emb / "encoder_meta.json").read_text())
    if item_embeddings.ndim != 2 or item_embeddings.shape[0] != len(item_id_order):
        raise ValueError(f"embedding shape {item_embeddings.shape} inconsistent with item ID order")

    if full_data:
        held_out = []
        train_df = df.copy()
        val_df = train_df.iloc[:0].copy()
    else:
        held_out, train_df, val_df = split_faithful_eval_rows(
            df,
            item_id_order,
            val_frac=val_frac,
            seed=seed,
            max_rows=val_rows,
            per_category=per_category,
        )
    if max_rows > 0 and len(train_df) > max_rows:
        train_df = train_df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
        print(f"sampled train rows to {len(train_df):,}", flush=True)

    train_subject_idx, train_item_idx, train_labels = _arrays_for_rows(train_df, subject_to_id, item_to_row)
    val_subject_idx, val_item_idx, val_labels = _arrays_for_rows(val_df, subject_to_id, item_to_row)
    if len(train_labels) == 0:
        raise ValueError("no train rows have both normalized subject key and cached item embedding")
    if not full_data and len(val_labels) == 0:
        raise ValueError("no validation rows have both normalized subject key and cached item embedding")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    item_embeddings_t = torch.from_numpy(item_embeddings).to(device)
    model = JEIRTModel(
        n_subjects=len(subject_to_id),
        in_dim=int(item_embeddings.shape[1]),
        hidden=hidden,
        dim=dim,
        dropout=dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(
        f"train={len(train_labels):,} val={len(val_labels):,} "
        f"subjects={len(subject_to_id):,} items={len(item_id_order):,} device={device}",
        flush=True,
    )
    best_mll = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    train_mll_trajectory: list[float] = []
    val_mll_trajectory: list[float] = []
    val_auc_trajectory: list[float] = []
    for epoch in range(epochs):
        train_mll = _train_epoch(
            model,
            opt,
            train_subject_idx,
            train_item_idx,
            train_labels,
            item_embeddings_t,
            batch,
            seed + epoch,
            device,
        )
        train_mll_trajectory.append(_finite_float(train_mll))
        if full_data:
            print(f"epoch={epoch + 1} train_mll={train_mll:.6f}", flush=True)
            continue
        metrics = _evaluate(model, val_subject_idx, val_item_idx, val_labels, item_embeddings_t, batch, device)
        val_mll_trajectory.append(_finite_float(metrics["mean_log_likelihood"]))
        val_auc_trajectory.append(_finite_float(metrics["auc_roc"]))
        print(
            f"epoch={epoch + 1} train_mll={train_mll:.6f} "
            f"val_mll={metrics['mean_log_likelihood']:.6f} val_auc={metrics['auc_roc']:.6f}",
            flush=True,
        )
        if metrics["mean_log_likelihood"] > best_mll + 1e-7:
            best_mll = float(metrics["mean_log_likelihood"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early_stop epoch={epoch + 1} best_val_mll={best_mll:.6f}", flush=True)
                break

    if not full_data and best_state is not None:
        model.load_state_dict(best_state)
    if full_data:
        final_metrics = {"mean_log_likelihood": float("nan"), "auc_roc": float("nan")}
    else:
        final_metrics = _evaluate(model, val_subject_idx, val_item_idx, val_labels, item_embeddings_t, batch, device)

    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / "je_irt_head.pt")
    (out / "subject_to_id.json").write_text(json.dumps(subject_to_id, indent=2, sort_keys=True) + "\n")
    config: dict[str, Any] = {
        "model": "je_irt",
        "formula": "(E_M dot E_Q) / ||E_Q|| - ||E_Q||",
        "subject_key": "normalize_subject(subject_content)",
        "encoder": enc_meta.get("encoder"),
        "encoder_dim": int(item_embeddings.shape[1]),
        "embedding_representation_version": enc_meta.get("representation_version"),
        "max_chars": int(enc_meta.get("max_chars", 4000)),
        "hidden": int(hidden),
        "dim": int(dim),
        "dropout": float(dropout),
        "batch": int(batch),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "epochs_requested": int(epochs),
        "epochs_ran": int(len(train_mll_trajectory)),
        "patience": int(patience),
        "val_frac": float(val_frac),
        "seed": int(seed),
        "split": "item-cold-category-balanced",
        "val_max_rows": int(val_rows),
        "per_category": int(per_category),
        "max_train_rows": int(max_rows),
        "n_subjects": int(len(subject_to_id)),
        "n_items": int(len(item_id_order)),
        "n_train_rows": int(len(train_labels)),
        "n_val_rows": int(len(val_labels)),
        "n_held_out_items": int(len(held_out)),
        "full_data": bool(full_data),
        "val_mll": (None if full_data else _finite_float(final_metrics["mean_log_likelihood"])),
        "val_auc": (None if full_data else _finite_float(final_metrics["auc_roc"])),
        "train_mll_trajectory": train_mll_trajectory,
        "val_mll_trajectory": val_mll_trajectory,
        "val_auc_trajectory": val_auc_trajectory,
        "git_sha": _git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(f"wrote JE-IRT artifacts to {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--joined", default="data/joined.parquet", type=Path)
    parser.add_argument("--emb", default="data/embeddings/mpnet_v1", type=Path)
    parser.add_argument("--out", default="data/stage2/je_irt_v1", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--val-rows", type=int, default=1500)
    parser.add_argument("--per-category", type=int, default=300)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--full-data", action="store_true")
    args = parser.parse_args()
    main(
        args.joined,
        args.emb,
        args.out,
        args.epochs,
        args.lr,
        args.val_frac,
        args.seed,
        args.hidden,
        args.dim,
        args.batch,
        args.weight_decay,
        args.dropout,
        args.max_rows,
        args.val_rows,
        args.per_category,
        args.patience,
        args.full_data,
    )
