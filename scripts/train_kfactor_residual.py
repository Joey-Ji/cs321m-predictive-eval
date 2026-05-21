"""Train the Lever L BCE residual MLP on response rows.

The residual is trained with the split-faithful split: held-out items are chosen
first, validation rows are sampled with the same category-balanced quotas as
modal_eval_submission.py, priors are fitted on non-held-out item rows, and
subject parameters are refit against frozen item predictions using only those
non-held-out rows.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn

from lever_l_utils import (
    PRIOR_COLUMNS,
    auc_roc,
    clean_str,
    coerce_binary_label,
    group_rows_by_category,
    mean_log_likelihood,
    normalize_subject,
    prior_vector_for_frame,
    sample_groups,
    sigmoid,
    subject_category_probs_for_frame,
    tune_priors,
    validation_item_ids,
    write_prior_artifacts,
)
from scripts.train_kfactor_head import build_head
from src.kfactor import load_subject_state
from src.advanced_features import AdvancedFeatureExtractor


@dataclass
class ItemState:
    item_id_order: list[str]
    item_to_row: dict[str, int]
    embeddings: np.ndarray
    item_v: np.ndarray
    item_z: np.ndarray


@dataclass
class SubjectState:
    subject_to_id: dict[str, int]
    subject_bias: np.ndarray
    subject_u: np.ndarray
    fallback_bias: float
    fallback_u: np.ndarray


@dataclass
class RowArrays:
    subject_idx: np.ndarray
    item_idx: np.ndarray
    labels: np.ndarray
    priors: np.ndarray


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        dim = input_dim
        for _ in range(layers):
            blocks.extend([nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout)])
            dim = hidden
        blocks.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_joined_frame(path: Path):
    import pandas as pd

    columns = [
        "subject_id",
        "item_id",
        "subject_content",
        "item_content",
        "benchmark",
        "condition",
        "label",
    ]
    df = pd.read_parquet(path, columns=columns)
    df["label"] = df["label"].map(coerce_binary_label)
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype("int8")
    for col in ("subject_id", "item_id", "subject_content", "benchmark", "condition"):
        df[col] = df[col].map(clean_str)
    df["subject_key"] = df["subject_content"].map(normalize_subject)
    return df.reset_index(drop=True)


def _eval_rows_for_split(df, item_ids: list[str], val_frac: float, seed: int, max_rows: int, per_category: int):
    held_out = validation_item_ids(item_ids, val_frac=val_frac, seed=seed)
    val_all = df[df["item_id"].isin(held_out)]
    rows = val_all[["item_id", "subject_id", "subject_key", "benchmark", "condition", "label"]].to_dict(
        orient="records"
    )
    sampled_groups = sample_groups(
        group_rows_by_category(rows),
        random.Random(seed),
        max_rows=max_rows,
        max_per_category=per_category,
    )
    sampled_rows = [row for category in sorted(sampled_groups) for row in sampled_groups[category]]
    if not sampled_rows:
        raise ValueError("split produced no validation rows")
    sampled_index = {(r["subject_id"], r["item_id"]) for r in sampled_rows}
    val_df = val_all[
        [
            (row.subject_id, row.item_id) in sampled_index
            for row in val_all[["subject_id", "item_id"]].itertuples(index=False)
        ]
    ].copy()
    train_df = df[~df["item_id"].isin(held_out)].copy()
    return held_out, train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def load_item_state(stage2_dir: Path, emb_dir: Path, joined_path: Path | None = None, batch_size: int = 8192) -> ItemState:
    head_meta = json.loads((stage2_dir / "head_meta.json").read_text())
    scaler = json.loads((stage2_dir / "target_scaler.json").read_text())
    item_id_order = [str(iid) for iid in json.loads((emb_dir / "item_id_order.json").read_text())]
    embeddings = np.load(emb_dir / "item_embeddings.npy").astype(np.float32)
    side_features = np.load(emb_dir / "item_side_features.npy").astype(np.float32)
    if embeddings.ndim != 2 or side_features.ndim != 2 or embeddings.shape[0] != side_features.shape[0]:
        raise ValueError(f"bad embedding/side shapes: {embeddings.shape} {side_features.shape}")

    # Check if this model uses advanced features
    uses_advanced = head_meta.get("uses_advanced_features", False)
    advanced_features_array = None

    if uses_advanced:
        if joined_path is None:
            raise ValueError("Model uses advanced features but --joined path not provided")

        # Load advanced feature extractor
        extractor_path = stage2_dir / "advanced_feature_extractor.json"
        if not extractor_path.exists():
            raise FileNotFoundError(f"Advanced feature extractor not found: {extractor_path}")

        feature_extractor = AdvancedFeatureExtractor.load(extractor_path)

        # Load joined data to extract features
        import pandas as pd
        joined_df = pd.read_parquet(joined_path)
        item_to_row = {str(row["item_id"]): row for _, row in joined_df.iterrows() if pd.notna(row.get("item_id"))}

        # Extract advanced features for all items
        advanced_features_list = []
        for item_id in item_id_order:
            joined_row = item_to_row.get(item_id)
            if joined_row is None:
                # Item not in joined data, use zero vector
                advanced_features_list.append(np.zeros(feature_extractor.feature_dim, dtype=np.float32))
            else:
                advanced_features_list.append(feature_extractor.extract(joined_row))

        advanced_features_array = np.stack(advanced_features_list).astype(np.float32)

        # Normalize advanced features using saved statistics
        adv_mean = np.array(head_meta["advanced_feature_mean"], dtype=np.float32)
        adv_std = np.array(head_meta["advanced_feature_std"], dtype=np.float32)
        advanced_features_array = (advanced_features_array - adv_mean) / adv_std

    head = build_head(
        int(head_meta["in_dim"]),
        int(head_meta["out_dim"]),
        str(head_meta["head_type"]),
        int(head_meta.get("hidden", 256)),
    )
    head.load_state_dict(_torch_load(stage2_dir / "head.pt"))
    head.eval()
    mean = torch.tensor(scaler["mean"], dtype=torch.float32)
    std = torch.tensor(scaler["std"], dtype=torch.float32)

    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(item_id_order), batch_size):
            stop = min(start + batch_size, len(item_id_order))

            # Concatenate features: embeddings, side_features, and optionally advanced_features
            if uses_advanced and advanced_features_array is not None:
                x_np = np.concatenate([
                    embeddings[start:stop],
                    side_features[start:stop],
                    advanced_features_array[start:stop]
                ], axis=1).astype(np.float32)
            else:
                x_np = np.concatenate([embeddings[start:stop], side_features[start:stop]], axis=1).astype(np.float32)

            pred = head(torch.from_numpy(x_np)) * std + mean
            preds.append(pred.cpu().numpy().astype(np.float32))
    pred_all = np.concatenate(preds, axis=0)
    k = int(head_meta["k"])
    return ItemState(
        item_id_order=item_id_order,
        item_to_row={iid: idx for idx, iid in enumerate(item_id_order)},
        embeddings=embeddings,
        item_v=pred_all[:, :k].astype(np.float32),
        item_z=pred_all[:, k].astype(np.float32),
    )


def load_runtime_subject_state(stage1_dir: Path) -> SubjectState:
    raw = load_subject_state(stage1_dir / "subject_state.pt")
    subject_to_id = json.loads((stage1_dir / "subject_to_id.json").read_text())
    fallback_bias = raw["fallback_bias"]
    fallback_bias_f = float(fallback_bias.detach().cpu()) if torch.is_tensor(fallback_bias) else float(fallback_bias)
    return SubjectState(
        subject_to_id={str(k): int(v) for k, v in subject_to_id.items()},
        subject_bias=raw["subject_bias"].detach().cpu().numpy().astype(np.float32),
        subject_u=raw["subject_u"].detach().cpu().numpy().astype(np.float32),
        fallback_bias=fallback_bias_f,
        fallback_u=raw["fallback_u"].detach().cpu().numpy().astype(np.float32),
    )


def refit_subject_state(
    train_df,
    base_subjects: SubjectState,
    item_state: ItemState,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
) -> SubjectState:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_subjects = len(base_subjects.subject_to_id)
    k = item_state.item_v.shape[1]

    subject_idx = train_df["subject_id"].map(base_subjects.subject_to_id).fillna(-1).to_numpy(dtype=np.int64)
    item_idx = train_df["item_id"].map(item_state.item_to_row).fillna(-1).to_numpy(dtype=np.int64)
    labels = train_df["label"].to_numpy(dtype=np.float32)
    keep = (subject_idx >= 0) & (item_idx >= 0)
    subject_idx = subject_idx[keep]
    item_idx = item_idx[keep]
    labels = labels[keep]
    if len(labels) == 0:
        raise ValueError("no rows left for subject refit")

    global_rate = float(labels.mean())
    counts = np.bincount(subject_idx, minlength=n_subjects).astype(np.float64)
    sums = np.bincount(subject_idx, weights=labels, minlength=n_subjects).astype(np.float64)
    rates = (sums + 20.0 * global_rate) / (counts + 20.0)
    bias_init = np.asarray(np.log(np.clip(rates, 1e-5, 1.0 - 1e-5) / np.clip(1.0 - rates, 1e-5, 1.0)), dtype=np.float32)
    u_init = rng.normal(0.0, 0.02, size=(n_subjects, k)).astype(np.float32)

    subject_bias = nn.Embedding(n_subjects, 1).to(device)
    subject_u = nn.Embedding(n_subjects, k).to(device)
    with torch.no_grad():
        subject_bias.weight.copy_(torch.from_numpy(bias_init).reshape(-1, 1).to(device))
        subject_u.weight.copy_(torch.from_numpy(u_init).to(device))
    opt = torch.optim.AdamW(list(subject_bias.parameters()) + list(subject_u.parameters()), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    item_v = torch.from_numpy(item_state.item_v).to(device)
    item_z = torch.from_numpy(item_state.item_z).to(device)
    s_all = torch.from_numpy(subject_idx)
    i_all = torch.from_numpy(item_idx)
    y_all = torch.from_numpy(labels)
    for epoch in range(epochs):
        order = torch.randperm(len(y_all), generator=torch.Generator().manual_seed(seed + epoch))
        total_loss = 0.0
        total_n = 0
        subject_bias.train()
        subject_u.train()
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            s = s_all[idx].to(device)
            i = i_all[idx].to(device)
            y = y_all[idx].to(device)
            opt.zero_grad(set_to_none=True)
            logits = subject_bias(s).squeeze(-1) + (subject_u(s) * item_v[i]).sum(dim=1) + item_z[i]
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach()) * len(y)
            total_n += len(y)
        print(f"subject_refit epoch={epoch + 1} loss={total_loss / total_n:.6f}", flush=True)

    return SubjectState(
        subject_to_id=base_subjects.subject_to_id,
        subject_bias=subject_bias.weight.detach().cpu().numpy().reshape(-1).astype(np.float32),
        subject_u=subject_u.weight.detach().cpu().numpy().astype(np.float32),
        fallback_bias=float(bias_init.mean()),
        fallback_u=subject_u.weight.detach().cpu().mean(dim=0).numpy().astype(np.float32),
    )


def rows_to_arrays(df, subjects: SubjectState, item_state: ItemState, priors) -> RowArrays:
    subject_idx = df["subject_id"].map(subjects.subject_to_id).fillna(-1).to_numpy(dtype=np.int64)
    item_idx = df["item_id"].map(item_state.item_to_row).fillna(-1).to_numpy(dtype=np.int64)
    labels = df["label"].to_numpy(dtype=np.float32)
    prior_values = prior_vector_for_frame(df, priors)
    keep = (subject_idx >= 0) & (item_idx >= 0)
    if not keep.all():
        subject_idx = subject_idx[keep]
        item_idx = item_idx[keep]
        labels = labels[keep]
        prior_values = prior_values[keep]
    return RowArrays(
        subject_idx=subject_idx.astype(np.int64),
        item_idx=item_idx.astype(np.int64),
        labels=labels.astype(np.float32),
        priors=prior_values.astype(np.float32),
    )


def base_logits(rows: RowArrays, subjects: SubjectState, item_state: ItemState) -> np.ndarray:
    return (
        subjects.subject_bias[rows.subject_idx]
        + (subjects.subject_u[rows.subject_idx] * item_state.item_v[rows.item_idx]).sum(axis=1)
        + item_state.item_z[rows.item_idx]
    ).astype(np.float32)


def _feature_batch(rows: RowArrays, row_ids: np.ndarray, subjects: SubjectState, item_state: ItemState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s_idx = rows.subject_idx[row_ids]
    i_idx = rows.item_idx[row_ids]
    base = (
        subjects.subject_bias[s_idx]
        + (subjects.subject_u[s_idx] * item_state.item_v[i_idx]).sum(axis=1)
        + item_state.item_z[i_idx]
    ).astype(np.float32)
    x = np.concatenate(
        [
            base.reshape(-1, 1),
            subjects.subject_bias[s_idx].reshape(-1, 1),
            subjects.subject_u[s_idx],
            item_state.embeddings[i_idx],
            rows.priors[row_ids],
        ],
        axis=1,
    ).astype(np.float32)
    return x, base, rows.labels[row_ids]


def feature_stats(rows: RowArrays, subjects: SubjectState, item_state: ItemState, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(1 + 1 + item_state.item_v.shape[1] + item_state.embeddings.shape[1] + len(PRIOR_COLUMNS), dtype=np.float64)
    total_sq = np.zeros_like(total)
    n = 0
    row_ids = np.arange(len(rows.labels))
    for start in range(0, len(row_ids), batch_size):
        batch_ids = row_ids[start : start + batch_size]
        x, _, _ = _feature_batch(rows, batch_ids, subjects, item_state)
        total += x.sum(axis=0, dtype=np.float64)
        total_sq += np.square(x, dtype=np.float64).sum(axis=0, dtype=np.float64)
        n += len(x)
    mean = total / max(n, 1)
    var = np.maximum(total_sq / max(n, 1) - mean * mean, 1e-6)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def evaluate_residual(
    model: ResidualMLP | None,
    rows: RowArrays,
    subjects: SubjectState,
    item_state: ItemState,
    feature_mean: np.ndarray | None,
    feature_std: np.ndarray | None,
    batch_size: int,
    device: str,
) -> dict[str, float]:
    labels = rows.labels.astype(np.int8)
    if model is None:
        logits = base_logits(rows, subjects, item_state).astype(np.float64)
    else:
        model.eval()
        logits_parts: list[np.ndarray] = []
        row_ids = np.arange(len(labels))
        with torch.no_grad():
            for start in range(0, len(row_ids), batch_size):
                batch_ids = row_ids[start : start + batch_size]
                x, base, _ = _feature_batch(rows, batch_ids, subjects, item_state)
                x = (x - feature_mean) / feature_std
                residual = model(torch.from_numpy(x).to(device)).detach().cpu().numpy()
                logits_parts.append(base.astype(np.float64) + residual.astype(np.float64))
        logits = np.concatenate(logits_parts)
    probs = sigmoid(logits).astype(np.float64)
    return {
        "mean_log_likelihood": mean_log_likelihood(probs, labels),
        "auc_roc": auc_roc(probs, labels),
        "n": float(len(labels)),
        "p_pos": float(labels.mean()),
    }


def train_residual(
    train_rows: RowArrays,
    val_rows: RowArrays,
    subjects: SubjectState,
    item_state: ItemState,
    out_dir: Path,
    hidden: int,
    layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    mean, std = feature_stats(train_rows, subjects, item_state, batch_size=batch_size)
    model = ResidualMLP(len(mean), hidden=hidden, layers=layers, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    best_mll = -float("inf")
    best_state = None
    stale = 0
    train_ids = np.arange(len(train_rows.labels))
    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        rng.shuffle(train_ids)
        total_loss = 0.0
        total_n = 0
        model.train()
        for start in range(0, len(train_ids), batch_size):
            batch_ids = train_ids[start : start + batch_size]
            x, base, y = _feature_batch(train_rows, batch_ids, subjects, item_state)
            x = (x - mean) / std
            x_t = torch.from_numpy(x).to(device)
            base_t = torch.from_numpy(base).to(device)
            y_t = torch.from_numpy(y).to(device)
            opt.zero_grad(set_to_none=True)
            residual = model(x_t)
            loss = loss_fn(base_t + residual, y_t)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach()) * len(y)
            total_n += len(y)

        metrics = evaluate_residual(model, val_rows, subjects, item_state, mean, std, batch_size, device)
        print(
            f"residual epoch={epoch + 1} train_bce={total_loss / total_n:.6f} "
            f"val_mll={metrics['mean_log_likelihood']:.6f} val_auc={metrics['auc_roc']:.6f}",
            flush=True,
        )
        if metrics["mean_log_likelihood"] > best_mll + 1e-7:
            best_mll = metrics["mean_log_likelihood"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        raise ValueError("residual training did not produce a best state")
    model.load_state_dict(best_state)
    final_metrics = evaluate_residual(model, val_rows, subjects, item_state, mean, std, batch_size, device)
    torch.save(model.state_dict(), out_dir / "residual.pt")
    head = {
        "model_type": "bce_residual_mlp",
        "input_dim": int(len(mean)),
        "hidden": int(hidden),
        "layers": int(layers),
        "dropout": float(dropout),
        "feature_order": [
            "base_logit",
            "subject_bias",
            "subject_u_0",
            "subject_u_1",
            "subject_u_2",
            "subject_u_3",
            "item_embedding_0..767",
            *PRIOR_COLUMNS,
        ],
        "feature_mean": mean.astype(float).tolist(),
        "feature_std": std.astype(float).tolist(),
        "embedding_dim": int(item_state.embeddings.shape[1]),
        "subject_dim": int(item_state.item_v.shape[1]),
        "prior_columns": PRIOR_COLUMNS,
        "seed": int(seed),
        "val_metrics": final_metrics,
    }
    (out_dir / "head.json").write_text(json.dumps(head, indent=2) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2) + "\n")
    return final_metrics


def _finite_metrics(name: str, probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        f"{name}_mll": mean_log_likelihood(probs, labels.astype(np.int8)),
        f"{name}_auc": auc_roc(probs, labels.astype(np.int8)),
    }


def main(
    joined: Path,
    stage1: Path,
    stage2: Path,
    emb: Path,
    out: Path,
    priors_out: Path,
    val_frac: float,
    seed: int,
    max_rows: int,
    per_category: int,
    hidden: int,
    layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    subject_epochs: int,
    subject_lr: float,
    max_train_rows: int | None,
    device: str,
) -> None:
    print("loading joined rows", flush=True)
    df = _load_joined_frame(joined)
    item_ids = [str(iid) for iid in json.loads((emb / "item_id_order.json").read_text())]
    held_out, train_df, val_df = _eval_rows_for_split(
        df, item_ids, val_frac=val_frac, seed=seed, max_rows=max_rows, per_category=per_category
    )
    print(
        f"split seed={seed} held_out_items={len(held_out)} "
        f"train_rows={len(train_df)} val_rows={len(val_df)}",
        flush=True,
    )

    print("tuning train-only priors", flush=True)
    priors, kappas = tune_priors(train_df, val_df)
    prior_probs = subject_category_probs_for_frame(val_df, priors)
    prior_metrics = _finite_metrics("priors", prior_probs, val_df["label"].to_numpy(dtype=np.int8))
    print(f"priors kappas={kappas} val_mll={prior_metrics['priors_mll']:.6f}", flush=True)

    print("exporting full-data priors with tuned kappas", flush=True)
    write_prior_artifacts(df, kappas, priors_out)

    print("loading frozen item/base state", flush=True)
    item_state = load_item_state(stage2, emb)
    runtime_subjects = load_runtime_subject_state(stage1)
    split_subjects = refit_subject_state(
        train_df,
        runtime_subjects,
        item_state,
        epochs=subject_epochs,
        batch_size=batch_size,
        lr=subject_lr,
        seed=seed,
        device=device,
    )

    if max_train_rows is not None and len(train_df) > max_train_rows:
        train_df = train_df.sample(n=max_train_rows, random_state=seed).reset_index(drop=True)
        print(f"sampled residual train rows to {len(train_df)}", flush=True)

    train_rows = rows_to_arrays(train_df, split_subjects, item_state, priors)
    val_rows = rows_to_arrays(val_df, split_subjects, item_state, priors)
    base_probs = sigmoid(base_logits(val_rows, split_subjects, item_state).astype(np.float64))
    base_metrics = _finite_metrics("base", base_probs, val_rows.labels.astype(np.int8))
    print(
        f"base val_mll={base_metrics['base_mll']:.6f} "
        f"base_auc={base_metrics['base_auc']:.6f}",
        flush=True,
    )

    residual_metrics = train_residual(
        train_rows,
        val_rows,
        split_subjects,
        item_state,
        out,
        hidden=hidden,
        layers=layers,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        seed=seed,
        device=device,
    )
    summary = {
        "seed": seed,
        "val_frac": val_frac,
        "max_rows": max_rows,
        "per_category": per_category,
        "n_held_out_items": len(held_out),
        "n_train_rows": len(train_df),
        "n_val_rows": len(val_df),
        "kappas": kappas,
        **prior_metrics,
        **base_metrics,
        "residual_mll": residual_metrics["mean_log_likelihood"],
        "residual_auc": residual_metrics["auc_roc"],
    }
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--joined", default="data/joined.parquet", type=Path)
    parser.add_argument("--stage1", default="data/stage1/kfactor_k4", type=Path)
    parser.add_argument("--stage2", default="data/stage2/kfactor_mpnet_mlp_v1", type=Path)
    parser.add_argument("--emb", default="data/embeddings/mpnet_v1", type=Path)
    parser.add_argument("--out", default="data/stage2/kfactor_mpnet_residual_v1", type=Path)
    parser.add_argument("--priors-out", default="data/stage2/priors_v1", type=Path)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=1500)
    parser.add_argument("--per-category", type=int, default=300)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--subject-epochs", type=int, default=3)
    parser.add_argument("--subject-lr", type=float, default=5e-2)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(
        args.joined,
        args.stage1,
        args.stage2,
        args.emb,
        args.out,
        args.priors_out,
        args.val_frac,
        args.seed,
        args.max_rows,
        args.per_category,
        args.hidden,
        args.layers,
        args.dropout,
        args.epochs,
        args.batch_size,
        args.lr,
        args.patience,
        args.subject_epochs,
        args.subject_lr,
        args.max_train_rows,
        args.device,
    )
