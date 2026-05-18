"""Fine-tune K-factor Stage 2 end-to-end with response-level BCE loss.

This trains the sentence-transformer encoder and K-factor item head directly
against response labels from joined.parquet while keeping Stage 1 subject
parameters frozen.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import EMBEDDING_REPRESENTATION_VERSION, REPRESENTATION_VERSION, encode_side_features
from src.kfactor import load_subject_state, score_kfactor_logits, sigmoid_logits, validation_item_ids
from src.validation import auc_roc, mean_log_likelihood

NAME_LINE = re.compile(r"^\s*Name:\s*(.+?)\s*$", re.MULTILINE)
DEFAULT_SIDE_FEATURE_META = Path("data/stage2/kfactor_mpnet_mlp_v1/side_feature_meta.json")


def _normalize_subject(subject_content: str) -> str:
    """Match submissions/v1_kfactor/model.py without importing the runtime."""
    if not subject_content:
        return ""
    m = NAME_LINE.search(subject_content)
    return m.group(1).strip().lower() if m else subject_content.strip().lower()


def _clean_field(value: Any) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except TypeError:
        pass
    return str(value)


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite_or_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite_or_none(v) for v in value]
    return value


def _build_head(in_dim: int, out_dim: int, head_type: str, hidden: int):
    import torch.nn as nn

    if head_type == "linear":
        return nn.Linear(in_dim, out_dim)
    if head_type == "mlp":
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(hidden, out_dim),
        )
    raise ValueError(f"unsupported head type: {head_type}")


def _fit_platt(logits: list[float], labels: list[int]) -> tuple[float, float] | None:
    import torch

    if len(set(int(y) for y in labels)) < 2:
        return None

    x = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    alpha = torch.nn.Parameter(torch.tensor(1.0))
    beta = torch.nn.Parameter(torch.tensor(0.0))
    opt = torch.optim.LBFGS([alpha, beta], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = loss_fn(alpha * x + beta, y)
        loss.backward()
        return loss

    opt.step(closure)
    a = float(alpha.detach())
    b = float(beta.detach())
    if not (math.isfinite(a) and math.isfinite(b)):
        return None
    return a, b


class ResponseDataset:
    def __init__(
        self,
        columns: dict[str, list[Any]],
        indices: list[int],
        subject_name_to_id: dict[str, int],
        side_feature_meta: dict[str, Any],
        max_chars: int,
    ) -> None:
        self.columns = columns
        self.indices = indices
        self.subject_name_to_id = subject_name_to_id
        self.side_feature_meta = side_feature_meta
        self.max_chars = max_chars
        self._side_cache: dict[tuple[str, str], Any] = {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, offset: int) -> dict[str, Any]:
        idx = self.indices[offset]
        benchmark = _clean_field(self.columns["benchmark"][idx])
        condition = _clean_field(self.columns["condition"][idx])
        side_key = (benchmark, condition)
        side = self._side_cache.get(side_key)
        if side is None:
            side = encode_side_features({"benchmark": benchmark, "condition": condition}, self.side_feature_meta)
            self._side_cache[side_key] = side

        subject_key = _normalize_subject(_clean_field(self.columns["subject_content"][idx]))
        subject_idx = self.subject_name_to_id.get(subject_key, -1)
        label = self.columns["label"][idx]
        return {
            "subject_idx": int(subject_idx),
            "text": _clean_field(self.columns["item_content"][idx])[: self.max_chars],
            "side": side,
            "label": float(label),
        }


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np
    import torch

    return {
        "subject_idx": torch.tensor([row["subject_idx"] for row in batch], dtype=torch.long),
        "texts": [row["text"] for row in batch],
        "side": torch.from_numpy(np.stack([row["side"] for row in batch]).astype("float32")),
        "label": torch.tensor([row["label"] for row in batch], dtype=torch.float32),
    }


def _slice_batch(batch: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    return {
        "subject_idx": batch["subject_idx"][start:end],
        "texts": batch["texts"][start:end],
        "side": batch["side"][start:end],
        "label": batch["label"][start:end],
    }


def _iter_micro_batches(batch: dict[str, Any], micro_batch_size: int):
    n_rows = int(batch["label"].shape[0])
    for start in range(0, n_rows, micro_batch_size):
        yield _slice_batch(batch, start, min(start + micro_batch_size, n_rows))


def _move_features_to_device(features: dict[str, Any], device: str) -> dict[str, Any]:
    import torch

    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in features.items()}


def _select_subject_params(
    subject_idx,
    subject_bias_all,
    subject_u_all,
    fallback_bias,
    fallback_u,
):
    import torch

    known = subject_idx >= 0
    safe_idx = torch.clamp(subject_idx, min=0)
    subject_bias = subject_bias_all.index_select(0, safe_idx)
    subject_u = subject_u_all.index_select(0, safe_idx)
    if not bool(known.all()):
        subject_bias = torch.where(known, subject_bias, fallback_bias.expand_as(subject_bias))
        subject_u = torch.where(known.unsqueeze(1), subject_u, fallback_u.expand_as(subject_u))
    return subject_bias, subject_u


def _forward_logits(
    encoder,
    head,
    batch: dict[str, Any],
    subject_bias_all,
    subject_u_all,
    fallback_bias,
    fallback_u,
    k: int,
    device: str,
):
    import torch

    features = _move_features_to_device(encoder.tokenize(batch["texts"]), device)
    encoder_out = encoder(features)
    emb = encoder_out["sentence_embedding"].float()
    side = batch["side"].to(device)
    pred = head(torch.cat([emb, side], dim=1))
    item_v = pred[:, :k]
    item_z = pred[:, k]
    subject_idx = batch["subject_idx"].to(device)
    subject_bias, subject_u = _select_subject_params(
        subject_idx,
        subject_bias_all,
        subject_u_all,
        fallback_bias,
        fallback_u,
    )
    logits = score_kfactor_logits(subject_bias, subject_u, item_v, item_z)
    if not torch.isfinite(logits).all():
        raise ValueError("non-finite logits during fine-tuning")
    return logits


def _evaluate(
    encoder,
    head,
    loader,
    subject_bias_all,
    subject_u_all,
    fallback_bias,
    fallback_u,
    k: int,
    device: str,
    micro_batch_size: int,
    amp_enabled: bool,
) -> dict[str, Any]:
    import torch

    encoder.eval()
    head.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    total_rows = 0
    logits_all: list[float] = []
    labels_all: list[int] = []
    with torch.no_grad():
        for batch in loader:
            for micro_batch in _iter_micro_batches(batch, micro_batch_size):
                labels = micro_batch["label"].to(device)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    logits = _forward_logits(
                        encoder,
                        head,
                        micro_batch,
                        subject_bias_all,
                        subject_u_all,
                        fallback_bias,
                        fallback_u,
                        k,
                        device,
                    )
                    loss = loss_fn(logits, labels)
                total_loss += float(loss.detach().cpu())
                total_rows += int(labels.numel())
                logits_all.extend(float(x) for x in logits.detach().cpu().numpy().tolist())
                labels_all.extend(int(x) for x in labels.detach().cpu().numpy().tolist())

    if total_rows == 0:
        raise ValueError("validation loader produced no rows")
    probs = sigmoid_logits(torch.tensor(logits_all, dtype=torch.float32)).numpy().astype(float).tolist()
    return {
        "bce": total_loss / total_rows,
        "mll": float(mean_log_likelihood(probs, labels_all)),
        "auc": float(auc_roc(probs, labels_all)),
        "logits": logits_all,
        "labels": labels_all,
    }


def _stratified_sample_indices(
    groups: dict[tuple[str, str], list[int]],
    max_train_rows: int,
    seed: int,
) -> list[int]:
    import numpy as np

    total = sum(len(indices) for indices in groups.values())
    if total <= max_train_rows:
        out = [idx for indices in groups.values() for idx in indices]
        rng = np.random.default_rng(seed)
        rng.shuffle(out)
        return out

    rng = np.random.default_rng(seed)
    keys = sorted(groups)
    raw = {key: len(groups[key]) * max_train_rows / total for key in keys}
    quotas = {key: min(len(groups[key]), int(math.floor(raw[key]))) for key in keys}
    assigned = sum(quotas.values())

    # Allocate remaining rows by largest fractional remainder while respecting group capacity.
    remainders = sorted(keys, key=lambda key: (raw[key] - math.floor(raw[key]), len(groups[key])), reverse=True)
    while assigned < max_train_rows:
        progressed = False
        for key in remainders:
            if assigned >= max_train_rows:
                break
            if quotas[key] < len(groups[key]):
                quotas[key] += 1
                assigned += 1
                progressed = True
        if not progressed:
            break

    sampled: list[int] = []
    for key in keys:
        indices = groups[key]
        quota = quotas[key]
        if quota >= len(indices):
            sampled.extend(indices)
        elif quota > 0:
            sampled.extend(int(x) for x in rng.choice(indices, size=quota, replace=False).tolist())
    rng.shuffle(sampled)
    return sampled


def _load_joined_columns(joined_path: Path) -> dict[str, list[Any]]:
    import pyarrow.parquet as pq

    columns = ["item_id", "subject_content", "item_content", "benchmark", "condition", "label"]
    print(f"Reading {joined_path} columns={columns} ...", flush=True)
    table = pq.read_table(joined_path, columns=columns)
    return table.to_pydict()


def _load_or_build_side_feature_meta(path: Path, columns: dict[str, list[Any]]) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())

    from src.features import build_side_feature_vocab

    print(f"WARN: side feature meta not found at {path}; rebuilding from joined rows", file=sys.stderr)
    return build_side_feature_vocab(
        {
            "benchmark": columns["benchmark"][idx],
            "condition": columns["condition"][idx],
        }
        for idx in range(len(columns["label"]))
    )


def _prepare_splits(
    columns: dict[str, list[Any]],
    val_frac: float,
    seed: int,
    max_train_rows: int,
) -> tuple[list[int], list[int], set[str]]:
    n_rows = len(columns["label"])
    item_ids = [_clean_field(value) for value in columns["item_id"]]
    held_out = validation_item_ids(item_ids, val_frac=val_frac, seed=seed)
    if not held_out:
        raise ValueError("validation split selected no held-out items")

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    val_indices: list[int] = []
    for idx in range(n_rows):
        if item_ids[idx] in held_out:
            val_indices.append(idx)
        else:
            groups[(_clean_field(columns["benchmark"][idx]), _clean_field(columns["condition"][idx]))].append(idx)

    if not val_indices:
        raise ValueError("validation split left no validation response rows")
    if not groups:
        raise ValueError("validation split left no training response rows")

    train_indices = _stratified_sample_indices(groups, max_train_rows=max_train_rows, seed=seed)
    train_items = {item_ids[idx] for idx in train_indices}
    val_items = {item_ids[idx] for idx in val_indices}
    if train_items & val_items:
        raise AssertionError("item-cold split violated: train and val share item IDs")
    return train_indices, val_indices, held_out


def main(
    encoder_base: str,
    max_train_rows: int,
    val_frac: float,
    seed: int,
    epochs: int,
    batch_size: int,
    lr_encoder: float,
    lr_head: float,
    hidden: int,
    out_dir: Path,
    stage1_dir: Path,
    joined_path: Path,
    side_feature_meta_path: Path,
    max_chars: int,
    micro_batch_size: int,
) -> None:
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from torch.utils.data import DataLoader

    if max_train_rows <= 0:
        raise ValueError(f"max_train_rows must be positive, got {max_train_rows}")
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if micro_batch_size <= 0:
        raise ValueError(f"micro_batch_size must be positive, got {micro_batch_size}")
    if micro_batch_size > batch_size:
        micro_batch_size = batch_size

    started = time.time()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")

    out_dir.mkdir(parents=True, exist_ok=True)
    columns = _load_joined_columns(joined_path)
    side_feature_meta = _load_or_build_side_feature_meta(side_feature_meta_path, columns)
    subject_name_to_id = json.loads((stage1_dir / "subject_name_to_id.json").read_text())
    subject_name_to_id = {str(key): int(value) for key, value in subject_name_to_id.items()}

    train_indices, val_indices, held_out = _prepare_splits(
        columns,
        val_frac=val_frac,
        seed=seed,
        max_train_rows=max_train_rows,
    )
    print(
        f"Prepared item-cold response split: train={len(train_indices):,} "
        f"val={len(val_indices):,} val_items={len(held_out):,}",
        flush=True,
    )

    subject_state = load_subject_state(stage1_dir / "subject_state.pt")
    subject_bias_cpu = subject_state["subject_bias"].detach().cpu().float().requires_grad_(False)
    subject_u_cpu = subject_state["subject_u"].detach().cpu().float().requires_grad_(False)
    fallback_bias_raw = subject_state["fallback_bias"]
    fallback_bias_cpu = (
        fallback_bias_raw.detach().cpu().float().reshape(())
        if torch.is_tensor(fallback_bias_raw)
        else torch.tensor(float(fallback_bias_raw), dtype=torch.float32)
    ).requires_grad_(False)
    fallback_u_cpu = subject_state["fallback_u"].detach().cpu().float().requires_grad_(False)
    if subject_bias_cpu.requires_grad or subject_u_cpu.requires_grad:
        raise AssertionError("Stage 1 subject params must be frozen")
    print(
        "Stage 1 frozen: "
        f"subject_bias.requires_grad={subject_bias_cpu.requires_grad} "
        f"subject_u.requires_grad={subject_u_cpu.requires_grad}",
        flush=True,
    )

    k = int(subject_u_cpu.shape[1])
    out_dim = k + 1
    side_feature_dim = int(side_feature_meta["side_feature_dim"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading encoder {encoder_base} on {device} ...", flush=True)
    encoder = SentenceTransformer(encoder_base, cache_folder=str(Path("data") / "hf_cache"))
    encoder.to(device)
    embedding_dim = int(encoder.get_sentence_embedding_dimension())
    in_dim = embedding_dim + side_feature_dim
    head = _build_head(in_dim=in_dim, out_dim=out_dim, head_type="mlp", hidden=hidden).to(device)

    subject_bias_all = subject_bias_cpu.to(device)
    subject_u_all = subject_u_cpu.to(device)
    fallback_bias = fallback_bias_cpu.to(device)
    fallback_u = fallback_u_cpu.to(device)

    train_dataset = ResponseDataset(columns, train_indices, subject_name_to_id, side_feature_meta, max_chars=max_chars)
    val_dataset = ResponseDataset(columns, val_indices, subject_name_to_id, side_feature_meta, max_chars=max_chars)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device == "cuda",
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device == "cuda",
        collate_fn=_collate,
    )

    total_steps = epochs * len(train_loader)
    warmup_steps = max(1, int(math.ceil(total_steps * 0.1)))
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder.parameters(), "lr": lr_encoder},
            {"params": head.parameters(), "lr": lr_head},
        ],
        weight_decay=0.01,
    )

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = min(1.0, float(step - warmup_steps) / float(total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    loss_sum_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
    amp_enabled = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    metrics: dict[str, Any] = {
        "train_bce_per_epoch": [],
        "val_bce_per_epoch": [],
        "val_mll_per_epoch": [],
        "val_auc_per_epoch": [],
        "n_train": int(len(train_indices)),
        "n_val": int(len(val_indices)),
        "n_val_items": int(len(held_out)),
        "max_train_rows": int(max_train_rows),
        "val_frac": float(val_frac),
        "seed": int(seed),
        "epochs_requested": int(epochs),
        "batch_size": int(batch_size),
        "micro_batch_size": int(micro_batch_size),
        "effective_batch_size": int(batch_size),
        "lr_encoder": float(lr_encoder),
        "lr_head": float(lr_head),
        "hidden": int(hidden),
        "encoder_base": encoder_base,
        "split": "item-cold",
        "best_epoch": None,
    }
    best_val_mll = -float("inf")
    best_epoch: int | None = None
    best_eval: dict[str, Any] | None = None
    global_step = 0
    stopped_early = False

    print(
        f"Training BCE model: epochs={epochs} batch_size={batch_size} "
        f"micro_batch_size={micro_batch_size} steps={total_steps:,} warmup_steps={warmup_steps:,}",
        flush=True,
    )
    for epoch in range(epochs):
        encoder.train()
        head.train()
        train_loss_sum = 0.0
        train_rows = 0
        epoch_started = time.time()
        for batch_idx, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            batch_rows_total = int(batch["label"].numel())
            batch_loss_sum = 0.0
            for micro_batch in _iter_micro_batches(batch, micro_batch_size):
                labels = micro_batch["label"].to(device)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    logits = _forward_logits(
                        encoder,
                        head,
                        micro_batch,
                        subject_bias_all,
                        subject_u_all,
                        fallback_bias,
                        fallback_u,
                        k,
                        device,
                    )
                    loss_sum = loss_sum_fn(logits, labels)
                    loss = loss_sum / batch_rows_total
                scaler.scale(loss).backward()
                batch_loss_sum += float(loss_sum.detach().cpu())
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            train_loss_sum += batch_loss_sum
            train_rows += batch_rows_total
            if batch_idx == 1 or batch_idx % 500 == 0:
                print(
                    f"  epoch {epoch + 1}/{epochs} batch {batch_idx:,}/{len(train_loader):,} "
                    f"train_bce={train_loss_sum / train_rows:.5f}",
                    flush=True,
                )

        train_bce = train_loss_sum / max(1, train_rows)
        val_eval = _evaluate(
            encoder,
            head,
            val_loader,
            subject_bias_all,
            subject_u_all,
            fallback_bias,
            fallback_u,
            k,
            device,
            micro_batch_size,
            amp_enabled,
        )
        metrics["train_bce_per_epoch"].append(float(train_bce))
        metrics["val_bce_per_epoch"].append(float(val_eval["bce"]))
        metrics["val_mll_per_epoch"].append(float(val_eval["mll"]))
        metrics["val_auc_per_epoch"].append(float(val_eval["auc"]))
        print(
            f"epoch {epoch + 1}/{epochs} "
            f"train_bce={train_bce:.6f} val_bce={val_eval['bce']:.6f} "
            f"val_mll={val_eval['mll']:.6f} val_auc={val_eval['auc']:.6f} "
            f"elapsed_min={(time.time() - epoch_started) / 60.0:.1f}",
            flush=True,
        )

        if float(val_eval["mll"]) > best_val_mll + 1e-12:
            best_val_mll = float(val_eval["mll"])
            best_epoch = epoch + 1
            best_eval = val_eval
            encoder_dir = out_dir / "encoder"
            if encoder_dir.exists():
                shutil.rmtree(encoder_dir)
            encoder.save(str(encoder_dir))
            torch.save(head.state_dict(), out_dir / "head.pt")
            print(f"  saved new best epoch {best_epoch} to {out_dir}", flush=True)
        else:
            stopped_early = True
            print(
                f"  early stop: val_mll {float(val_eval['mll']):.6f} "
                f"did not improve best {best_val_mll:.6f}",
                flush=True,
            )
            break

    if best_epoch is None or best_eval is None:
        raise RuntimeError("no best epoch was saved")

    target_order = [f"v_{i}" for i in range(k)] + ["z"]
    head_meta = {
        "head_type": "mlp",
        "hidden": int(hidden),
        "in_dim": int(in_dim),
        "embedding_dim": int(embedding_dim),
        "side_feature_dim": int(side_feature_dim),
        "out_dim": int(out_dim),
        "k": int(k),
        "target_order": target_order,
        "encoder": "local:./encoder",
        "encoder_dim": int(embedding_dim),
        "encoder_base": encoder_base,
        "representation_version": REPRESENTATION_VERSION,
        "embedding_representation_version": EMBEDDING_REPRESENTATION_VERSION,
        "max_chars": int(max_chars),
        "val_frac": float(val_frac),
        "seed": int(seed),
        "split": "item-cold",
        "val_item_ids": sorted(str(iid) for iid in held_out),
        "epochs": int(epochs),
        "best_epoch": int(best_epoch),
        "lr_encoder": float(lr_encoder),
        "lr_head": float(lr_head),
    }
    (out_dir / "head_meta.json").write_text(json.dumps(_finite_or_none(head_meta), indent=2))
    (out_dir / "target_scaler.json").write_text(
        json.dumps(
            {
                "target_order": target_order,
                "mean": [0.0] * out_dim,
                "std": [1.0] * out_dim,
            },
            indent=2,
        )
    )
    if side_feature_meta_path.exists():
        shutil.copyfile(side_feature_meta_path, out_dir / "side_feature_meta.json")
    else:
        (out_dir / "side_feature_meta.json").write_text(json.dumps(side_feature_meta, indent=2))

    calibration: dict[str, Any] = {
        "fit": False,
        "improved": False,
        "reason": None,
        "base_mean_log_likelihood": float(best_eval["mll"]),
    }
    platt = _fit_platt(best_eval["logits"], best_eval["labels"])
    if platt is None:
        calibration["reason"] = "requires both positive and negative validation labels"
    else:
        alpha, beta = platt
        calibrated_probs = sigmoid_logits(
            torch.tensor(best_eval["logits"], dtype=torch.float32) * alpha + beta
        ).numpy().astype(float).tolist()
        calibrated_mll = float(mean_log_likelihood(calibrated_probs, best_eval["labels"]))
        calibration.update(
            {
                "fit": True,
                "alpha": float(alpha),
                "beta": float(beta),
                "calibrated_mean_log_likelihood": calibrated_mll,
                "improved": calibrated_mll > float(best_eval["mll"]) + 1e-12,
            }
        )
    (out_dir / "calibration.json").write_text(json.dumps(_finite_or_none(calibration), indent=2))

    wall_clock_seconds = time.time() - started
    metrics.update(
        {
            "best_epoch": int(best_epoch),
            "best_val_mll": float(best_val_mll),
            "best_val_auc": float(best_eval["auc"]),
            "global_steps": int(global_step),
            "stopped_early": bool(stopped_early),
            "wall_clock_seconds": float(wall_clock_seconds),
            "wall_clock_minutes": float(wall_clock_seconds / 60.0),
            "calibration": calibration,
        }
    )
    (out_dir / "metrics.json").write_text(json.dumps(_finite_or_none(metrics), indent=2))
    print(
        f"Done. best_epoch={best_epoch} best_val_mll={best_val_mll:.6f} "
        f"best_val_auc={float(best_eval['auc']):.6f} wall_clock_min={wall_clock_seconds / 60.0:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-base", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--max-train-rows", type=int, default=1_000_000)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr-encoder", type=float, default=2e-5)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--out", default="data/stage2/kfactor_mpnet_finetuned_v1", type=Path)
    parser.add_argument("--stage1", default="data/stage1/kfactor_k4", type=Path)
    parser.add_argument("--joined", default="data/joined.parquet", type=Path)
    parser.add_argument("--side-feature-meta", default=DEFAULT_SIDE_FEATURE_META, type=Path)
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--micro-batch-size", type=int, default=64)
    args = parser.parse_args()
    main(
        encoder_base=args.encoder_base,
        max_train_rows=args.max_train_rows,
        val_frac=args.val_frac,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_encoder=args.lr_encoder,
        lr_head=args.lr_head,
        hidden=args.hidden,
        out_dir=args.out,
        stage1_dir=args.stage1,
        joined_path=args.joined,
        side_feature_meta_path=args.side_feature_meta,
        max_chars=args.max_chars,
        micro_batch_size=args.micro_batch_size,
    )
