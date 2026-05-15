"""Evaluate K-factor Stage 2 item-parameter predictions on held-out responses."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.kfactor import load_subject_state, score_kfactor_logits, sigmoid_logits, validation_item_ids
from src.validation import auc_roc, mean_log_likelihood

NAME_LINE = re.compile(r"^\s*Name:\s*(.+?)\s*$", re.MULTILINE)


def normalize_subject(subject_content: str) -> str:
    if not subject_content:
        return ""
    m = NAME_LINE.search(subject_content)
    return m.group(1).strip().lower() if m else subject_content.strip().lower()


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


def _torch_load(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _finite_or_none(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite_or_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite_or_none(v) for v in value]
    return value


def _predict_item_params(stage2_dir: Path, emb_dir: Path, item_ids: set[str]):
    import numpy as np
    import torch

    head_meta = json.loads((stage2_dir / "head_meta.json").read_text())
    scaler = json.loads((stage2_dir / "target_scaler.json").read_text())
    item_id_order = [str(iid) for iid in json.loads((emb_dir / "item_id_order.json").read_text())]
    embeddings = np.load(emb_dir / "item_embeddings.npy").astype(np.float32)
    side_features = np.load(emb_dir / "item_side_features.npy").astype(np.float32)
    if side_features.shape[0] != embeddings.shape[0]:
        raise ValueError(
            f"side features rows {side_features.shape[0]} != embedding rows {embeddings.shape[0]}"
        )
    emb_by_id = {iid: idx for idx, iid in enumerate(item_id_order)}

    missing = sorted(iid for iid in item_ids if iid not in emb_by_id)
    if missing:
        raise ValueError(f"{len(missing)} validation item IDs missing embeddings; first={missing[:3]}")

    head = build_head(
        int(head_meta["in_dim"]),
        int(head_meta["out_dim"]),
        str(head_meta["head_type"]),
        int(head_meta.get("hidden", 256)),
    )
    head.load_state_dict(_torch_load(stage2_dir / "head.pt"))
    head.eval()

    rows = [emb_by_id[iid] for iid in sorted(item_ids)]
    X = torch.from_numpy(
        np.concatenate([embeddings[rows], side_features[rows]], axis=1).astype(np.float32)
    )
    mean = torch.tensor(scaler["mean"], dtype=torch.float32)
    std = torch.tensor(scaler["std"], dtype=torch.float32)
    with torch.no_grad():
        pred_std = head(X)
        pred = pred_std * std + mean
    if not torch.isfinite(pred).all():
        raise ValueError("Stage 2 item parameter predictions contain NaN/inf")

    k = int(head_meta["k"])
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for iid, row in zip(sorted(item_ids), pred):
        out[iid] = (row[:k].clone(), row[k].clone())
    return out, head_meta


def _fit_platt(logits, labels):
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


def main(joined_path: Path, stage1_dir: Path, stage2_dir: Path, emb_dir: Path, split: str, val_frac: float, seed: int) -> None:
    import pyarrow.parquet as pq
    import torch

    if split != "item-cold":
        raise ValueError(f"unsupported split {split!r}; only item-cold is implemented")

    head_meta_path = stage2_dir / "head_meta.json"
    head_meta = json.loads(head_meta_path.read_text())
    split_val_frac = float(head_meta.get("val_frac", val_frac))
    split_seed = int(head_meta.get("seed", seed))

    item_to_id = json.loads((stage1_dir / "item_to_id.json").read_text())
    if "val_item_ids" in head_meta:
        held_out = {str(iid) for iid in head_meta["val_item_ids"]}
    else:
        held_out = validation_item_ids([str(iid) for iid in item_to_id.keys()], split_val_frac, split_seed)
    if not held_out:
        raise ValueError("no held-out item IDs for response evaluation")

    item_params, _ = _predict_item_params(stage2_dir, emb_dir, held_out)
    subject_state = load_subject_state(stage1_dir / "subject_state.pt")
    subject_name_to_id = json.loads((stage1_dir / "subject_name_to_id.json").read_text())
    subject_bias_all = subject_state["subject_bias"].detach().cpu().float()
    subject_u_all = subject_state["subject_u"].detach().cpu().float()
    fallback_bias = subject_state["fallback_bias"]
    fallback_u = subject_state["fallback_u"].detach().cpu().float()
    fallback_bias_t = fallback_bias.detach().cpu().float() if torch.is_tensor(fallback_bias) else torch.tensor(float(fallback_bias))

    table = pq.read_table(joined_path, columns=["item_id", "subject_content", "label"])
    response_rows = table.to_pylist()

    subject_bias_rows = []
    subject_u_rows = []
    item_v_rows = []
    item_z_rows = []
    labels = []
    fallback_subject_count = 0
    for row in response_rows:
        item_id = "" if row.get("item_id") is None else str(row.get("item_id"))
        if item_id not in held_out:
            continue
        subject_key = normalize_subject(row.get("subject_content") or "")
        subject_idx = subject_name_to_id.get(subject_key)
        if subject_idx is None:
            fallback_subject_count += 1
            subject_bias_rows.append(fallback_bias_t)
            subject_u_rows.append(fallback_u)
        else:
            subject_bias_rows.append(subject_bias_all[int(subject_idx)])
            subject_u_rows.append(subject_u_all[int(subject_idx)])
        item_v, item_z = item_params[item_id]
        item_v_rows.append(item_v)
        item_z_rows.append(item_z)
        labels.append(int(row.get("label")))

    if not labels:
        raise ValueError("no validation response rows found for held-out items")

    subject_bias = torch.stack([b.reshape(()) for b in subject_bias_rows])
    subject_u = torch.stack(subject_u_rows)
    item_v = torch.stack(item_v_rows)
    item_z = torch.stack([z.reshape(()) for z in item_z_rows])
    logits = score_kfactor_logits(subject_bias, subject_u, item_v, item_z)
    if not torch.isfinite(logits).all():
        raise ValueError("response logits contain NaN/inf")

    probs = sigmoid_logits(logits).detach().cpu().numpy().astype(float).tolist()
    labels_list = [int(y) for y in labels]
    base_ll = float(mean_log_likelihood(probs, labels_list))
    base_auc = float(auc_roc(probs, labels_list))
    logits_list = logits.detach().cpu().numpy().astype(float).tolist()

    calibration = {
        "fit": False,
        "improved": False,
        "reason": None,
        "base_mean_log_likelihood": base_ll,
    }
    platt = _fit_platt(logits_list, labels_list)
    calibration_path = stage2_dir / "calibration.json"
    if platt is None:
        calibration["reason"] = "requires both positive and negative validation labels"
        if calibration_path.exists():
            calibration_path.unlink()
    else:
        alpha, beta = platt
        calibrated_probs = sigmoid_logits(torch.tensor(logits_list, dtype=torch.float32) * alpha + beta).numpy().tolist()
        calibrated_ll = float(mean_log_likelihood(calibrated_probs, labels_list))
        calibration.update(
            {
                "fit": True,
                "alpha": alpha,
                "beta": beta,
                "calibrated_mean_log_likelihood": calibrated_ll,
                "improved": calibrated_ll > base_ll + 1e-12,
            }
        )
        if calibration["improved"]:
            calibration_path.write_text(json.dumps(_finite_or_none(calibration), indent=2))
        elif calibration_path.exists():
            calibration_path.unlink()

    response_metrics = {
        "split": split,
        "mean_log_likelihood": base_ll,
        "auc_roc": base_auc,
        "n_val_items": int(len(held_out)),
        "n_val_response_rows": int(len(labels_list)),
        "fallback_subject_count": int(fallback_subject_count),
        "p_pos": float(sum(labels_list) / len(labels_list)),
        "calibration": calibration,
    }

    metrics_path = stage2_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics["response_eval"] = response_metrics
    metrics_path.write_text(json.dumps(_finite_or_none(metrics), indent=2))

    print(
        "response_eval "
        f"mll={base_ll:.6f} auc={base_auc:.6f} "
        f"items={len(held_out)} rows={len(labels_list)} fallback_subjects={fallback_subject_count}"
    )
    if calibration.get("fit"):
        print(
            "calibration "
            f"improved={calibration['improved']} "
            f"base_mll={base_ll:.6f} calibrated_mll={calibration['calibrated_mean_log_likelihood']:.6f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--joined", default="data/joined.parquet", type=Path)
    parser.add_argument("--stage1", default="data/stage1/kfactor_k4", type=Path)
    parser.add_argument("--stage2", default="data/stage2/kfactor_mpnet_linear_v1", type=Path)
    parser.add_argument("--emb", default="data/embeddings/mpnet_v1", type=Path)
    parser.add_argument("--split", default="item-cold")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.joined, args.stage1, args.stage2, args.emb, args.split, args.val_frac, args.seed)
