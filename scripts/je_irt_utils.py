"""JE-IRT model helpers shared by local training and proxy evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class JEIRTModel(nn.Module):
    def __init__(
        self,
        n_subjects: int,
        in_dim: int = 768,
        hidden: int = 256,
        dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.subject_embedding = nn.Embedding(n_subjects, dim)

    def forward(self, subject_idx: torch.Tensor, item_embeddings: torch.Tensor) -> torch.Tensor:
        item_q = self.adapter(item_embeddings)
        subject_m = self.subject_embedding(subject_idx)
        return je_irt_logits(subject_m, item_q)


def je_irt_logits(subject_m: torch.Tensor, item_q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    q_norm = torch.linalg.vector_norm(item_q, dim=-1).clamp_min(eps)
    return (subject_m * item_q).sum(dim=-1) / q_norm - q_norm


def torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_je_irt_artifacts(path: Path, device: str = "cpu") -> tuple[JEIRTModel, dict[str, int], dict[str, Any]]:
    config = json.loads((path / "config.json").read_text())
    subject_to_id = {str(k): int(v) for k, v in json.loads((path / "subject_to_id.json").read_text()).items()}
    model = JEIRTModel(
        n_subjects=len(subject_to_id),
        in_dim=int(config.get("encoder_dim", config.get("in_dim", 768))),
        hidden=int(config.get("hidden", 256)),
        dim=int(config.get("dim", 256)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(torch_load_cpu(path / "je_irt_head.pt"))
    model.eval()
    return model, subject_to_id, config


def predict_je_irt_probs(
    model: JEIRTModel,
    subject_to_id: dict[str, int],
    item_embeddings: np.ndarray,
    item_to_row: dict[str, int],
    df,
    prior_probs: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    subject_idx = df["subject_key"].map(subject_to_id).fillna(-1).to_numpy(dtype=np.int64)
    item_idx = df["item_id"].map(item_to_row).fillna(-1).to_numpy(dtype=np.int64)
    fallback = (subject_idx < 0) | (item_idx < 0)
    probs = np.asarray(prior_probs, dtype=np.float64).copy()
    keep = np.flatnonzero(~fallback)
    if keep.size == 0:
        return np.clip(probs, 1e-6, 1.0 - 1e-6)

    embeddings_t = torch.from_numpy(item_embeddings).to(device)
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(keep), batch_size):
            rows = keep[start : start + batch_size]
            s = torch.from_numpy(subject_idx[rows]).to(device)
            i = torch.from_numpy(item_idx[rows]).to(device)
            logits = model(s, embeddings_t[i])
            out.append(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64))
    probs[keep] = np.concatenate(out, axis=0)
    return np.clip(probs, 1e-6, 1.0 - 1e-6)
