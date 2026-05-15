"""Shared helpers for K-factor Stage 1/Stage 2 scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def sigmoid_logits(logits: torch.Tensor) -> torch.Tensor:
    """Numerically stable sigmoid for Torch logits."""
    return torch.sigmoid(logits)


def score_kfactor_logits(
    subject_bias: torch.Tensor,
    subject_u: torch.Tensor,
    item_v: torch.Tensor,
    item_z: torch.Tensor,
) -> torch.Tensor:
    """Return K-factor logits: bias_s + U_s dot V_i + z_i.

    Inputs may be scalar, single-row, or batched tensors as long as Torch
    broadcasting makes the final dimensions compatible.
    """
    interaction = (subject_u * item_v).sum(dim=-1)
    return subject_bias + interaction + item_z


def hash_to_unit(value: str, seed: int) -> float:
    """Deterministic [0, 1) hash used for item-cold validation splits."""
    import hashlib

    digest = hashlib.sha256(f"{seed}|{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def validation_item_ids(item_ids: list[str], val_frac: float, seed: int) -> set[str]:
    """Select deterministic held-out item IDs, keeping train/val non-empty when possible."""
    if not 0.0 <= val_frac < 1.0:
        raise ValueError(f"val_frac must be in [0, 1), got {val_frac}")
    ordered = list(dict.fromkeys(str(iid) for iid in item_ids))
    held = {iid for iid in ordered if hash_to_unit(iid, seed) < val_frac}
    if val_frac > 0.0 and not held and len(ordered) > 1:
        held.add(min(ordered, key=lambda iid: hash_to_unit(iid, seed)))
    if len(held) == len(ordered) and len(ordered) > 1:
        held.remove(max(held, key=lambda iid: hash_to_unit(iid, seed)))
    return held


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_subject_state(path: str | Path) -> dict:
    """Load Stage 1 subject state from a .pt file."""
    state = _torch_load(Path(path))
    if not isinstance(state, dict):
        raise TypeError(f"subject state must be a dict, got {type(state).__name__}")
    return state


def load_item_targets(path: str | Path) -> dict:
    """Load Stage 1 item target tensors from a .pt file."""
    targets = _torch_load(Path(path))
    if not isinstance(targets, dict):
        raise TypeError(f"item targets must be a dict, got {type(targets).__name__}")
    return targets
