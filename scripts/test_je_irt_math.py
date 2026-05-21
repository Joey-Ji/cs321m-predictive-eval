"""Self-checks for JE-IRT scoring math and optimization behavior."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn

from je_irt_utils import JEIRTModel, je_irt_logits
from lever_l_utils import mean_log_likelihood


def _set_identity_adapter(model: JEIRTModel) -> None:
    first = model.adapter[0]
    second = model.adapter[3]
    with torch.no_grad():
        first.weight.zero_()
        first.bias.zero_()
        second.weight.zero_()
        second.bias.zero_()
        first.weight.copy_(torch.eye(first.weight.shape[0], first.weight.shape[1]))
        second.weight.copy_(torch.eye(second.weight.shape[0], second.weight.shape[1]))


def check_formula() -> None:
    torch.manual_seed(0)
    model = JEIRTModel(n_subjects=10, in_dim=256, hidden=256, dim=256, dropout=0.0)
    _set_identity_adapter(model)
    model.eval()
    with torch.no_grad():
        model.subject_embedding.weight.copy_(torch.randn(10, 256))
    item_embeddings = torch.rand(20, 256) + 0.1
    subject_idx = torch.tensor([0, 3, 9], dtype=torch.long)
    item_idx = torch.tensor([2, 11, 17], dtype=torch.long)
    logits = model(subject_idx, item_embeddings[item_idx])
    item_q = item_embeddings[item_idx]
    subject_m = model.subject_embedding(subject_idx)
    manual = (subject_m * item_q).sum(dim=-1) / torch.linalg.vector_norm(item_q, dim=-1) - torch.linalg.vector_norm(
        item_q, dim=-1
    )
    torch.testing.assert_close(logits, manual, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(logits, je_irt_logits(subject_m, item_q), atol=1e-6, rtol=0.0)


def check_training_improves() -> None:
    torch.manual_seed(1)
    np.random.seed(1)
    n_subjects = 10
    n_items = 20
    dim = 256
    item_embeddings = torch.rand(n_items, dim) * 0.05 + 0.01
    true_subjects = torch.randn(n_subjects, dim) * 1.5
    logits_true = je_irt_logits(
        true_subjects[:, None, :].expand(n_subjects, n_items, dim).reshape(-1, dim),
        item_embeddings[None, :, :].expand(n_subjects, n_items, dim).reshape(-1, dim),
    )
    labels = (torch.sigmoid(logits_true) > 0.5).float()
    subject_idx = torch.arange(n_subjects).repeat_interleave(n_items)
    item_idx = torch.arange(n_items).repeat(n_subjects)

    model = JEIRTModel(n_subjects=n_subjects, in_dim=dim, hidden=dim, dim=dim, dropout=0.0)
    _set_identity_adapter(model)
    for param in model.adapter.parameters():
        param.requires_grad_(False)
    with torch.no_grad():
        model.subject_embedding.weight.zero_()
    opt = torch.optim.SGD(model.subject_embedding.parameters(), lr=0.05)
    loss_fn = nn.BCEWithLogitsLoss()
    mlls: list[float] = []
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        logits = model(subject_idx, item_embeddings[item_idx])
        loss = loss_fn(logits, labels)
        loss.backward()
        opt.step()
        with torch.no_grad():
            probs = torch.sigmoid(model(subject_idx, item_embeddings[item_idx])).numpy()
        mlls.append(mean_log_likelihood(probs.astype(np.float64), labels.numpy().astype(np.int8)))

    smoothed = [float(np.mean(mlls[i : i + 10])) for i in range(0, len(mlls) - 9, 10)]
    for prev, cur in zip(smoothed, smoothed[1:]):
        if cur + 1e-6 < prev:
            raise AssertionError(f"smoothed validation mll decreased: {prev} -> {cur}")
    if smoothed[-1] <= smoothed[0] + 0.05:
        raise AssertionError(f"validation mll did not improve enough: {smoothed[0]} -> {smoothed[-1]}")


if __name__ == "__main__":
    check_formula()
    check_training_improves()
    print("PASS: JE-IRT math checks passed")
