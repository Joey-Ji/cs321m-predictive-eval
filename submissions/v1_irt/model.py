"""v1 submission: amortized IRT + content head.

Pipeline (mirrors Truong et al. 2025, arXiv:2503.13335):
  Stage 1: per-subject ability theta_s fit offline by Rasch IRT.
  Stage 2a: predicted item difficulty b_hat from item text via frozen encoder + head.
  Stage 2b: theta_s lookup for known subjects (normalized name from subject_content).
  Stage 3: P(correct) = sigmoid(exp(log_a) * (theta_s - b_hat)).
           (log_a = 0 for 1PL/Rasch; log_a from Stage 1 for 2PL.)

Module-level globals load the encoder, head, and IRT state once when the
container starts. predict() runs encoder + head + sigmoid per call.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

EPS = 1e-6
CLIP_LO, CLIP_HI = 0.02, 0.98
NAME_LINE = re.compile(r"^\s*Name:\s*(.+?)\s*$", re.MULTILINE)


def _normalize_subject(subject_content: str) -> str:
    if not subject_content:
        return ""
    m = NAME_LINE.search(subject_content)
    return m.group(1).strip().lower() if m else subject_content.strip().lower()


# ---------------------------------------------------------------------------
# Module-level init: runs once when the container starts.
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
META = json.loads((ROOT / "head_meta.json").read_text())
FEATURE_TEXT_VERSION = META.get("feature_text_version", "item_content_v1")
if FEATURE_TEXT_VERSION != "item_content_v1":
    raise ValueError(
        f"v1_irt encodes raw item_content at runtime, but head_meta.json has "
        f"feature_text_version={FEATURE_TEXT_VERSION!r}. Rebuild the IRT head with raw item_content embeddings."
    )
ENCODER_REPO = META["encoder"]
HEAD_TYPE = META["head_type"]
HIDDEN = int(META.get("hidden", 256))
IN_DIM = int(META["in_dim"])

print(f"[v1_irt] loading encoder {ENCODER_REPO} ...", flush=True)
from sentence_transformers import SentenceTransformer  # noqa: E402

ENCODER = SentenceTransformer(ENCODER_REPO)
ENCODER.eval()

if HEAD_TYPE == "linear":
    HEAD = nn.Linear(IN_DIM, 1)
else:
    HEAD = nn.Sequential(
        nn.Linear(IN_DIM, HIDDEN),
        nn.ReLU(),
        nn.Dropout(0.0),  # disable dropout at inference
        nn.Linear(HIDDEN, 1),
    )
HEAD.load_state_dict(torch.load(ROOT / "head.pt", map_location="cpu"))
HEAD.eval()

THETA = torch.load(ROOT / "theta.pt", map_location="cpu")          # [n_subjects]
LOG_A_GLOBAL = torch.load(ROOT / "log_a.pt", map_location="cpu") if (ROOT / "log_a.pt").exists() else None
SUBJECT_TO_ID = json.loads((ROOT / "subject_to_id.json").read_text())

# Population-mean fallback for unknown subjects
THETA_FALLBACK = float(THETA.mean())

# Optional: a global log_a scalar if 2PL was fit but we don't have item-level a's at test time
# For 1PL, log_a is zeros; for 2PL we'd use a separately predicted a_hat (Stage 2a extension).
GLOBAL_LOG_A = 0.0
if LOG_A_GLOBAL is not None:
    GLOBAL_LOG_A = float(LOG_A_GLOBAL.mean())  # crude; refine if 2PL

print(f"[v1_irt] loaded. n_subjects={len(SUBJECT_TO_ID)}  in_dim={IN_DIM}  head={HEAD_TYPE}", flush=True)


# ---------------------------------------------------------------------------
# predict() — runs once per test input
# ---------------------------------------------------------------------------


def _theta_for(subject_content: str) -> float:
    key = _normalize_subject(subject_content)
    sid = SUBJECT_TO_ID.get(key)
    if sid is None:
        return THETA_FALLBACK
    return float(THETA[sid])


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """Return probability that the subject answers the item correctly.

    Args:
        input: dict with keys benchmark, condition, subject_content, item_content.
        labeled: optional list of dicts with the same keys plus "label" (0/1).
                 Used in later submissions for online calibration; ignored in v1.
    """
    theta_s = _theta_for(input["subject_content"])
    item_text = (input.get("item_content") or "")[:4000]

    with torch.no_grad():
        emb = ENCODER.encode(item_text, convert_to_tensor=True)
        b_hat = float(HEAD(emb.unsqueeze(0)).item())

    a = math.exp(GLOBAL_LOG_A)
    p = 1.0 / (1.0 + math.exp(-a * (theta_s - b_hat)))
    return float(max(min(p, CLIP_HI), CLIP_LO))
