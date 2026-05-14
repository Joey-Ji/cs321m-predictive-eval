"""v1 K-factor submission runtime.

Predicts cold-start item K-factor parameters from text, then combines them
with known Stage 1 subject parameters.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    from src.features import FEATURE_TEXT_VERSION, build_item_feature_text
except ImportError:
    FEATURE_TEXT_VERSION = "benchmark_condition_item_v1"

    def _clean_field(value) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and math.isnan(value):
            return ""
        return str(value)

    def build_item_feature_text(row_or_input: dict, max_chars: int = 4000) -> str:
        benchmark = _clean_field(row_or_input.get("benchmark", ""))
        condition = _clean_field(row_or_input.get("condition", ""))
        item_content = _clean_field(row_or_input.get("item_content", ""))[:max_chars]
        return f"Benchmark: {benchmark}\nCondition: {condition}\nItem:\n{item_content}"


EPS = 1e-6
CLIP_LO, CLIP_HI = EPS, 1.0 - EPS
NAME_LINE = re.compile(r"^\s*Name:\s*(.+?)\s*$", re.MULTILINE)
ROOT = Path(__file__).resolve().parent
DUMMY_ENCODER = os.environ.get("V1_KFACTOR_DUMMY_ENCODER") == "1"


def _repo_root() -> Path:
    return ROOT.parent.parent


def _resolve_file(filename: str, fixture_subdir: str | None = None, required: bool = True) -> Path | None:
    local = ROOT / filename
    if local.exists():
        return local
    if DUMMY_ENCODER and fixture_subdir is not None:
        fixture = _repo_root() / "data" / "fixtures" / "kfactor" / fixture_subdir / filename
        if fixture.exists():
            return fixture
    if required:
        raise FileNotFoundError(f"required runtime file missing: {local}")
    return None


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _normalize_subject(subject_content: str) -> str:
    if not subject_content:
        return ""
    m = NAME_LINE.search(subject_content)
    return m.group(1).strip().lower() if m else subject_content.strip().lower()


def _build_head(in_dim: int, out_dim: int, head_type: str, hidden: int):
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


class _DummyEncoder:
    def __init__(self, dim: int):
        self.dim = dim

    def encode(self, text: str, convert_to_tensor: bool = False, **_: object):
        digest = hashlib.sha256(f"dummy-encoder-v1|{text}".encode()).digest()
        seed = int.from_bytes(digest[:8], "big", signed=False)
        rng = np.random.default_rng(seed)
        emb = rng.normal(0.0, 1.0, size=self.dim).astype(np.float32)
        if convert_to_tensor:
            return torch.from_numpy(emb)
        return emb

    def eval(self) -> None:
        return None


HEAD_META = json.loads(_resolve_file("head_meta.json", "stage2").read_text())
TARGET_SCALER = json.loads(_resolve_file("target_scaler.json", "stage2").read_text())
MAX_CHARS = int(HEAD_META.get("max_chars", 4000))
if HEAD_META.get("feature_text_version") not in (None, FEATURE_TEXT_VERSION):
    raise ValueError(
        f"runtime feature text version {FEATURE_TEXT_VERSION!r} does not match "
        f"head version {HEAD_META.get('feature_text_version')!r}; rebuild the submission with current src/features.py"
    )

IN_DIM = int(HEAD_META["in_dim"])
OUT_DIM = int(HEAD_META["out_dim"])
K = int(HEAD_META["k"])

if DUMMY_ENCODER:
    ENCODER = _DummyEncoder(IN_DIM)
else:
    from sentence_transformers import SentenceTransformer  # noqa: E402

    ENCODER = SentenceTransformer(HEAD_META["encoder"])
ENCODER.eval()

HEAD = _build_head(IN_DIM, OUT_DIM, str(HEAD_META["head_type"]), int(HEAD_META.get("hidden", 256)))
HEAD.load_state_dict(_torch_load(_resolve_file("head.pt", "stage2")))
HEAD.eval()
TARGET_MEAN = torch.tensor(TARGET_SCALER["mean"], dtype=torch.float32)
TARGET_STD = torch.tensor(TARGET_SCALER["std"], dtype=torch.float32)

SUBJECT_STATE = _torch_load(_resolve_file("subject_state.pt", "stage1"))
SUBJECT_NAME_TO_ID = json.loads(_resolve_file("subject_name_to_id.json", "stage1").read_text())
SUBJECT_BIAS = SUBJECT_STATE["subject_bias"].detach().cpu().float()
SUBJECT_U = SUBJECT_STATE["subject_u"].detach().cpu().float()
FALLBACK_BIAS = SUBJECT_STATE["fallback_bias"]
FALLBACK_BIAS = float(FALLBACK_BIAS.detach().cpu()) if torch.is_tensor(FALLBACK_BIAS) else float(FALLBACK_BIAS)
FALLBACK_U = SUBJECT_STATE["fallback_u"].detach().cpu().float()

CALIBRATION_PATH = _resolve_file("calibration.json", "stage2", required=False)
CALIBRATION = json.loads(CALIBRATION_PATH.read_text()) if CALIBRATION_PATH else None


@lru_cache(maxsize=4096)
def _feature_text_cached(benchmark: str, condition: str, item_content: str) -> str:
    return build_item_feature_text(
        {"benchmark": benchmark, "condition": condition, "item_content": item_content},
        max_chars=MAX_CHARS,
    )


@lru_cache(maxsize=2048)
def _item_params_from_text(feature_text: str) -> tuple[tuple[float, ...], float]:
    with torch.no_grad():
        emb = ENCODER.encode(feature_text, convert_to_tensor=True).float()
        pred_std = HEAD(emb.reshape(1, -1)).reshape(-1)
        pred = pred_std * TARGET_STD + TARGET_MEAN
    if not torch.isfinite(pred).all():
        raise ValueError("non-finite K-factor item parameter prediction")
    item_v = tuple(float(x) for x in pred[:K])
    item_z = float(pred[K])
    return item_v, item_z


def _subject_params(subject_content: str) -> tuple[float, torch.Tensor]:
    key = _normalize_subject(subject_content)
    subject_idx = SUBJECT_NAME_TO_ID.get(key)
    if subject_idx is None:
        return FALLBACK_BIAS, FALLBACK_U
    idx = int(subject_idx)
    return float(SUBJECT_BIAS[idx]), SUBJECT_U[idx]


def _sigmoid(logit: float) -> float:
    if logit >= 0:
        z = math.exp(-logit)
        return 1.0 / (1.0 + z)
    z = math.exp(logit)
    return z / (1.0 + z)


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """Return probability that the subject answers the item correctly.

    The initial K-factor submission ignores online labeled examples; adaptive
    calibration can be layered on this interface later.
    """
    feature_text = _feature_text_cached(
        str(input.get("benchmark") or ""),
        str(input.get("condition") or ""),
        str(input.get("item_content") or ""),
    )
    item_v_tuple, item_z = _item_params_from_text(feature_text)
    subject_bias, subject_u = _subject_params(str(input.get("subject_content") or ""))
    item_v = torch.tensor(item_v_tuple, dtype=torch.float32)
    logit = subject_bias + float((subject_u * item_v).sum()) + item_z
    if CALIBRATION is not None and CALIBRATION.get("improved"):
        logit = float(CALIBRATION.get("alpha", 1.0)) * logit + float(CALIBRATION.get("beta", 0.0))
    p = _sigmoid(logit)
    return float(max(min(p, CLIP_HI), CLIP_LO))
