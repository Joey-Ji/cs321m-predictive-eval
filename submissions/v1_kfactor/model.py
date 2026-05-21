"""v1 K-factor submission runtime.

Predicts cold-start item K-factor parameters from text + side features
(benchmark/condition one-hots), then combines with Stage 1 subject state.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import math
import os
import re
import unicodedata
from functools import lru_cache
from numbers import Real
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    from src.features import REPRESENTATION_VERSION, encode_side_features
except ImportError:
    REPRESENTATION_VERSION = "item_text_plus_side_features_v1"

    def _clean_field(value) -> str:
        if value is None:
            return ""
        if isinstance(value, Real) and math.isnan(float(value)):
            return ""
        return str(value)

    def encode_side_features(row: dict, vocab: dict):
        b_dim = int(vocab["benchmark_dim"])
        c_dim = int(vocab["condition_dim"])
        out = np.zeros(b_dim + c_dim, dtype=np.float32)
        bench = _clean_field(row.get("benchmark", ""))
        cond = _clean_field(row.get("condition", ""))
        b_idx = vocab["benchmark"].get(bench)
        c_idx = vocab["condition"].get(cond)
        if b_idx is not None:
            out[int(b_idx)] = 1.0
        if c_idx is not None:
            out[b_dim + int(c_idx)] = 1.0
        return out


CLIP_LO, CLIP_HI = 0.02, 0.98
NAME_LINE = re.compile(
    r"^\s*(?:[-*]\s*|>\s*)?(?:Name|Subject|Model|display_name)\s*[:：]\s*(.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
PIPE_SEP = re.compile(r"\s*\|\s*")
WHITESPACE = re.compile(r"\s+")
BENCHMARK_ALIAS_SEP = re.compile(r"[-_\s]+")
WRAPPING_QUOTES = (('"', '"'), ("'", "'"), ("`", "`"))
SUBJECT_TRAILING_PUNCT = ".,;"
CONDITION_EMPTY_ALIASES = {"", "none", "null", "n/a", "na", "-"}
ROOT = Path(__file__).resolve().parent
DUMMY_ENCODER = os.environ.get("V1_KFACTOR_DUMMY_ENCODER") == "1"
PRIOR_KEY_SEP = "\x1f"
PRIOR_COLUMNS = (
    "prior_global",
    "prior_benchmark",
    "prior_benchmark_condition",
    "prior_subject",
    "prior_subject_benchmark",
    "prior_subject_category",
)
LOOKUP_AUDIT_OUTCOMES = (
    "hit_subject_category",
    "hit_subject_benchmark",
    "hit_subject",
    "hit_benchmark_condition_only",
    "hit_benchmark_only",
    "fell_to_global",
    "prior_none",
)
LOOKUP_AUDIT_SAMPLE_CAP = 500
_LOOKUP_AUDIT_COUNTERS = {outcome: 0 for outcome in LOOKUP_AUDIT_OUTCOMES}
_LOOKUP_AUDIT_SAMPLES: list[dict[str, str]] = []


def _resolve_cache_dir() -> str | None:
    """Find an HF cache dir the runtime can use (mirrors templates/hf_submission)."""
    candidates = [
        os.environ.get("HF_HOME", "").strip(),
        os.environ.get("TRANSFORMERS_CACHE", "").strip(),
        "/app/hf_cache",
        str(ROOT / ".hf_cache"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if os.access(path, os.W_OK):
            os.environ.setdefault("HF_HOME", str(path))
            return str(path)
    return None


HF_CACHE_DIR = _resolve_cache_dir()
print(f"[v1_kfactor] HF cache dir: {HF_CACHE_DIR}", flush=True)


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


def _clean_text(value: object) -> str:
    return unicodedata.normalize("NFKC", str(value)).strip()


def _strip_wrapping_quotes(text: str) -> str:
    out = text.strip()
    for left, right in WRAPPING_QUOTES:
        if len(out) >= 2 and out.startswith(left) and out.endswith(right):
            return out[1:-1].strip()
    return out


def _subject_key_text(value: object, *, strip_trailing_punct: bool = False) -> str:
    text = _strip_wrapping_quotes(_clean_text(value))
    if strip_trailing_punct:
        text = text.rstrip(SUBJECT_TRAILING_PUNCT).strip()
    return text.lower()


def _normalize_subject(subject_content: str) -> str:
    if not subject_content:
        return ""
    text = _strip_wrapping_quotes(_clean_text(subject_content))
    m = NAME_LINE.search(text)
    if m:
        return _subject_key_text(m.group(1), strip_trailing_punct=True)
    return _subject_key_text(text)


def _normalize_benchmark_basic(benchmark: object) -> str:
    return WHITESPACE.sub(" ", _clean_text(benchmark).lower())


def _benchmark_signature(benchmark: object) -> str:
    return BENCHMARK_ALIAS_SEP.sub("", _normalize_benchmark_basic(benchmark))


def _split_prior_key(key: str, expected_parts: int) -> list[str]:
    sep = str(globals().get("PRIOR_KEY_SEP", "\x1f"))
    parts = str(key).split(sep)
    if len(parts) == expected_parts:
        return parts
    return []


@lru_cache(maxsize=1)
def _prior_benchmark_keys() -> set[str]:
    priors = globals().get("PRIORS")
    if not isinstance(priors, dict):
        return set()
    keys = {str(k) for k in priors.get("benchmark", {})}
    for key in priors.get("benchmark_condition", {}):
        parts = _split_prior_key(str(key), 2)
        if parts:
            keys.add(parts[0])
    for key in priors.get("subject_benchmark", {}):
        parts = _split_prior_key(str(key), 2)
        if parts:
            keys.add(parts[1])
    for key in priors.get("subject_category", {}):
        parts = _split_prior_key(str(key), 3)
        if parts:
            keys.add(parts[1])
    return keys


@lru_cache(maxsize=1)
def _prior_condition_keys() -> set[str]:
    priors = globals().get("PRIORS")
    if not isinstance(priors, dict):
        return set()
    keys: set[str] = set()
    for key in priors.get("benchmark_condition", {}):
        parts = _split_prior_key(str(key), 2)
        if parts:
            keys.add(parts[1])
    for key in priors.get("subject_category", {}):
        parts = _split_prior_key(str(key), 3)
        if parts:
            keys.add(parts[2])
    return keys


def _unique_alias_map(keys: set[str], signature_fn) -> dict[str, str]:
    buckets: dict[str, list[str]] = {}
    for key in keys:
        buckets.setdefault(signature_fn(key), []).append(key)
    return {
        signature: values[0]
        for signature, values in buckets.items()
        if len(set(values)) == 1
    }


@lru_cache(maxsize=1)
def _benchmark_alias_map() -> dict[str, str]:
    return _unique_alias_map(_prior_benchmark_keys(), _benchmark_signature)


def _normalize_benchmark_key(benchmark: str) -> str:
    basic = _normalize_benchmark_basic(benchmark)
    keys = _prior_benchmark_keys()
    if basic in keys:
        return basic
    return _benchmark_alias_map().get(_benchmark_signature(basic), basic)


def _normalize_condition_basic(condition: object) -> str:
    text = PIPE_SEP.sub("|", _clean_text(condition))
    return WHITESPACE.sub(" ", text)


def _empty_condition_canonical(keys: set[str]) -> str | None:
    if "none" in keys and "" not in keys:
        return "none"
    if "" in keys and "none" not in keys:
        return ""
    return None


def _condition_signature(condition: object) -> str:
    basic = _normalize_condition_basic(condition)
    lowered = basic.lower()
    if lowered in CONDITION_EMPTY_ALIASES:
        return "none"
    return lowered


@lru_cache(maxsize=1)
def _condition_alias_map() -> dict[str, str]:
    return _unique_alias_map(_prior_condition_keys(), _condition_signature)


def _normalize_condition_key(condition: str) -> str:
    basic = _normalize_condition_basic(condition)
    keys = _prior_condition_keys()
    if basic in keys:
        return basic
    lowered = basic.lower()
    empty_canonical = _empty_condition_canonical(keys)
    if empty_canonical is not None and lowered in CONDITION_EMPTY_ALIASES:
        return empty_canonical
    return _condition_alias_map().get(_condition_signature(basic), basic)


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


def _build_residual(input_dim: int, hidden: int, layers: int, dropout: float):
    blocks: list[nn.Module] = []
    dim = input_dim
    for _ in range(layers):
        blocks.extend([nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout)])
        dim = hidden
    blocks.append(nn.Linear(dim, 1))
    return nn.Sequential(*blocks)


class _JEIRTModel(nn.Module):
    def __init__(self, n_subjects: int, in_dim: int, hidden: int, dim: int, dropout: float) -> None:
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.subject_embedding = nn.Embedding(n_subjects, dim)


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
VOCAB = json.loads(_resolve_file("side_feature_meta.json", "stage2").read_text())
MAX_CHARS = int(HEAD_META.get("max_chars", 4000))
if HEAD_META.get("representation_version") != REPRESENTATION_VERSION:
    raise ValueError(
        f"runtime representation {REPRESENTATION_VERSION!r} does not match "
        f"head representation {HEAD_META.get('representation_version')!r}; "
        "rebuild the submission with current src/features.py"
    )

IN_DIM = int(HEAD_META["in_dim"])
OUT_DIM = int(HEAD_META["out_dim"])
K = int(HEAD_META["k"])
EMBEDDING_DIM = int(HEAD_META["embedding_dim"])
SIDE_FEATURE_DIM = int(HEAD_META["side_feature_dim"])
if EMBEDDING_DIM + SIDE_FEATURE_DIM != IN_DIM:
    raise ValueError(
        f"head_meta in_dim {IN_DIM} != embedding_dim {EMBEDDING_DIM} + side_feature_dim {SIDE_FEATURE_DIM}"
    )
if int(VOCAB["side_feature_dim"]) != SIDE_FEATURE_DIM:
    raise ValueError(
        f"side_feature_meta side_feature_dim {VOCAB['side_feature_dim']} != head side_feature_dim {SIDE_FEATURE_DIM}"
    )

ENCODER = None
ENCODER_OK = False
ENCODER_LOAD_ERROR: str | None = None
if DUMMY_ENCODER:
    ENCODER = _DummyEncoder(EMBEDDING_DIM)
    ENCODER_OK = True
else:
    try:
        from sentence_transformers import SentenceTransformer  # noqa: E402

        ENCODER = SentenceTransformer(HEAD_META["encoder"], cache_folder=HF_CACHE_DIR)
        ENCODER_OK = True
    except Exception as exc:  # noqa: BLE001
        ENCODER_LOAD_ERROR = repr(exc)
        print(f"[v1_kfactor] encoder load failed: {ENCODER_LOAD_ERROR}", flush=True)
if ENCODER_OK:
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

PRIORS_PATH = _resolve_file("runtime_priors.json", "priors", required=False)
PRIORS = json.loads(PRIORS_PATH.read_text()) if PRIORS_PATH else None
if PRIORS is not None:
    PRIOR_KEY_SEP = str(PRIORS.get("key_sep", PRIOR_KEY_SEP))
PRIOR_ONLY_PATH = _resolve_file("prior_only.json", "priors", required=False)
PRIOR_ONLY = PRIOR_ONLY_PATH is not None

# Submission ZIP entries preserve je_irt/ under ROOT; keep this nested lookup.
JE_IRT_HEAD_PATH = _resolve_file("je_irt/je_irt_head.pt", "je_irt", required=False)
JE_IRT_CONFIG_PATH = _resolve_file("je_irt/config.json", "je_irt", required=False)
JE_IRT_SUBJECT_PATH = _resolve_file("je_irt/subject_to_id.json", "je_irt", required=False)
JE_IRT = None
JE_IRT_ENCODER = None
JE_IRT_CONFIG = None
JE_IRT_SUBJECT_TO_ID: dict[str, int] = {}
JE_IRT_MAX_CHARS = MAX_CHARS
JE_IRT_ACTIVE = False
if JE_IRT_HEAD_PATH is not None and not PRIOR_ONLY:
    try:
        if JE_IRT_CONFIG_PATH is None or JE_IRT_SUBJECT_PATH is None:
            raise FileNotFoundError("JE-IRT config.json and subject_to_id.json are required with je_irt_head.pt")
        JE_IRT_CONFIG = json.loads(JE_IRT_CONFIG_PATH.read_text())
        JE_IRT_SUBJECT_TO_ID = {
            str(k): int(v)
            for k, v in json.loads(JE_IRT_SUBJECT_PATH.read_text()).items()
        }
        JE_IRT_MAX_CHARS = int(JE_IRT_CONFIG.get("max_chars", MAX_CHARS))
        JE_IRT = _JEIRTModel(
            len(JE_IRT_SUBJECT_TO_ID),
            int(JE_IRT_CONFIG.get("encoder_dim", EMBEDDING_DIM)),
            int(JE_IRT_CONFIG.get("hidden", 256)),
            int(JE_IRT_CONFIG.get("dim", 256)),
            float(JE_IRT_CONFIG.get("dropout", 0.1)),
        )
        JE_IRT.load_state_dict(_torch_load(JE_IRT_HEAD_PATH))
        JE_IRT.eval()
        je_encoder_name = str(JE_IRT_CONFIG.get("encoder", HEAD_META["encoder"]))
        if DUMMY_ENCODER:
            JE_IRT_ENCODER = _DummyEncoder(int(JE_IRT_CONFIG.get("encoder_dim", EMBEDDING_DIM)))
        elif ENCODER_OK and je_encoder_name == str(HEAD_META["encoder"]):
            JE_IRT_ENCODER = ENCODER
        else:
            from sentence_transformers import SentenceTransformer  # noqa: E402

            JE_IRT_ENCODER = SentenceTransformer(je_encoder_name, cache_folder=HF_CACHE_DIR)
        JE_IRT_ENCODER.eval()
        JE_IRT_ACTIVE = True
    except Exception as exc:  # noqa: BLE001
        JE_IRT = None
        JE_IRT_ENCODER = None
        JE_IRT_SUBJECT_TO_ID = {}
        JE_IRT_ACTIVE = False
        print(f"[v1_kfactor] JE-IRT load failed: {exc!r}", flush=True)

RESIDUAL_META_PATH = _resolve_file("head.json", "residual", required=False)
RESIDUAL_WEIGHTS_PATH = _resolve_file("residual.pt", "residual", required=False)
RESIDUAL = None
RESIDUAL_FEATURE_MEAN = None
RESIDUAL_FEATURE_STD = None
RESIDUAL_OK = False
if RESIDUAL_META_PATH is not None and RESIDUAL_WEIGHTS_PATH is not None:
    try:
        RESIDUAL_META = json.loads(RESIDUAL_META_PATH.read_text())
        RESIDUAL = _build_residual(
            int(RESIDUAL_META["input_dim"]),
            int(RESIDUAL_META.get("hidden", 256)),
            int(RESIDUAL_META.get("layers", 2)),
            float(RESIDUAL_META.get("dropout", 0.1)),
        )
        residual_state = _torch_load(RESIDUAL_WEIGHTS_PATH)
        if isinstance(residual_state, dict) and any(str(k).startswith("net.") for k in residual_state):
            residual_state = {
                str(k)[4:] if str(k).startswith("net.") else str(k): v
                for k, v in residual_state.items()
            }
        RESIDUAL.load_state_dict(residual_state)
        RESIDUAL.eval()
        RESIDUAL_FEATURE_MEAN = torch.tensor(RESIDUAL_META["feature_mean"], dtype=torch.float32)
        RESIDUAL_FEATURE_STD = torch.tensor(RESIDUAL_META["feature_std"], dtype=torch.float32)
        RESIDUAL_FEATURE_STD = torch.clamp(RESIDUAL_FEATURE_STD, min=1e-6)
        RESIDUAL_OK = PRIORS is not None
        if not RESIDUAL_OK:
            print("[v1_kfactor] residual disabled: runtime_priors.json missing", flush=True)
    except Exception as exc:  # noqa: BLE001
        RESIDUAL = None
        RESIDUAL_FEATURE_MEAN = None
        RESIDUAL_FEATURE_STD = None
        RESIDUAL_OK = False
        print(f"[v1_kfactor] residual load failed: {exc!r}", flush=True)

ONLINE_PLATT_MIN_EXAMPLES = 4
ONLINE_PLATT_L2 = 0.05
ONLINE_PLATT_MIN_ALPHA = 0.05
ONLINE_PLATT_MAX_ALPHA = 3.0
ONLINE_PLATT_MAX_ABS_BETA = 3.0
_ONLINE_PLATT_CACHE_KEY: str | None = None
_ONLINE_PLATT_CACHE: tuple[float, float] | None = None

PER_SUBJECT_SHIFT_KAPPA = 5.0
PER_SUBJECT_SHIFT_LABEL_SMOOTH = 0.5
PER_SUBJECT_SHIFT_MIN_OBS = 1
PER_SUBJECT_SHIFT_CLIP = 1.0

_PER_SUBJECT_CACHE_KEY: str | None = None
_PER_SUBJECT_CACHE: dict[str, float] | None = None


@lru_cache(maxsize=2048)
def _item_context(
    item_content: str, benchmark: str, condition: str
) -> tuple[tuple[float, ...], float, tuple[float, ...]]:
    text = item_content[:MAX_CHARS]
    side = encode_side_features({"benchmark": benchmark, "condition": condition}, VOCAB)
    with torch.no_grad():
        emb = ENCODER.encode(text, convert_to_tensor=True).float().reshape(-1)
        side_t = torch.from_numpy(side).float()
        x = torch.cat([emb, side_t], dim=0).reshape(1, -1)
        pred_std = HEAD(x).reshape(-1)
        pred = pred_std * TARGET_STD + TARGET_MEAN
    if not torch.isfinite(pred).all():
        raise ValueError("non-finite K-factor item parameter prediction")
    item_v = tuple(float(x) for x in pred[:K])
    item_z = float(pred[K])
    emb_tuple = tuple(float(x) for x in emb)
    return item_v, item_z, emb_tuple


def _item_params(item_content: str, benchmark: str, condition: str) -> tuple[tuple[float, ...], float]:
    item_v, item_z, _ = _item_context(item_content, benchmark, condition)
    return item_v, item_z


def _subject_params(subject_content: str) -> tuple[float, torch.Tensor]:
    key = _normalize_subject(subject_content)
    subject_idx = SUBJECT_NAME_TO_ID.get(key)
    if subject_idx is None:
        return FALLBACK_BIAS, FALLBACK_U
    idx = int(subject_idx)
    return float(SUBJECT_BIAS[idx]), SUBJECT_U[idx]


def _prior_key(*parts: str) -> str:
    return PRIOR_KEY_SEP.join(str(part) for part in parts)


def _record_lookup_audit(
    outcome: str,
    raw_subject_content: str | None,
    subject_key: str,
    benchmark: str,
    condition: str,
) -> None:
    _LOOKUP_AUDIT_COUNTERS[outcome] = _LOOKUP_AUDIT_COUNTERS.get(outcome, 0) + 1
    if outcome in ("hit_subject_category", "hit_subject_benchmark"):
        return
    if len(_LOOKUP_AUDIT_SAMPLES) >= LOOKUP_AUDIT_SAMPLE_CAP:
        return
    _LOOKUP_AUDIT_SAMPLES.append(
        {
            "raw_subject_content": "" if raw_subject_content is None else str(raw_subject_content),
            "normalized_subject_key": str(subject_key),
            "benchmark": str(benchmark),
            "condition": str(condition),
            "outcome": outcome,
        }
    )


def _lookup_audit_payload() -> dict[str, object]:
    return {
        "counters": dict(_LOOKUP_AUDIT_COUNTERS),
        "samples": list(_LOOKUP_AUDIT_SAMPLES),
        "sample_cap": LOOKUP_AUDIT_SAMPLE_CAP,
    }


def dump_lookup_audit(path: str | os.PathLike[str]) -> None:
    payload = _lookup_audit_payload()
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        "[v1_kfactor] lookup_audit_json="
        + json.dumps(payload, sort_keys=True, separators=(",", ":")),
        flush=True,
    )


def _prior_values(
    subject_key: str,
    benchmark: str,
    condition: str,
    raw_subject_content: str | None = None,
) -> tuple[float, ...]:
    benchmark_key = _normalize_benchmark_key(benchmark)
    condition_key = _normalize_condition_key(condition)
    if PRIORS is None:
        p = 0.654
        _record_lookup_audit("prior_none", raw_subject_content, subject_key, benchmark, condition)
        return (p, p, p, p, p, p)
    benchmark_table = PRIORS.get("benchmark", {})
    benchmark_condition_table = PRIORS.get("benchmark_condition", {})
    subject_table = PRIORS.get("subject", {})
    subject_benchmark_table = PRIORS.get("subject_benchmark", {})
    subject_category_table = PRIORS.get("subject_category", {})
    bc_key = _prior_key(benchmark_key, condition_key)
    sb_key = _prior_key(subject_key, benchmark_key)
    sc_key = _prior_key(subject_key, benchmark_key, condition_key)

    global_p = float(PRIORS.get("global", 0.654))
    bench_p = float(benchmark_table.get(benchmark_key, global_p))
    bc_p = float(benchmark_condition_table.get(bc_key, bench_p))
    subj_p = float(subject_table.get(subject_key, global_p))
    sb_p = float(subject_benchmark_table.get(sb_key, subj_p))
    sc_p = float(subject_category_table.get(sc_key, sb_p))

    if sc_key in subject_category_table:
        outcome = "hit_subject_category"
    elif sb_key in subject_benchmark_table:
        outcome = "hit_subject_benchmark"
    elif subject_key in subject_table:
        outcome = "hit_subject"
    elif bc_key in benchmark_condition_table:
        outcome = "hit_benchmark_condition_only"
    elif benchmark_key in benchmark_table:
        outcome = "hit_benchmark_only"
    else:
        outcome = "fell_to_global"
    _record_lookup_audit(outcome, raw_subject_content, subject_key, benchmark, condition)
    return global_p, bench_p, bc_p, subj_p, sb_p, sc_p


_LOOKUP_AUDIT_PATH = os.environ.get("V1_KFACTOR_LOOKUP_AUDIT")
if _LOOKUP_AUDIT_PATH:
    atexit.register(dump_lookup_audit, _LOOKUP_AUDIT_PATH)


def _sigmoid(logit: float) -> float:
    if logit >= 0:
        z = math.exp(-logit)
        return 1.0 / (1.0 + z)
    z = math.exp(logit)
    return z / (1.0 + z)


def _offline_calibration_params() -> tuple[float, float]:
    if CALIBRATION is not None and CALIBRATION.get("improved"):
        return float(CALIBRATION.get("alpha", 1.0)), float(CALIBRATION.get("beta", 0.0))
    return 1.0, 0.0


def _apply_calibration(logit: float) -> float:
    alpha, beta = _offline_calibration_params()
    return alpha * logit + beta


def _clip_prob(p: float) -> float:
    if not math.isfinite(p):
        return 0.5
    return float(max(min(p, CLIP_HI), CLIP_LO))


def _base_logit_parts(input: dict) -> tuple[float, float, torch.Tensor, tuple[float, ...], tuple[float, ...]]:
    benchmark = str(input.get("benchmark") or "")
    condition = str(input.get("condition") or "")
    item_v_tuple, item_z, emb_tuple = _item_context(
        str(input.get("item_content") or ""),
        benchmark,
        condition,
    )
    subject_content = str(input.get("subject_content") or "")
    subject_bias, subject_u = _subject_params(subject_content)
    item_v = torch.tensor(item_v_tuple, dtype=torch.float32)
    base_logit = subject_bias + float((subject_u * item_v).sum()) + item_z
    return float(base_logit), float(subject_bias), subject_u, emb_tuple, _prior_values(
        _normalize_subject(subject_content),
        benchmark,
        condition,
        raw_subject_content=subject_content,
    )


def _residual_value(
    base_logit: float,
    subject_bias: float,
    subject_u: torch.Tensor,
    emb_tuple: tuple[float, ...],
    priors: tuple[float, ...],
) -> float:
    if not RESIDUAL_OK or RESIDUAL is None or RESIDUAL_FEATURE_MEAN is None or RESIDUAL_FEATURE_STD is None:
        return 0.0
    with torch.no_grad():
        feature = torch.cat(
            [
                torch.tensor([base_logit, subject_bias], dtype=torch.float32),
                subject_u.detach().cpu().float().reshape(-1),
                torch.tensor(emb_tuple, dtype=torch.float32),
                torch.tensor(priors, dtype=torch.float32),
            ],
            dim=0,
        )
        if feature.numel() != RESIDUAL_FEATURE_MEAN.numel():
            raise ValueError(
                f"residual feature dim {feature.numel()} != expected {RESIDUAL_FEATURE_MEAN.numel()}"
            )
        x = ((feature - RESIDUAL_FEATURE_MEAN) / RESIDUAL_FEATURE_STD).reshape(1, -1)
        residual = RESIDUAL(x).reshape(-1)[0]
    value = float(residual)
    if not math.isfinite(value):
        raise ValueError("non-finite residual prediction")
    return value


def _raw_logit(input: dict) -> float:
    if PRIOR_ONLY:
        benchmark = str(input.get("benchmark") or "")
        condition = str(input.get("condition") or "")
        subject_content = str(input.get("subject_content") or "")
        subject_key = _normalize_subject(subject_content)
        p = _prior_values(subject_key, benchmark, condition, raw_subject_content=subject_content)[-1]
        return _logit_prob(float(p))
    base_logit, subject_bias, subject_u, emb_tuple, priors = _base_logit_parts(input)
    return base_logit + _residual_value(base_logit, subject_bias, subject_u, emb_tuple, priors)


def _prior_probability_for_input(input: dict) -> float:
    benchmark = str(input.get("benchmark") or "")
    condition = str(input.get("condition") or "")
    subject_content = str(input.get("subject_content") or "")
    subject_key = _normalize_subject(subject_content)
    return _clip_prob(_prior_values(subject_key, benchmark, condition, raw_subject_content=subject_content)[-1])


@lru_cache(maxsize=2048)
def _je_irt_item_q(item_content: str) -> tuple[float, ...]:
    if JE_IRT is None or JE_IRT_ENCODER is None:
        raise ValueError("JE-IRT requested but artifacts are not loaded")
    text = item_content[:JE_IRT_MAX_CHARS]
    with torch.no_grad():
        emb = JE_IRT_ENCODER.encode(text, convert_to_tensor=True).float().reshape(1, -1)
        q = JE_IRT.adapter(emb).reshape(-1)
    if not torch.isfinite(q).all():
        raise ValueError("non-finite JE-IRT item embedding")
    return tuple(float(x) for x in q)


def _je_irt_probability(input: dict) -> float:
    if JE_IRT is None:
        return _prior_probability_for_input(input)
    subject_key = _normalize_subject(str(input.get("subject_content") or ""))
    if not subject_key:
        return _prior_probability_for_input(input)
    subject_idx = JE_IRT_SUBJECT_TO_ID.get(subject_key)
    if subject_idx is None:
        return _prior_probability_for_input(input)
    q = torch.tensor(_je_irt_item_q(str(input.get("item_content") or "")), dtype=torch.float32)
    with torch.no_grad():
        subject = JE_IRT.subject_embedding(torch.tensor([int(subject_idx)], dtype=torch.long)).reshape(-1)
        q_norm = torch.linalg.vector_norm(q).clamp_min(1e-8)
        logit = (subject * q).sum() / q_norm - q_norm
    value = float(logit)
    if not math.isfinite(value):
        raise ValueError("non-finite JE-IRT logit")
    return _clip_prob(_sigmoid(value))


def _label_value(row: dict) -> int | None:
    label = row.get("label")
    if isinstance(label, bool):
        return int(label)
    if isinstance(label, Real):
        value = float(label)
        if math.isfinite(value) and value in (0.0, 1.0):
            return int(value)
    return None


def _labeled_cache_key(labeled: list[dict] | None) -> str | None:
    if not labeled:
        return None
    payload = []
    for row in labeled:
        if not isinstance(row, dict):
            continue
        payload.append(
            (
                str(row.get("benchmark") or ""),
                str(row.get("condition") or ""),
                str(row.get("subject_content") or ""),
                str(row.get("item_content") or ""),
                str(row.get("label")),
            )
        )
    if not payload:
        return None
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8", errors="surrogatepass"
    )
    return hashlib.sha256(encoded).hexdigest()


def _fit_online_platt(labeled: list[dict]) -> tuple[float, float] | None:
    logits: list[float] = []
    labels: list[int] = []
    skipped = 0
    for row in labeled:
        if not isinstance(row, dict):
            skipped += 1
            continue
        y = _label_value(row)
        if y is None:
            skipped += 1
            continue
        try:
            logit = _raw_logit(row)
        except Exception:
            skipped += 1
            continue
        if not math.isfinite(logit):
            skipped += 1
            continue
        logits.append(float(logit))
        labels.append(y)

    if len(logits) < ONLINE_PLATT_MIN_EXAMPLES or len(set(labels)) < 2:
        if labeled:
            print(
                "[v1_kfactor] online Platt skipped: "
                f"usable={len(logits)} positives={sum(labels)} skipped={skipped}",
                flush=True,
            )
        return None

    prior_alpha, prior_beta = _offline_calibration_params()
    if not math.isfinite(prior_alpha) or prior_alpha <= 0.0:
        prior_alpha = 1.0
    if not math.isfinite(prior_beta):
        prior_beta = 0.0
    prior_alpha = max(min(prior_alpha, ONLINE_PLATT_MAX_ALPHA), ONLINE_PLATT_MIN_ALPHA)
    prior_beta = max(min(prior_beta, ONLINE_PLATT_MAX_ABS_BETA), -ONLINE_PLATT_MAX_ABS_BETA)

    try:
        x = torch.tensor(logits, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32)
        prior_log_alpha = torch.tensor(math.log(prior_alpha), dtype=torch.float32)
        prior_beta_t = torch.tensor(prior_beta, dtype=torch.float32)
        log_alpha = torch.nn.Parameter(prior_log_alpha.clone())
        beta = torch.nn.Parameter(prior_beta_t.clone())
        opt = torch.optim.LBFGS(
            [log_alpha, beta],
            lr=0.25,
            max_iter=30,
            line_search_fn="strong_wolfe",
        )
        loss_fn = torch.nn.BCEWithLogitsLoss()

        def closure():
            opt.zero_grad()
            alpha = torch.exp(log_alpha)
            loss = loss_fn(alpha * x + beta, y)
            reg = ONLINE_PLATT_L2 * (
                (log_alpha - prior_log_alpha).pow(2) + (beta - prior_beta_t).pow(2)
            )
            total = loss + reg
            total.backward()
            return total

        opt.step(closure)
        alpha = float(torch.exp(log_alpha).detach())
        beta_v = float(beta.detach())
    except Exception as exc:  # noqa: BLE001
        print(f"[v1_kfactor] online Platt fit failed: {exc!r}", flush=True)
        return None

    if not (math.isfinite(alpha) and math.isfinite(beta_v)):
        return None
    alpha = max(min(alpha, ONLINE_PLATT_MAX_ALPHA), ONLINE_PLATT_MIN_ALPHA)
    beta_v = max(min(beta_v, ONLINE_PLATT_MAX_ABS_BETA), -ONLINE_PLATT_MAX_ABS_BETA)
    print(
        "[v1_kfactor] online Platt fit: "
        f"n={len(labels)} positives={sum(labels)} skipped={skipped} "
        f"alpha={alpha:.3f} beta={beta_v:.3f}",
        flush=True,
    )
    return alpha, beta_v


def _logit_prob(p: float) -> float:
    p = max(min(float(p), 1.0 - 1e-6), 1e-6)
    return math.log(p / (1.0 - p))


def _fit_per_subject_shifts(labeled: list[dict]) -> dict[str, float]:
    try:
        by_subject: dict[str, list[tuple[int, float]]] = {}
        for row in labeled:
            if not isinstance(row, dict):
                continue
            y = _label_value(row)
            if y is None:
                continue
            try:
                logit = _raw_logit(row)
            except Exception:
                continue
            if not math.isfinite(logit):
                continue
            p = _sigmoid(float(logit))
            if not math.isfinite(p):
                continue
            subject_key = _normalize_subject(str(row.get("subject_content") or ""))
            by_subject.setdefault(subject_key, []).append((y, p))

        shifts: dict[str, float] = {}
        for subject_key, obs in by_subject.items():
            k_s = len(obs)
            if k_s < PER_SUBJECT_SHIFT_MIN_OBS:
                continue
            sum_y = sum(y for y, _ in obs)
            y_bar_smoothed = (sum_y + PER_SUBJECT_SHIFT_LABEL_SMOOTH) / (
                k_s + 2.0 * PER_SUBJECT_SHIFT_LABEL_SMOOTH
            )
            p_bar = sum(p for _, p in obs) / k_s
            raw_shift = _logit_prob(y_bar_smoothed) - _logit_prob(p_bar)
            delta_s = (k_s / (k_s + PER_SUBJECT_SHIFT_KAPPA)) * raw_shift
            delta_s = max(min(delta_s, PER_SUBJECT_SHIFT_CLIP), -PER_SUBJECT_SHIFT_CLIP)
            if math.isfinite(delta_s):
                shifts[subject_key] = float(delta_s)
        return shifts
    except Exception:  # noqa: BLE001
        return {}


def _per_subject_shifts(labeled: list[dict] | None) -> dict[str, float]:
    global _PER_SUBJECT_CACHE_KEY, _PER_SUBJECT_CACHE

    key = _labeled_cache_key(labeled)
    if key is None:
        return {}
    if key != _PER_SUBJECT_CACHE_KEY:
        _PER_SUBJECT_CACHE_KEY = key
        _PER_SUBJECT_CACHE = _fit_per_subject_shifts(labeled or [])
    return _PER_SUBJECT_CACHE or {}


def _online_calibration_params(labeled: list[dict] | None) -> tuple[float, float] | None:
    global _ONLINE_PLATT_CACHE_KEY, _ONLINE_PLATT_CACHE

    key = _labeled_cache_key(labeled)
    if key is None:
        return None
    if key != _ONLINE_PLATT_CACHE_KEY:
        _ONLINE_PLATT_CACHE_KEY = key
        _ONLINE_PLATT_CACHE = _fit_online_platt(labeled or [])
    return _ONLINE_PLATT_CACHE


def _calibrate_logit(logit: float, labeled: list[dict] | None) -> float:
    online = _online_calibration_params(labeled)
    if online is not None:
        alpha, beta = online
        return alpha * logit + beta
    return _apply_calibration(logit)


def _subject_only_prob(input: dict) -> float:
    """Fallback when the full pipeline can't run (e.g. encoder unavailable).

    Uses only the Stage 1 per-subject bias (item contribution = 0); this is
    always cheaper and never depends on the encoder. Returns the marginal floor
    if even the subject lookup fails.
    """
    try:
        subject_bias, _ = _subject_params(str(input.get("subject_content") or ""))
        return _clip_prob(_sigmoid(_apply_calibration(float(subject_bias))))
    except Exception:  # noqa: BLE001
        return 0.654  # marginal p(correct) from training data


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """Return probability that the subject answers the item correctly.

    Uses revealed per-round labels for online Platt calibration when possible.
    Layered fallback: full K-factor scoring -> Stage 1 subject-only -> marginal.
    predict() never raises; any exception degrades gracefully so the platform
    always receives a valid float and the round is scored.
    """
    if not ENCODER_OK and not JE_IRT_ACTIVE:
        return _subject_only_prob(input)
    try:
        if PRIOR_ONLY:
            logit = _raw_logit(input)
            # Lever F: per-subject shrunk shift from labeled rows on top of the prior.
            # _fit_per_subject_shifts already uses _raw_logit, which returns the prior
            # logit under PRIOR_ONLY, so the existing helper does the right thing.
            # NEVER apply _calibrate_logit here — calibration.json was fit against
            # K-factor logits, and applying it to prior logits regressed sub 8.
            shifts = _per_subject_shifts(labeled)
            if shifts:
                subject_key = _normalize_subject(str(input.get("subject_content") or ""))
                if subject_key in shifts:
                    logit = logit + shifts[subject_key]
            return _clip_prob(_sigmoid(logit))
        if JE_IRT_ACTIVE:
            return _je_irt_probability(input)
        logit = _raw_logit(input)
        shifts = _per_subject_shifts(labeled)
        subject_key = _normalize_subject(str(input.get("subject_content") or ""))
        if shifts and subject_key in shifts:
            logit = logit + shifts[subject_key]
        else:
            logit = _calibrate_logit(logit, labeled)
        p = _sigmoid(logit)
        return _clip_prob(p)
    except Exception as exc:  # noqa: BLE001
        print(f"[v1_kfactor] predict() fell back to subject-only: {exc!r}", flush=True)
        return _subject_only_prob(input)
