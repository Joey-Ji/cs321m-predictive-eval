"""Export Stage 1 K-factor parquet artifacts into the runtime contract.

The committed Stage 1 handoff stores MIRT parameters as parquet tables.  The
v1_kfactor runtime expects the compact torch/json contract under
data/stage1/kfactor_k4/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

K = 4
NAME_LINE = re.compile(r"^\s*Name:\s*(.+?)\s*$", re.MULTILINE)
SUBJECT_BIAS_CANDIDATES = ("subject_bias", "bias")
ITEM_Z_CANDIDATES = ("z", "item_z", "offset")
SUBJECT_NAME_CANDIDATES = (
    "normalized_name",
    "subject_name_normalized",
    "normalized_subject_name",
    "subject_key",
    "subject_name",
    "display_name",
    "name",
)


def _normalize_subject(subject_content: str) -> str:
    """Mirror submissions/v1_kfactor/model.py subject lookup normalization."""
    if not subject_content:
        return ""
    m = NAME_LINE.search(subject_content)
    return m.group(1).strip().lower() if m else subject_content.strip().lower()


def _normalize_display_name(display_name: str) -> str:
    return _normalize_subject(f"Name: {display_name}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _choose_column(df: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"{label} column not found; expected one of {candidates}, got {df.columns.tolist()}")


def _sort_by_string_id(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    if id_col not in df.columns:
        raise ValueError(f"required column {id_col!r} missing; got {df.columns.tolist()}")
    if df[id_col].isna().any():
        raise ValueError(f"{id_col} contains null values")
    out = df.assign(_sort_id=df[id_col].astype(str))
    out = out.sort_values("_sort_id", kind="mergesort").drop(columns="_sort_id")
    return out.reset_index(drop=True)


def _prefixed_columns(df: pd.DataFrame, prefix: str, k: int) -> list[str]:
    cols = [f"{prefix}_{idx}" for idx in range(k)]
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"missing {prefix} columns {missing}; got {df.columns.tolist()}")
    return cols


def _clean_optional_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _float32_numpy(df: pd.DataFrame, cols: list[str] | str, label: str) -> np.ndarray:
    values = df[cols].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains NaN/inf")
    return values


def _subject_registry_name_keys(raw_subject_ids: list[str], subjects_parquet: Path) -> tuple[list[str], str]:
    if not subjects_parquet.exists():
        raise FileNotFoundError(subjects_parquet)

    registry = pd.read_parquet(subjects_parquet)
    missing_cols = [col for col in ("subject_id", "display_name") if col not in registry.columns]
    if missing_cols:
        raise ValueError(
            f"subjects parquet missing required column(s) {missing_cols}; got {registry.columns.tolist()}"
        )
    if registry["subject_id"].isna().any():
        raise ValueError("subjects parquet subject_id contains null values")

    registry = registry.assign(subject_id=registry["subject_id"].astype(str))
    duplicate_ids = registry.loc[registry["subject_id"].duplicated(), "subject_id"].head(5).tolist()
    if duplicate_ids:
        preview = ", ".join(repr(subject_id) for subject_id in duplicate_ids)
        raise ValueError(f"subjects parquet has duplicate subject_id values: {preview}")

    joined = pd.DataFrame({"subject_id": raw_subject_ids}).merge(
        registry[["subject_id", "display_name"]],
        on="subject_id",
        how="left",
        validate="one_to_one",
    )

    keys: list[str] = []
    missing_display_names: list[str] = []
    for subject_id, display_name in zip(joined["subject_id"].tolist(), joined["display_name"].tolist()):
        clean_name = _clean_optional_text(display_name)
        if clean_name is None:
            missing_display_names.append(subject_id)
            keys.append(_normalize_subject(subject_id))
            continue
        keys.append(_normalize_display_name(clean_name))

    if missing_display_names:
        preview = ", ".join(repr(subject_id) for subject_id in missing_display_names[:5])
        print(
            "WARN: subjects parquet did not provide display_name for "
            f"{len(missing_display_names)} of {len(raw_subject_ids)} subject_id values "
            f"({preview}); using normalized raw subject_id fallback for those rows. "
            "Pass a complete --subjects-parquet to avoid runtime subject lookup misses.",
            file=sys.stderr,
        )
    return keys, "subjects_parquet.display_name"


def _subject_name_keys(
    subjects: pd.DataFrame,
    raw_subject_ids: list[str],
    subjects_parquet: Path | None,
) -> tuple[list[str], str]:
    if subjects_parquet is not None:
        return _subject_registry_name_keys(raw_subject_ids, subjects_parquet)

    for col in SUBJECT_NAME_CANDIDATES:
        if col in subjects.columns:
            return [_normalize_subject(str(value)) for value in subjects[col].tolist()], col
    if "subject_content" in subjects.columns:
        return [_normalize_subject(str(value)) for value in subjects["subject_content"].tolist()], "subject_content"

    print(
        "WARN: subject parquet has no normalized name or subject_content column; "
        "using normalized raw subject_id values for subject_name_to_id.json. "
        "Pass --subjects-parquet data/subjects.parquet so display_name can be joined by subject_id.",
        file=sys.stderr,
    )
    return [_normalize_subject(subject_id) for subject_id in raw_subject_ids], "subject_id"


def _mapping(keys: list[str], label: str) -> dict[str, int]:
    mapping: dict[str, int] = {}
    duplicates: list[str] = []
    for idx, key in enumerate(keys):
        if key in mapping:
            duplicates.append(key)
            continue
        mapping[key] = idx
    if duplicates:
        preview = ", ".join(repr(key) for key in duplicates[:5])
        raise ValueError(f"{label} has duplicate keys after normalization: {preview}")
    return mapping


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def export(subject_parquet: Path, item_parquet: Path, out_dir: Path, subjects_parquet: Path | None) -> None:
    if not subject_parquet.exists():
        raise FileNotFoundError(subject_parquet)
    if not item_parquet.exists():
        raise FileNotFoundError(item_parquet)

    subjects = _sort_by_string_id(pd.read_parquet(subject_parquet), "subject_id")
    items = _sort_by_string_id(pd.read_parquet(item_parquet), "item_id")

    subject_bias_col = _choose_column(subjects, SUBJECT_BIAS_CANDIDATES, "subject bias")
    item_z_col = _choose_column(items, ITEM_Z_CANDIDATES, "item offset")
    subject_u_cols = _prefixed_columns(subjects, "u", K)
    item_v_cols = _prefixed_columns(items, "v", K)

    raw_subject_ids = [str(value) for value in subjects["subject_id"].tolist()]
    raw_item_ids = [str(value) for value in items["item_id"].tolist()]
    if len(raw_subject_ids) != len(set(raw_subject_ids)):
        raise ValueError("subject_id values are not unique")
    if len(raw_item_ids) != len(set(raw_item_ids)):
        raise ValueError("item_id values are not unique")

    subject_name_keys, subject_name_source = _subject_name_keys(subjects, raw_subject_ids, subjects_parquet)
    subject_to_id = {subject_id: idx for idx, subject_id in enumerate(raw_subject_ids)}
    subject_name_to_id = _mapping(subject_name_keys, "subject_name_to_id")
    item_to_id = {item_id: idx for idx, item_id in enumerate(raw_item_ids)}

    subject_bias = torch.as_tensor(
        _float32_numpy(subjects, subject_bias_col, "subject_bias"), dtype=torch.float32
    )
    subject_u = torch.as_tensor(
        _float32_numpy(subjects, subject_u_cols, "subject_u"), dtype=torch.float32
    )
    item_v = torch.as_tensor(_float32_numpy(items, item_v_cols, "item_v"), dtype=torch.float32)
    item_z = torch.as_tensor(_float32_numpy(items, item_z_col, "item_z"), dtype=torch.float32)

    if subject_u.shape != (len(subjects), K):
        raise ValueError(f"subject_u shape {tuple(subject_u.shape)} != ({len(subjects)}, {K})")
    if item_v.shape != (len(items), K):
        raise ValueError(f"item_v shape {tuple(item_v.shape)} != ({len(items)}, {K})")

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "subject_bias": subject_bias,
            "subject_u": subject_u,
            "fallback_bias": subject_bias.mean(),
            "fallback_u": subject_u.mean(dim=0),
        },
        out_dir / "subject_state.pt",
    )
    torch.save(
        {
            "item_v": item_v,
            "item_z": item_z,
        },
        out_dir / "item_targets.pt",
    )
    _write_json(out_dir / "subject_to_id.json", subject_to_id)
    _write_json(out_dir / "subject_name_to_id.json", subject_name_to_id)
    _write_json(out_dir / "item_to_id.json", item_to_id)

    source_manifest = _read_json_if_exists(subject_parquet.parent / "manifest.json")
    input_parquet_paths = {
        "subject_parquet": str(subject_parquet),
        "item_parquet": str(item_parquet),
    }
    input_sha256 = {
        "subject_parquet": _sha256(subject_parquet),
        "item_parquet": _sha256(item_parquet),
    }
    if subjects_parquet is not None:
        input_parquet_paths["subjects_parquet"] = str(subjects_parquet)
        input_sha256["subjects_parquet"] = _sha256(subjects_parquet)

    manifest = {
        "command": sys.argv,
        "git_sha": _git_sha(),
        "input_parquet_paths": input_parquet_paths,
        "input_sha256": input_sha256,
        "n_subjects": len(subjects),
        "n_items": len(items),
        "k": K,
        "source_run": source_manifest,
        "subject_name_source": subject_name_source,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    _write_json(out_dir / "manifest.json", manifest)

    print(f"Exported K-factor Stage 1 contract to {out_dir}")
    print(f"  subjects={len(subjects):,} items={len(items):,} k={K}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject-parquet",
        default="stage_1/k_factor_irt/artifacts/k4_full_train/subject_capabilities.parquet",
        type=Path,
    )
    parser.add_argument(
        "--item-parquet",
        default="stage_1/k_factor_irt/artifacts/k4_full_train/item_parameters.parquet",
        type=Path,
    )
    parser.add_argument(
        "--subjects-parquet",
        default=None,
        type=Path,
        help="Optional subject registry parquet with subject_id and display_name columns.",
    )
    parser.add_argument("--out-dir", default="data/stage1/kfactor_k4", type=Path)
    args = parser.parse_args()
    export(args.subject_parquet, args.item_parquet, args.out_dir, args.subjects_parquet)


if __name__ == "__main__":
    main()
