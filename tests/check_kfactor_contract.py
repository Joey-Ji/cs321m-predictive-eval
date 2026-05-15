"""Contract test for K-factor Stage 1 and (optionally) Stage 2 artifacts.

Run:
    python tests/check_kfactor_contract.py --stage1 data/stage1/kfactor_k4
    python tests/check_kfactor_contract.py --make-fixture data/fixtures/kfactor
    python tests/check_kfactor_contract.py --stage1 data/fixtures/kfactor/stage1
    python tests/check_kfactor_contract.py --stage2 data/fixtures/kfactor/stage2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REQUIRED_STAGE1_FILES = {
    "subject_state.pt": "subject K-factor parameters",
    "item_targets.pt": "item K-factor targets",
    "subject_to_id.json": "raw subject id -> int id",
    "subject_name_to_id.json": "normalized subject name -> int id",
    "item_to_id.json": "raw item_id -> int id",
    "manifest.json": "Stage 1 run metadata",
}


def _check_contiguous_mapping(name: str, mapping: object, n: int, errs: list[str]) -> None:
    if not isinstance(mapping, dict):
        errs.append(f"{name} must be a JSON object, got {type(mapping).__name__}")
        return
    values = list(mapping.values())
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        errs.append(f"{name} ids must all be integers")
        return
    if sorted(values) != list(range(n)):
        errs.append(f"{name} ids must be contiguous 0..{n - 1}")


def _load_pt(path: Path, errs: list[str]):
    import torch

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
    if caught:
        first = str(caught[0].message)
        errs.append(f"{path.name} emitted warning while loading: {first}")
    return obj


def _load_json(path: Path, errs: list[str]) -> object:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        errs.append(f"{path.name} failed to parse as JSON: {exc}")
        return {}


def _check_finite_tensor(name: str, tensor, errs: list[str]) -> None:
    import torch

    if not torch.isfinite(tensor).all():
        errs.append(f"{name} contains NaN/inf")


def check_stage1(stage1_dir: Path) -> list[str]:
    import torch

    errs: list[str] = []
    if not stage1_dir.is_dir():
        return [f"FATAL: stage1 directory {stage1_dir} does not exist"]

    for filename, desc in REQUIRED_STAGE1_FILES.items():
        if not (stage1_dir / filename).exists():
            errs.append(f"MISSING: {stage1_dir / filename} ({desc})")
    if errs:
        return errs

    subject_state = _load_pt(stage1_dir / "subject_state.pt", errs)
    item_targets = _load_pt(stage1_dir / "item_targets.pt", errs)
    subject_to_id = _load_json(stage1_dir / "subject_to_id.json", errs)
    subject_name_to_id = _load_json(stage1_dir / "subject_name_to_id.json", errs)
    item_to_id = _load_json(stage1_dir / "item_to_id.json", errs)
    manifest = _load_json(stage1_dir / "manifest.json", errs)

    if not isinstance(subject_state, dict):
        errs.append(f"subject_state.pt must load to dict, got {type(subject_state).__name__}")
    if not isinstance(item_targets, dict):
        errs.append(f"item_targets.pt must load to dict, got {type(item_targets).__name__}")
    if errs:
        return errs

    for key in ("subject_bias", "subject_u", "fallback_bias", "fallback_u"):
        if key not in subject_state:
            errs.append(f"subject_state.pt missing key: {key}")
    for key in ("item_v", "item_z"):
        if key not in item_targets:
            errs.append(f"item_targets.pt missing key: {key}")
    if errs:
        return errs

    subject_bias = subject_state["subject_bias"]
    subject_u = subject_state["subject_u"]
    fallback_bias = subject_state["fallback_bias"]
    fallback_u = subject_state["fallback_u"]
    item_v = item_targets["item_v"]
    item_z = item_targets["item_z"]

    for name, tensor in (
        ("subject_bias", subject_bias),
        ("subject_u", subject_u),
        ("fallback_u", fallback_u),
        ("item_v", item_v),
        ("item_z", item_z),
    ):
        if not torch.is_tensor(tensor):
            errs.append(f"{name} must be a tensor, got {type(tensor).__name__}")
    if torch.is_tensor(fallback_bias):
        pass
    elif not isinstance(fallback_bias, (int, float)):
        errs.append(f"fallback_bias must be a scalar tensor or python float, got {type(fallback_bias).__name__}")
    if errs:
        return errs

    if subject_bias.dtype != torch.float32 or subject_bias.ndim != 1:
        errs.append(f"subject_bias must be float32 [n_subjects], got {subject_bias.dtype} {tuple(subject_bias.shape)}")
    if subject_u.dtype != torch.float32 or subject_u.ndim != 2:
        errs.append(f"subject_u must be float32 [n_subjects, k], got {subject_u.dtype} {tuple(subject_u.shape)}")
    if fallback_u.dtype != torch.float32 or fallback_u.ndim != 1:
        errs.append(f"fallback_u must be float32 [k], got {fallback_u.dtype} {tuple(fallback_u.shape)}")
    if torch.is_tensor(fallback_bias) and fallback_bias.dtype != torch.float32:
        errs.append(f"fallback_bias tensor must be float32, got {fallback_bias.dtype}")
    if item_v.dtype != torch.float32 or item_v.ndim != 2:
        errs.append(f"item_v must be float32 [n_items, k], got {item_v.dtype} {tuple(item_v.shape)}")
    if item_z.dtype != torch.float32 or item_z.ndim != 1:
        errs.append(f"item_z must be float32 [n_items], got {item_z.dtype} {tuple(item_z.shape)}")

    if errs:
        return errs

    n_subjects, k = subject_u.shape
    n_items = item_v.shape[0]
    if k != 4:
        errs.append(f"K-factor v1 expects k=4, got {k}")
    if subject_bias.shape != (n_subjects,):
        errs.append(f"subject_bias length {len(subject_bias)} != subject_u rows {n_subjects}")
    if fallback_u.shape != (k,):
        errs.append(f"fallback_u shape {tuple(fallback_u.shape)} != ({k},)")
    if item_v.shape[1] != k:
        errs.append(f"item_v k {item_v.shape[1]} != subject_u k {k}")
    if item_z.shape != (n_items,):
        errs.append(f"item_z length {len(item_z)} != item_v rows {n_items}")

    if torch.is_tensor(fallback_bias):
        if fallback_bias.ndim != 0:
            errs.append(f"fallback_bias tensor must be scalar, got shape {tuple(fallback_bias.shape)}")
        elif not torch.isfinite(fallback_bias):
            errs.append("fallback_bias contains NaN/inf")
    elif not isinstance(fallback_bias, (int, float)) or not math.isfinite(float(fallback_bias)):
        errs.append(f"fallback_bias must be finite float or scalar tensor, got {type(fallback_bias).__name__}")

    for name, tensor in (
        ("subject_bias", subject_bias),
        ("subject_u", subject_u),
        ("fallback_u", fallback_u),
        ("item_v", item_v),
        ("item_z", item_z),
    ):
        _check_finite_tensor(name, tensor, errs)

    if isinstance(subject_to_id, dict) and len(subject_to_id) != n_subjects:
        errs.append(f"subject_to_id size {len(subject_to_id)} != n_subjects {n_subjects}")
    if isinstance(subject_name_to_id, dict) and len(subject_name_to_id) != n_subjects:
        errs.append(f"subject_name_to_id size {len(subject_name_to_id)} != n_subjects {n_subjects}")
    if isinstance(item_to_id, dict) and len(item_to_id) != n_items:
        errs.append(f"item_to_id size {len(item_to_id)} != n_items {n_items}")

    _check_contiguous_mapping("subject_to_id", subject_to_id, n_subjects, errs)
    _check_contiguous_mapping("subject_name_to_id", subject_name_to_id, n_subjects, errs)
    _check_contiguous_mapping("item_to_id", item_to_id, n_items, errs)

    manifest_required = {"command", "git_sha", "n_subjects", "n_items", "k"}
    if not isinstance(manifest, dict):
        errs.append(f"manifest.json must be a JSON object, got {type(manifest).__name__}")
    else:
        missing_manifest = sorted(manifest_required - set(manifest))
        if missing_manifest:
            errs.append(f"manifest.json missing keys: {missing_manifest}")
        else:
            if int(manifest["n_subjects"]) != n_subjects:
                errs.append(f"manifest n_subjects {manifest['n_subjects']} != {n_subjects}")
            if int(manifest["n_items"]) != n_items:
                errs.append(f"manifest n_items {manifest['n_items']} != {n_items}")
            if int(manifest["k"]) != k:
                errs.append(f"manifest k {manifest['k']} != {k}")

    print(f"  K-factor Stage 1: subjects={n_subjects:,} items={n_items:,} k={k}")
    return errs


REPRESENTATION_VERSION = "item_text_plus_side_features_v1"
REQUIRED_STAGE2_FILES = {
    "head.pt": "trained K-factor head weights",
    "head_meta.json": "head architecture + representation metadata",
    "target_scaler.json": "per-target mean/std for inverse-transform",
    "side_feature_meta.json": "one-hot vocab for benchmark + condition",
}


def check_stage2(stage2_dir: Path) -> list[str]:
    errs: list[str] = []
    if not stage2_dir.is_dir():
        return [f"FATAL: stage2 directory {stage2_dir} does not exist"]

    for filename, desc in REQUIRED_STAGE2_FILES.items():
        if not (stage2_dir / filename).exists():
            errs.append(f"MISSING: {stage2_dir / filename} ({desc})")
    if errs:
        return errs

    head_meta = json.loads((stage2_dir / "head_meta.json").read_text())
    vocab = json.loads((stage2_dir / "side_feature_meta.json").read_text())

    rep = head_meta.get("representation_version")
    if rep != REPRESENTATION_VERSION:
        errs.append(
            f"head_meta.json representation_version {rep!r} != expected {REPRESENTATION_VERSION!r}"
        )

    for key in ("in_dim", "embedding_dim", "side_feature_dim"):
        if key not in head_meta:
            errs.append(f"head_meta.json missing key: {key}")
    if errs:
        return errs

    in_dim = int(head_meta["in_dim"])
    embedding_dim = int(head_meta["embedding_dim"])
    side_feature_dim = int(head_meta["side_feature_dim"])
    if embedding_dim + side_feature_dim != in_dim:
        errs.append(
            f"in_dim {in_dim} != embedding_dim {embedding_dim} + side_feature_dim {side_feature_dim}"
        )

    vocab_dim = int(vocab.get("side_feature_dim", -1))
    if vocab_dim != side_feature_dim:
        errs.append(
            f"side_feature_meta.json side_feature_dim {vocab_dim} != head_meta side_feature_dim {side_feature_dim}"
        )
    if int(vocab.get("benchmark_dim", -1)) + int(vocab.get("condition_dim", -1)) != vocab_dim:
        errs.append(
            f"side_feature_meta.json benchmark_dim + condition_dim != side_feature_dim {vocab_dim}"
        )

    print(
        f"  K-factor Stage 2: in_dim={in_dim} embedding_dim={embedding_dim} "
        f"side_feature_dim={side_feature_dim} representation={rep!r}"
    )
    return errs


def main(stage1_dir: Path | None, stage2_dir: Path | None, make_fixture_dir: Path | None) -> int:
    if make_fixture_dir is not None:
        from scripts.make_kfactor_fixture import main as make_fixture

        make_fixture(make_fixture_dir, seed=0, n_subjects=5, n_items=20, k=4)
        return 0

    if stage1_dir is None and stage2_dir is None:
        print("ERROR: provide --stage1, --stage2, or --make-fixture", file=sys.stderr)
        return 2

    print("K-factor contract check")
    errs: list[str] = []
    if stage1_dir is not None:
        print(f"  stage1: {stage1_dir}")
        errs.extend(check_stage1(stage1_dir))
    if stage2_dir is not None:
        print(f"  stage2: {stage2_dir}")
        errs.extend(check_stage2(stage2_dir))

    if not errs:
        print("\nPASS - K-factor contract is satisfied.")
        return 0

    print(f"\nFAIL - {len(errs)} contract violation(s):")
    for err in errs:
        print(f"  - {err}")
    return 1 if any(not e.startswith("MISSING") for e in errs) else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", type=Path)
    parser.add_argument("--stage2", type=Path)
    parser.add_argument("--make-fixture", type=Path)
    args = parser.parse_args()
    sys.exit(main(args.stage1, args.stage2, args.make_fixture))
