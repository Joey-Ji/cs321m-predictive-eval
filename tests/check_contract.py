"""Stage 1 ↔ Stage 2 contract test.

Verifies that whatever is in data/irt/ has the schema Stage 2 (train_content_head.py)
and submissions/v1_irt/model.py expect to consume.

Run:
    make test
    or:  python tests/check_contract.py
    or:  python tests/check_contract.py --irt data/irt --head data/head

Exit codes:
    0  all checks pass
    1  schema violation
    2  missing required file
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED_IRT_FILES = {
    "theta.pt": "per-subject ability (float32 [n_subjects])",
    "b.pt": "per-item difficulty (float32 [n_items])",
    "log_a.pt": "per-item log-discrimination (float32 [n_items]; zeros for 1PL)",
    "subject_to_id.json": "normalized subject name -> int id",
    "item_to_id.json": "raw item_id -> int id",
}

REQUIRED_HEAD_FILES = {
    "head.pt": "trained content-head state_dict",
    "head_meta.json": "head architecture spec",
}

EXPECTED_HEAD_META_KEYS = {"head_type", "in_dim", "out_dim", "encoder", "targets", "target_order"}


def check_irt(irt_dir: Path) -> list[str]:
    import torch

    errs: list[str] = []
    if not irt_dir.is_dir():
        errs.append(f"FATAL: irt directory {irt_dir} does not exist")
        return errs

    for name, desc in REQUIRED_IRT_FILES.items():
        p = irt_dir / name
        if not p.exists():
            errs.append(f"MISSING: {p}  ({desc})")
    if errs:
        return errs

    theta = torch.load(irt_dir / "theta.pt", weights_only=True)
    b = torch.load(irt_dir / "b.pt", weights_only=True)
    log_a = torch.load(irt_dir / "log_a.pt", weights_only=True)
    subject_to_id = json.loads((irt_dir / "subject_to_id.json").read_text())
    item_to_id = json.loads((irt_dir / "item_to_id.json").read_text())

    if theta.dtype != torch.float32:
        errs.append(f"theta.pt dtype is {theta.dtype}, expected float32")
    if b.dtype != torch.float32:
        errs.append(f"b.pt dtype is {b.dtype}, expected float32")
    if log_a.dtype != torch.float32:
        errs.append(f"log_a.pt dtype is {log_a.dtype}, expected float32")

    if theta.ndim != 1 or b.ndim != 1 or log_a.ndim != 1:
        errs.append(f"all IRT tensors must be 1-D: theta={tuple(theta.shape)} b={tuple(b.shape)} log_a={tuple(log_a.shape)}")

    n_subj, n_items = len(theta), len(b)
    if len(log_a) != n_items:
        errs.append(f"log_a length {len(log_a)} != b length {n_items}")
    if len(subject_to_id) != n_subj:
        errs.append(f"subject_to_id size {len(subject_to_id)} != theta length {n_subj}")
    if len(item_to_id) != n_items:
        errs.append(f"item_to_id size {len(item_to_id)} != b length {n_items}")

    sids = set(subject_to_id.values())
    if sids and (min(sids) != 0 or max(sids) != n_subj - 1):
        errs.append("subject_to_id ids must be contiguous 0..n_subj-1")
    iids = set(item_to_id.values())
    if iids and (min(iids) != 0 or max(iids) != n_items - 1):
        errs.append("item_to_id ids must be contiguous 0..n_items-1")

    print(f"  IRT: subjects={n_subj:,}  items={n_items:,}")
    if (irt_dir / "fit_log.json").exists():
        info = json.loads((irt_dir / "fit_log.json").read_text())
        print(f"  IRT model={info.get('model', '?')}  is_mock={info.get('is_mock', False)}")
    return errs


def check_head(head_dir: Path) -> list[str]:
    errs: list[str] = []
    if not head_dir.is_dir():
        print(f"  head: directory {head_dir} does not exist (skipping head contract check)")
        return errs

    for name, desc in REQUIRED_HEAD_FILES.items():
        p = head_dir / name
        if not p.exists():
            errs.append(f"MISSING: {p}  ({desc})")
    if errs:
        return errs

    meta = json.loads((head_dir / "head_meta.json").read_text())
    missing_keys = EXPECTED_HEAD_META_KEYS - set(meta)
    if missing_keys:
        errs.append(f"head_meta.json missing keys: {sorted(missing_keys)}")
    if meta.get("out_dim") not in (1, 2):
        errs.append(f"head_meta.json out_dim must be 1 or 2, got {meta.get('out_dim')}")
    if meta.get("targets") not in ("b", "b+log_a"):
        errs.append(f"head_meta.json targets must be 'b' or 'b+log_a', got {meta.get('targets')!r}")

    expected_target_order = ["b"] if meta.get("targets") == "b" else ["b", "log_a"]
    if meta.get("target_order") != expected_target_order:
        errs.append(f"target_order={meta.get('target_order')} inconsistent with targets={meta.get('targets')}")

    print(f"  head: type={meta.get('head_type')} in_dim={meta.get('in_dim')} out_dim={meta.get('out_dim')} "
          f"targets={meta.get('targets')} encoder={meta.get('encoder')}")
    return errs


def main(irt_dir: Path, head_dir: Path) -> int:
    print(f"Contract check")
    print(f"  irt:  {irt_dir}")
    print(f"  head: {head_dir}")
    print()

    irt_errs = check_irt(irt_dir)
    head_errs = check_head(head_dir)
    errs = irt_errs + head_errs

    if not errs:
        print("\nPASS — Stage 1 ↔ Stage 2 contract is satisfied.")
        return 0

    print(f"\nFAIL — {len(errs)} contract violation(s):")
    for e in errs:
        print(f"  - {e}")
    return 1 if any(not e.startswith("MISSING") for e in errs) else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--irt", default="data/irt", type=Path)
    parser.add_argument("--head", default="data/head", type=Path)
    args = parser.parse_args()
    sys.exit(main(args.irt, args.head))
