"""Assemble a versioned submission ZIP from a submission directory.

A submission directory must contain at minimum a model.py. Optionally:
  - labeling.py
  - models.txt
  - requirements.txt
  - Any auxiliary state files (head.pt, theta.pt, subject_to_id.json, ...).

This script copies the named submission's runtime contents into a flat ZIP at
the top level (which is what Codabench expects). Include dirs are scanned
recursively, but only known runtime state files are flattened into the ZIP.

Usage:
    python scripts/build_submission.py v1_irt
    python scripts/build_submission.py v1_irt --include data/head data/irt data/embeddings
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

REQUIRED = {"model.py"}
OPTIONAL = {"labeling.py", "models.txt", "requirements.txt"}
RUNTIME_STATE_FILES = {
    # v1_irt
    "head.pt",
    "head_meta.json",
    "theta.pt",
    "log_a.pt",
    "subject_to_id.json",
    # v1_kfactor
    "target_scaler.json",
    "subject_state.pt",
    "subject_name_to_id.json",
    "calibration.json",
}
SUBMISSION_RUNTIME_STATE_FILES = {
    "v1_kfactor": {
        "head.pt",
        "head_meta.json",
        "target_scaler.json",
        "subject_state.pt",
        "subject_name_to_id.json",
        "calibration.json",
    },
    "v1_irt": {
        "head.pt",
        "head_meta.json",
        "theta.pt",
        "log_a.pt",
        "subject_to_id.json",
    },
}


def gather_state_files(extra_dirs: list[Path], allowed_names: set[str]) -> list[Path]:
    """Return runtime state files from include dirs, flattened by filename."""
    files_by_name: dict[str, Path] = {}
    for d in extra_dirs:
        if not d.exists():
            print(f"WARN: include dir not found: {d}", file=sys.stderr)
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.name in allowed_names:
                files_by_name[p.name] = p
    return [files_by_name[name] for name in sorted(files_by_name)]


def main(name: str, includes: list[Path], out_dir: Path) -> None:
    submissions_root = Path(__file__).resolve().parent.parent / "submissions"
    sub_dir = submissions_root / name
    if not sub_dir.exists():
        sys.exit(f"submission directory not found: {sub_dir}")

    model_py = sub_dir / "model.py"
    if not model_py.exists():
        sys.exit(f"required file missing: {model_py}")

    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{name}.zip"
    if zip_path.exists():
        zip_path.unlink()

    allowed_state_files = SUBMISSION_RUNTIME_STATE_FILES.get(name, RUNTIME_STATE_FILES)
    state_files = gather_state_files(includes, allowed_state_files)

    print(f"Building {zip_path} ...")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(sub_dir.iterdir()):
            if p.is_file() and (p.name in REQUIRED or p.name in OPTIONAL or p.name in allowed_state_files):
                z.write(p, p.name)
                print(f"  + {p.name}")
        for p in state_files:
            z.write(p, p.name)
            print(f"  + {p.name}  (from {p.parent})")

    size = zip_path.stat().st_size
    print(f"Done. {zip_path} ({size / 1024:.1f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Submission directory name under submissions/")
    parser.add_argument(
        "--include",
        nargs="*",
        default=["data/head", "data/irt"],
        help="Additional dirs to scan for .pt/.npy/.json/.pkl state files.",
    )
    parser.add_argument("--out", default="submissions", type=Path)
    args = parser.parse_args()
    main(args.name, [Path(p) for p in args.include], args.out)
