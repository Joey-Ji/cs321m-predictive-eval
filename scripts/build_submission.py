"""Assemble a versioned submission ZIP from a submission directory.

A submission directory must contain at minimum a model.py. Optionally:
  - labeling.py
  - models.txt
  - requirements.txt
  - Any auxiliary state files (head.pt, theta.pt, subject_to_id.json, ...).

This script copies the named submission's contents into a flat ZIP at the
top level (which is what Codabench expects). It refuses to include the data/
and submissions/ subdirs to keep ZIPs lean.

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


def gather_state_files(submission_dir: Path, extra_dirs: list[Path]) -> list[Path]:
    """Return paths of state files (.pt, .npy, .json, .pkl) to include."""
    files: list[Path] = []
    for d in extra_dirs:
        if not d.exists():
            print(f"WARN: include dir not found: {d}", file=sys.stderr)
            continue
        for p in sorted(d.iterdir()):
            if p.suffix in (".pt", ".npy", ".json", ".pkl"):
                files.append(p)
    return files


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

    state_files = gather_state_files(sub_dir, includes)

    print(f"Building {zip_path} ...")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(sub_dir.iterdir()):
            if p.is_file() and (p.name in REQUIRED or p.name in OPTIONAL or p.suffix in (".pt", ".npy", ".json", ".pkl")):
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
