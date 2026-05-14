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
SUBMISSION_REQUIRED_STATE_FILES = {
    "v1_kfactor": {
        "head.pt",
        "head_meta.json",
        "target_scaler.json",
        "subject_state.pt",
        "subject_name_to_id.json",
    },
    "v1_irt": {
        "head.pt",
        "head_meta.json",
        "theta.pt",
        "subject_to_id.json",
    },
}
SUBMISSION_DEFAULT_INCLUDES = {
    "v1_kfactor": [
        "data/stage2/kfactor_mpnet_linear_v1",
        "data/stage1/kfactor_k4",
        "data/calibration/kfactor_mpnet_linear_v1",
    ],
    "v1_irt": ["data/head", "data/irt"],
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
                previous = files_by_name.get(p.name)
                if previous is not None:
                    print(f"WARN: shadowed {previous} with {p}", file=sys.stderr)
                files_by_name[p.name] = p
    return [files_by_name[name] for name in sorted(files_by_name)]


def main(name: str, includes: list[Path] | None, out_dir: Path) -> None:
    submissions_root = Path(__file__).resolve().parent.parent / "submissions"
    sub_dir = submissions_root / name
    if not sub_dir.exists():
        sys.exit(f"submission directory not found: {sub_dir}")

    model_py = sub_dir / "model.py"
    if not model_py.exists():
        sys.exit(f"required file missing: {model_py}")

    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{name}.zip"

    if not includes:
        includes = [Path(p) for p in SUBMISSION_DEFAULT_INCLUDES.get(name, ["data/head", "data/irt"])]

    allowed_state_files = SUBMISSION_RUNTIME_STATE_FILES.get(name, RUNTIME_STATE_FILES)
    required_state_files = SUBMISSION_REQUIRED_STATE_FILES.get(name, set())
    local_state_files = {
        p.name: p
        for p in sorted(sub_dir.iterdir())
        if p.is_file() and p.name in allowed_state_files
    }
    state_files = gather_state_files(includes, allowed_state_files)
    include_state_files = {p.name: p for p in state_files}

    missing_required = sorted(required_state_files - set(local_state_files) - set(include_state_files))
    if missing_required:
        include_msg = ", ".join(str(p) for p in includes) if includes else "(none)"
        sys.exit(
            f"missing required runtime state for {name}: {', '.join(missing_required)}. "
            f"Checked submission dir {sub_dir} and include dirs: {include_msg}"
        )

    if zip_path.exists():
        zip_path.unlink()

    print(f"Building {zip_path} ...")
    written_names: set[str] = set()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(sub_dir.iterdir()):
            if p.is_file() and (p.name in REQUIRED or p.name in OPTIONAL or p.name in allowed_state_files):
                z.write(p, p.name)
                written_names.add(p.name)
                print(f"  + {p.name}")
        for p in state_files:
            if p.name in written_names:
                print(f"WARN: skipping {p}; {p.name} already provided by submission dir", file=sys.stderr)
                continue
            z.write(p, p.name)
            written_names.add(p.name)
            print(f"  + {p.name}  (from {p.parent})")

    size = zip_path.stat().st_size
    print(f"Done. {zip_path} ({size / 1024:.1f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Submission directory name under submissions/")
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Additional dirs to scan for known runtime state files. Defaults are submission-specific.",
    )
    parser.add_argument("--out", default="submissions", type=Path)
    args = parser.parse_args()
    main(args.name, None if args.include is None else [Path(p) for p in args.include], args.out)
