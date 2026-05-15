"""Create a tiny K-factor Stage 1 parquet fixture for export-contract CI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main(out_dir: Path, seed: int, n_subjects: int, n_items: int, k: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    subjects = pd.DataFrame(
        {
            "subject_id": [f"fixture_subject_{idx:03d}" for idx in range(n_subjects)],
            "subject_bias": rng.normal(0.0, 0.5, size=n_subjects).astype(np.float32),
            **{
                f"u_{dim}": rng.normal(0.0, 0.4, size=n_subjects).astype(np.float32)
                for dim in range(k)
            },
        }
    )
    items = pd.DataFrame(
        {
            "item_id": [f"fixture_item_{idx:03d}" for idx in range(n_items)],
            **{
                f"v_{dim}": rng.normal(0.0, 0.5, size=n_items).astype(np.float32)
                for dim in range(k)
            },
            "z": rng.normal(0.0, 0.6, size=n_items).astype(np.float32),
        }
    )

    subjects.to_parquet(out_dir / "subject_capabilities.parquet", index=False)
    items.to_parquet(out_dir / "item_parameters.parquet", index=False)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "fixture": True,
                "command": sys.argv,
                "git_sha": _git_sha(),
                "n_subjects": n_subjects,
                "n_items": n_items,
                "k": k,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Wrote K-factor Stage 1 parquet fixture to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/fixtures/kfactor/stage1_parquet", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-subjects", type=int, default=5)
    parser.add_argument("--n-items", type=int, default=10)
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()
    main(args.out, args.seed, args.n_subjects, args.n_items, args.k)
