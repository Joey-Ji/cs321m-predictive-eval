"""Create a deterministic synthetic K-factor fixture for local Stage 2 checks."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

NAME_LINE = re.compile(r"^\s*Name:\s*(.+?)\s*$", re.MULTILINE)


def normalize_subject(subject_content: str) -> str:
    if not subject_content:
        return ""
    m = NAME_LINE.search(subject_content)
    return m.group(1).strip().lower() if m else subject_content.strip().lower()


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
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    if out_dir.exists():
        shutil.rmtree(out_dir)
    stage1_dir = out_dir / "stage1"
    stage1_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    subject_names = [f"FixtureModel-{i}" for i in range(n_subjects)]
    subject_contents = [
        f"Name: {name}\nOrganization: Fixture Lab\nParameters: {7 + i}B"
        for i, name in enumerate(subject_names)
    ]
    item_ids = [f"fixture_item_{i:03d}" for i in range(n_items)]

    subject_bias = rng.normal(0.0, 0.6, size=n_subjects).astype(np.float32)
    subject_u = rng.normal(0.0, 0.45, size=(n_subjects, k)).astype(np.float32)
    item_v = rng.normal(0.0, 0.55, size=(n_items, k)).astype(np.float32)
    item_z = rng.normal(0.0, 0.7, size=n_items).astype(np.float32)

    rows: dict[str, list] = {
        "subject_id": [],
        "subject_content": [],
        "item_id": [],
        "benchmark": [],
        "condition": [],
        "item_content": [],
        "label": [],
    }
    for s_idx, subject_content in enumerate(subject_contents):
        for i_idx, item_id in enumerate(item_ids):
            logit = float(subject_bias[s_idx] + subject_u[s_idx].dot(item_v[i_idx]) + item_z[i_idx])
            p = 1.0 / (1.0 + np.exp(-logit))
            y = int(rng.random() < p)
            rows["subject_id"].append(f"subject_{s_idx:03d}")
            rows["subject_content"].append(subject_content)
            rows["item_id"].append(item_id)
            rows["benchmark"].append(f"fixture_benchmark_{i_idx % 3}")
            rows["condition"].append("zero-shot" if i_idx % 2 == 0 else "cot")
            rows["item_content"].append(
                f"Fixture question {i_idx}: choose the best answer for latent concept {i_idx % k}."
            )
            rows["label"].append(y)

    table = pa.table(rows)
    pq.write_table(table, out_dir / "joined.parquet")

    subject_to_id = {f"subject_{i:03d}": i for i in range(n_subjects)}
    subject_name_to_id = {normalize_subject(content): i for i, content in enumerate(subject_contents)}
    item_to_id = {item_id: i for i, item_id in enumerate(item_ids)}

    subject_state = {
        "subject_bias": torch.from_numpy(subject_bias),
        "subject_u": torch.from_numpy(subject_u),
        "fallback_bias": torch.tensor(float(subject_bias.mean()), dtype=torch.float32),
        "fallback_u": torch.from_numpy(subject_u.mean(axis=0).astype(np.float32)),
    }
    item_targets = {
        "item_v": torch.from_numpy(item_v),
        "item_z": torch.from_numpy(item_z),
    }

    torch.save(subject_state, stage1_dir / "subject_state.pt")
    torch.save(item_targets, stage1_dir / "item_targets.pt")
    (stage1_dir / "subject_to_id.json").write_text(json.dumps(subject_to_id, indent=2))
    (stage1_dir / "subject_name_to_id.json").write_text(json.dumps(subject_name_to_id, indent=2))
    (stage1_dir / "item_to_id.json").write_text(json.dumps(item_to_id, indent=2))
    (stage1_dir / "manifest.json").write_text(
        json.dumps(
            {
                "command": sys.argv,
                "fixture": True,
                "git_sha": _git_sha(),
                "seed": seed,
                "n_subjects": n_subjects,
                "n_items": n_items,
                "n_rows": len(rows["label"]),
                "k": k,
            },
            indent=2,
        )
    )

    print(f"Wrote fixture to {out_dir}")
    print(f"  rows={len(rows['label'])} subjects={n_subjects} items={n_items} k={k}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/fixtures/kfactor", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-subjects", type=int, default=5)
    parser.add_argument("--n-items", type=int, default=20)
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()
    main(args.out, args.seed, args.n_subjects, args.n_items, args.k)
