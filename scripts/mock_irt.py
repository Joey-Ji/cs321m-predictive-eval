"""Generate mock Stage-1 outputs so Stage-2 (content head) can develop in parallel.

Produces files in the same schema as scripts/fit_irt.py would, but with
synthetic values drawn from N(0, 1). Use this when:

  - Stage 1 hasn't produced real outputs yet (parallel development)
  - You want to test the Stage-2 + submission pipeline end-to-end before
    real IRT params are ready
  - You want to verify your Stage-2 code is robust to changes in Stage-1's
    actual numerical output

Reads:
  data/joined.parquet — to learn the set of unique subjects and items

Writes (same schema as scripts/fit_irt.py):
  data/irt/theta.pt           — float32 [n_subjects] ~ N(0, 1)
  data/irt/b.pt               — float32 [n_items]    ~ N(0, 1)
  data/irt/log_a.pt           — float32 [n_items]    = zeros (Rasch default)
  data/irt/subject_to_id.json — same as real
  data/irt/item_to_id.json    — same as real
  data/irt/fit_log.json       — marked {"model": "MOCK", "is_mock": true}

Usage:
    python scripts/mock_irt.py
    python scripts/mock_irt.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

NAME_LINE = re.compile(r"^\s*Name:\s*(.+?)\s*$", re.MULTILINE)


def normalize_subject(subject_content: str) -> str:
    if not subject_content:
        return ""
    m = NAME_LINE.search(subject_content)
    return m.group(1).strip().lower() if m else subject_content.strip().lower()


def main(joined_path: Path, out_dir: Path, seed: int, model: str, log_a_std: float) -> None:
    import numpy as np
    import pyarrow.parquet as pq
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {joined_path} (only subject + item columns) ...")
    table = pq.read_table(joined_path, columns=["subject_content", "item_id"])
    subj_keys = sorted({normalize_subject(s) for s in table.column("subject_content").to_pylist()})
    item_ids = sorted(set(table.column("item_id").to_pylist()))

    subject_to_id = {s: i for i, s in enumerate(subj_keys)}
    item_to_id = {it: i for i, it in enumerate(item_ids)}
    print(f"  subjects: {len(subject_to_id):,}   items: {len(item_to_id):,}   model: {model}")

    rng = np.random.default_rng(seed)
    theta = torch.from_numpy(rng.standard_normal(len(subject_to_id)).astype("float32"))
    b = torch.from_numpy(rng.standard_normal(len(item_to_id)).astype("float32"))
    if model == "2pl":
        log_a = torch.from_numpy((rng.standard_normal(len(item_to_id)) * log_a_std).astype("float32"))
    else:
        log_a = torch.zeros(len(item_to_id), dtype=torch.float32)

    torch.save(theta, out_dir / "theta.pt")
    torch.save(b, out_dir / "b.pt")
    torch.save(log_a, out_dir / "log_a.pt")
    (out_dir / "subject_to_id.json").write_text(json.dumps(subject_to_id, indent=2))
    (out_dir / "item_to_id.json").write_text(json.dumps(item_to_id, indent=2))
    (out_dir / "fit_log.json").write_text(
        json.dumps(
            {
                "model": f"MOCK_{model.upper()}",
                "is_mock": True,
                "seed": seed,
                "log_a_std": log_a_std if model == "2pl" else 0.0,
                "n_subjects": len(subject_to_id),
                "n_items": len(item_to_id),
            },
            indent=2,
        )
    )
    print(f"Done. Mock outputs in {out_dir}/   (do NOT submit a model trained on these as a real entry)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--joined", default="data/joined.parquet", type=Path)
    parser.add_argument("--out", default="data/irt", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", choices=("1pl", "2pl"), default="1pl",
                        help="1pl = log_a is zeros; 2pl = log_a ~ N(0, log_a_std).")
    parser.add_argument("--log-a-std", type=float, default=0.3,
                        help="Std for synthetic log_a values when --model 2pl. 0.3 ~ realistic 2PL variation.")
    args = parser.parse_args()
    main(args.joined, args.out, args.seed, args.model, args.log_a_std)
