"""Refit data/stage2/priors_v1/ with the current lever_l_utils.py code.

Mirrors train_kfactor_residual.py's prior path without the residual training:
load joined.parquet, tune kappas on a held-out item split, then write
full-data priors via write_prior_artifacts().
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lever_l_utils import (
    tune_priors,
    write_prior_artifacts,
)
from src.kfactor import validation_item_ids
from train_kfactor_residual import _load_joined_frame


def main(joined: Path, out_dir: Path, val_frac: float, seed: int) -> None:
    print(f"loading {joined}", flush=True)
    df = _load_joined_frame(joined)
    print(f"loaded rows={len(df)} subjects={df['subject_key'].nunique()}", flush=True)

    item_ids = sorted(df["item_id"].astype(str).unique().tolist())
    held_out = validation_item_ids(item_ids, val_frac=val_frac, seed=seed)
    train_df = df[~df["item_id"].isin(held_out)]
    val_df = df[df["item_id"].isin(held_out)]
    print(
        f"split val_frac={val_frac} seed={seed} "
        f"train_rows={len(train_df)} val_rows={len(val_df)}",
        flush=True,
    )

    print("tuning kappas on item-disjoint split", flush=True)
    _, kappas = tune_priors(train_df, val_df)
    print(f"tuned kappas: {json.dumps(kappas, indent=2)}", flush=True)

    print(f"writing full-data priors to {out_dir}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_prior_artifacts(df, kappas, out_dir)
    print("done.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--joined", default="data/joined.parquet", type=Path)
    parser.add_argument("--out", default="data/stage2/priors_v1", type=Path)
    parser.add_argument("--val-frac", default=0.1, type=float)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    main(args.joined, args.out, args.val_frac, args.seed)
