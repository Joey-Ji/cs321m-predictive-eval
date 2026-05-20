"""Smoke-test the scalar item residual path in a submission ZIP."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any


def _import_model(submission_dir: Path):
    spec = importlib.util.spec_from_file_location("item_residual_smoke_model", submission_dir / "model.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import {submission_dir / 'model.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sigmoid(logit: float) -> float:
    if logit >= 0:
        z = math.exp(-logit)
        return 1.0 / (1.0 + z)
    z = math.exp(logit)
    return z / (1.0 + z)


def _clip_prob(p: float) -> float:
    return float(max(min(p, 0.98), 0.02))


def _sample_inputs(joined: Path, n: int) -> list[dict[str, str]]:
    import pandas as pd

    df = pd.read_parquet(
        joined,
        columns=["subject_content", "item_content", "benchmark", "condition"],
    ).head(max(n, 1))
    return [
        {key: "" if value is None else str(value) for key, value in row.items()}
        for row in df.to_dict(orient="records")
    ]


def main(zip_path: Path, joined: Path, out: Path, sample_size: int, dummy_encoder: bool) -> None:
    previous_dummy = os.environ.get("V1_KFACTOR_DUMMY_ENCODER")
    if dummy_encoder:
        os.environ["V1_KFACTOR_DUMMY_ENCODER"] = "1"
    try:
        with tempfile.TemporaryDirectory(prefix="item-residual-smoke-") as tmp_str:
            tmp = Path(tmp_str)
            sub = tmp / "sub"
            sub.mkdir()
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(sub)
            sys.path.insert(0, str(sub))
            model = _import_model(sub)
            samples = _sample_inputs(joined, sample_size)
            if not samples:
                raise ValueError("no sample inputs available")

            dense = samples[0]
            prior_logit = float(model._raw_logit(dense))
            delta = float(model._item_residual_delta(dense))
            weight_w = float(getattr(model, "ITEM_RESIDUAL_WEIGHT", 0.0))
            expected = _clip_prob(_sigmoid(prior_logit + weight_w * delta))
            actual = model.predict(dense, labeled=[{**dense, "label": 1}])
            if not isinstance(actual, float) or not math.isfinite(actual):
                raise AssertionError(f"predict returned invalid value {actual!r}")
            if abs(actual - expected) > 1e-6:
                raise AssertionError(f"PRIOR_ONLY composition mismatch: actual={actual} expected={expected}")

            unseen_item = dict(dense)
            unseen_item["item_content"] = dense["item_content"] + "\n\nSynthetic cold-start variant for residual smoke."
            unseen_delta = float(model._item_residual_delta(unseen_item))
            if not math.isfinite(unseen_delta):
                raise AssertionError("unseen item delta is not finite")

            cold_subject = dict(dense)
            cold_subject["subject_content"] = "Name: definitely-unseen-subject-for-item-residual-smoke"
            cold_subject["benchmark"] = "definitely-unseen-benchmark"
            cold_subject["condition"] = "definitely-unseen-condition"
            cold_delta = float(model._item_residual_delta(cold_subject))
            cold_pred = model.predict(cold_subject, labeled=[{**dense, "label": 0}])
            if not math.isfinite(cold_delta) or not isinstance(cold_pred, float):
                raise AssertionError("cold subject/item path failed")

            deltas = [float(model._item_residual_delta(row)) for row in samples]
            clip = float(getattr(model, "ITEM_RESIDUAL_DELTA_CLIP", 0.25))
            if any((not math.isfinite(value)) or abs(value) > clip + 1e-8 for value in deltas):
                raise AssertionError(f"delta bound failed for clip={clip}: {deltas}")

            payload: dict[str, Any] = {
                "zip": str(zip_path),
                "prior_only": bool(getattr(model, "PRIOR_ONLY", False)),
                "item_residual_ok": bool(getattr(model, "ITEM_RESIDUAL_OK", False)),
                "weight_w": weight_w,
                "runtime_delta_clip": clip,
                "dense_prior_only_prob": _clip_prob(_sigmoid(prior_logit)),
                "dense_delta": delta,
                "dense_prediction": float(actual),
                "unseen_item_delta": unseen_delta,
                "cold_subject_delta": cold_delta,
                "cold_subject_prediction": float(cold_pred),
                "sample_delta_min": float(min(deltas)),
                "sample_delta_max": float(max(deltas)),
                "sample_size": int(len(samples)),
                "dummy_encoder": bool(dummy_encoder),
            }
            if not payload["prior_only"]:
                raise AssertionError("PRIOR_ONLY is not enabled")
            if weight_w != 0.0 and not payload["item_residual_ok"]:
                raise AssertionError("item residual has nonzero weight but did not load")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    finally:
        if previous_dummy is None:
            os.environ.pop("V1_KFACTOR_DUMMY_ENCODER", None)
        else:
            os.environ["V1_KFACTOR_DUMMY_ENCODER"] = previous_dummy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default="submissions/v1_kfactor_item_residual_v1.zip", type=Path)
    parser.add_argument("--joined", default="data/joined.parquet", type=Path)
    parser.add_argument("--out", default="reports/item_residual_smoke.json", type=Path)
    parser.add_argument("--sample-size", default=128, type=int)
    parser.add_argument(
        "--dummy-encoder",
        action="store_true",
        help="Use the deterministic dummy encoder for fast shape/bounds smoke testing.",
    )
    args = parser.parse_args()
    main(args.zip, args.joined, args.out, args.sample_size, args.dummy_encoder)

