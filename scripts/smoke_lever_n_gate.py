"""In-process smoke for the Lever N sparse-cell gate.

The script unpacks a submission zip, imports model.py with the dummy encoder,
selects representative runtime-prior cells by count, and prints the logits and
gate values that determine predict().
"""

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
from types import ModuleType


def _import_model(model_path: Path) -> ModuleType:
    os.environ["V1_KFACTOR_DUMMY_ENCODER"] = "1"
    sys.path.insert(0, str(model_path.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    spec = importlib.util.spec_from_file_location("lever_n_smoke_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _split_prior_key(model: ModuleType, key: str) -> tuple[str, str, str]:
    parts = key.split(model.PRIOR_KEY_SEP)
    if len(parts) != 3:
        raise ValueError(f"bad subject_category key: {key!r}")
    return parts[0], parts[1], parts[2]


def _pick_cells(model: ModuleType) -> list[tuple[str, str]]:
    counts = model.PRIORS["counts"]["subject_category"]
    dense = sorted((k, n) for k, n in counts.items() if int(n) >= 20)
    sparse = sorted((k, n) for k, n in counts.items() if int(n) < 5)
    mid = sorted((k, n) for k, n in counts.items() if 5 <= int(n) < 20)
    selected: list[tuple[str, str]] = []
    for label, bucket in (("dense", dense), ("dense", dense[1:]), ("mid", mid), ("sparse", sparse), ("sparse", sparse[1:])):
        if bucket:
            selected.append((label, bucket[0][0]))
    return selected[:5]


def _input_for_cell(model: ModuleType, key: str) -> dict[str, str]:
    subject_key, benchmark, condition = _split_prior_key(model, key)
    return {
        "subject_content": f"Name: {subject_key}\n",
        "item_content": f"Lever N smoke item for {benchmark} {condition}.",
        "benchmark": benchmark,
        "condition": condition,
    }


def _describe(model: ModuleType, label: str, input_row: dict[str, str]) -> dict[str, float | int | str]:
    subject_key = model._normalize_subject(input_row["subject_content"])
    benchmark = input_row["benchmark"]
    condition = input_row["condition"]
    prior_p = model._prior_values(subject_key, benchmark, condition)[-1]
    prior_logit = model._logit_prob(prior_p)
    kfactor_logit = model._kfactor_base_logit(input_row)
    cell_n = model._cell_count(subject_key, benchmark, condition)
    gate = model._reliability_gate(cell_n)
    final_p = model.predict(input_row, labeled=[])
    expected_logit = (1.0 - gate) * prior_logit + gate * kfactor_logit
    expected_p = model._clip_prob(model._sigmoid(expected_logit))
    return {
        "label": label,
        "subject_key": subject_key,
        "benchmark": benchmark,
        "condition": condition,
        "prior_logit": float(prior_logit),
        "kfactor_logit": float(kfactor_logit),
        "cell_n": int(cell_n),
        "gate": float(gate),
        "final_p": float(final_p),
        "expected_p": float(expected_p),
        "abs_diff": float(abs(final_p - expected_p)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        with zipfile.ZipFile(args.zip) as zf:
            zf.extractall(tmp)
        model = _import_model(tmp / "model.py")
        rows = []
        for label, key in _pick_cells(model):
            rows.append(_describe(model, label, _input_for_cell(model, key)))
        rows.append(
            _describe(
                model,
                "unknown",
                {
                    "subject_content": "Name: lever-n-unknown-subject\n",
                    "item_content": "Lever N smoke item for an unknown subject.",
                    "benchmark": "rewardbench",
                    "condition": "subset=alpacaeval-easy",
                },
            )
        )

        payload = {
            "zip": str(args.zip),
            "PRIOR_ONLY": bool(model.PRIOR_ONLY),
            "T_low": int(model.RELIABILITY_GATE_T_LOW),
            "T_high": int(model.RELIABILITY_GATE_T_HIGH),
            "rows": rows,
            "max_abs_diff": max((float(row["abs_diff"]) for row in rows), default=math.nan),
        }
        text = json.dumps(payload, indent=2, sort_keys=True)
        print(text)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
