"""In-process smoke test for the prior-only online-intercept submission zip.

Run:
    python scripts/smoke_online_intercept.py --zip submissions/v1_kfactor_online_intercept.zip
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import sys
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType


def _load_model_from_zip(zip_path: Path, tmpdir: Path) -> tuple[ModuleType, list[str]]:
    with zipfile.ZipFile(zip_path) as zf:
        manifest = sorted(zf.namelist())
        zf.extractall(tmpdir)
    model_path = tmpdir / "model.py"
    if not model_path.exists():
        raise FileNotFoundError(f"model.py not found in {zip_path}")

    previous_dummy = os.environ.get("V1_KFACTOR_DUMMY_ENCODER")
    os.environ["V1_KFACTOR_DUMMY_ENCODER"] = "1"
    sys.path.insert(0, str(tmpdir))
    try:
        spec = importlib.util.spec_from_file_location("online_intercept_smoke_model", model_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not import {model_path}")
        module = importlib.util.module_from_spec(spec)
        with contextlib.redirect_stdout(sys.stderr):
            spec.loader.exec_module(module)
        return module, manifest
    finally:
        try:
            sys.path.remove(str(tmpdir))
        except ValueError:
            pass
        if previous_dummy is None:
            os.environ.pop("V1_KFACTOR_DUMMY_ENCODER", None)
        else:
            os.environ["V1_KFACTOR_DUMMY_ENCODER"] = previous_dummy


def _prior_rows(priors: dict) -> list[tuple[str, str, str, float]]:
    sep = priors["key_sep"]
    rows: list[tuple[str, str, str, float]] = []
    for key, value in priors["subject_category"].items():
        parts = str(key).split(sep)
        if len(parts) == 3:
            rows.append((parts[0], parts[1], parts[2], float(value)))
    if not rows:
        raise ValueError("runtime_priors.json has no subject_category rows")
    return rows


def _input(subject_content: str, benchmark: str, condition: str, item: str = "Online intercept smoke item") -> dict:
    return {
        "subject_content": subject_content,
        "item_content": item,
        "benchmark": benchmark,
        "condition": condition,
    }


def _labeled(
    subject_content: str,
    benchmark: str,
    condition: str,
    label: int,
    idx: int,
) -> dict:
    return {
        **_input(subject_content, benchmark, condition, item=f"Online intercept labeled item {idx}"),
        "label": label,
    }


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _predict(model: ModuleType, row: dict, labeled: list[dict] | None = None) -> float:
    return float(model.predict(row, labeled=labeled))


def run_smoke(zip_path: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="online-intercept-smoke-") as tmp:
        tmpdir = Path(tmp)
        model, manifest = _load_model_from_zip(zip_path, tmpdir)
        priors = json.loads((tmpdir / "runtime_priors.json").read_text())
        rows = _prior_rows(priors)
        subject, benchmark, condition, prior_p = min(rows, key=lambda row: abs(row[3] - 0.5))
        subject_content = f"Name: {subject}"
        row = _input(subject_content, benchmark, condition)

        _assert(bool(getattr(model, "PRIOR_ONLY", False)), "PRIOR_ONLY is not enabled")
        lam = float(getattr(model, "ONLINE_INTERCEPT_LAM"))
        clip = float(getattr(model, "ONLINE_INTERCEPT_CLIP"))
        _assert(lam >= 0.0, f"invalid ONLINE_INTERCEPT_LAM={lam}")
        _assert(clip >= 0.0, f"invalid ONLINE_INTERCEPT_CLIP={clip}")

        prior_direct = float(model._clip_prob(model._sigmoid(model._raw_logit(row))))
        empty_pred = _predict(model, row, labeled=[])
        _assert(abs(empty_pred - prior_direct) <= 1e-12, "empty labels changed the prior prediction")

        positive_labels = [_labeled(subject_content, benchmark, condition, 1, i) for i in range(5)]
        positive_pred = _predict(model, row, labeled=positive_labels)
        _assert(positive_pred > empty_pred, "all-positive labels did not increase p")

        other_subjects = [candidate for candidate in rows if candidate[0] != subject]
        negative_labels = []
        for i, candidate in enumerate(other_subjects[:5]):
            negative_labels.append(_labeled(f"Name: {candidate[0]}", candidate[1], candidate[2], 0, i))
        while len(negative_labels) < 5:
            negative_labels.append(_labeled(f"Name: unrelated-{len(negative_labels)}", benchmark, condition, 0, len(negative_labels)))
        negative_pred = _predict(model, row, labeled=negative_labels)
        _assert(negative_pred < empty_pred, "all-negative labels did not decrease p")

        n_mixed = 20
        n_positive = round(n_mixed * prior_p)
        mixed_labels = [
            _labeled(subject_content, benchmark, condition, 1 if i < n_positive else 0, i)
            for i in range(n_mixed)
        ]
        mixed_delta = float(model._online_intercept_delta(mixed_labels))
        mixed_pred = _predict(model, row, labeled=mixed_labels)
        _assert(abs(mixed_delta) <= 0.025, f"mixed labels produced a large delta: {mixed_delta}")
        _assert(abs(mixed_pred - empty_pred) <= 0.01, "mixed labels moved p too far from the prior")

        cold_row = _input("Name: online-intercept-cold-subject", benchmark, condition)
        cold_pred = _predict(model, cold_row, labeled=[])
        global_p = float(model._clip_prob(float(priors["global"])))
        _assert(abs(cold_pred - global_p) <= 1e-12, "cold subject did not fall back to global p")

        bounded_delta = float(
            model._online_intercept_delta(
                [_labeled("Name: online-intercept-cold-subject", benchmark, condition, 1, i) for i in range(25)]
            )
        )
        _assert(abs(bounded_delta) <= clip + 1e-12, "delta exceeded CLIP")

        return {
            "zip": str(zip_path),
            "prior_only": bool(model.PRIOR_ONLY),
            "online_intercept_lam": lam,
            "online_intercept_clip": clip,
            "contains_labeling_py": "labeling.py" in manifest,
            "prior_row": {
                "subject": subject,
                "benchmark": benchmark,
                "condition": condition,
                "prior_p": prior_p,
            },
            "checks": {
                "empty_pred": empty_pred,
                "positive_pred": positive_pred,
                "negative_pred": negative_pred,
                "mixed_delta": mixed_delta,
                "mixed_pred": mixed_pred,
                "cold_pred": cold_pred,
                "bounded_delta": bounded_delta,
            },
        }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default="submissions/v1_kfactor_online_intercept.zip", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload = run_smoke(args.zip)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
