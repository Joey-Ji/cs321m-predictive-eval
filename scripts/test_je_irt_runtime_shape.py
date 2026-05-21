"""Check JE-IRT runtime uses platform-shaped inputs without subject_id."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from je_irt_utils import JEIRTModel


ROOT = Path(__file__).resolve().parent.parent


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _build_runtime_dir(path: Path) -> None:
    shutil.copy(ROOT / "submissions" / "v1_kfactor" / "model.py", path / "model.py")
    _write_json(
        path / "head_meta.json",
        {
            "encoder": "dummy",
            "embedding_dim": 4,
            "head_type": "linear",
            "hidden": 4,
            "in_dim": 6,
            "k": 1,
            "max_chars": 4000,
            "out_dim": 2,
            "representation_version": "item_text_plus_side_features_v1",
            "side_feature_dim": 2,
        },
    )
    _write_json(path / "target_scaler.json", {"mean": [0.0, 0.0], "std": [1.0, 1.0]})
    _write_json(
        path / "side_feature_meta.json",
        {"benchmark": {}, "benchmark_dim": 1, "condition": {}, "condition_dim": 1, "side_feature_dim": 2},
    )
    torch.save(torch.nn.Linear(6, 2).state_dict(), path / "head.pt")
    torch.save(
        {
            "subject_bias": torch.zeros(1, dtype=torch.float32),
            "subject_u": torch.zeros(1, 1, dtype=torch.float32),
            "fallback_bias": torch.tensor(0.0, dtype=torch.float32),
            "fallback_u": torch.zeros(1, dtype=torch.float32),
        },
        path / "subject_state.pt",
    )
    _write_json(path / "subject_name_to_id.json", {"alpha": 0})

    je_dir = path / "je_irt"
    je_dir.mkdir()
    _write_json(
        je_dir / "config.json",
        {
            "dim": 4,
            "dropout": 0.0,
            "encoder": "dummy",
            "encoder_dim": 4,
            "hidden": 4,
            "max_chars": 4000,
            "subject_key": "normalize_subject(subject_content)",
        },
    )
    _write_json(je_dir / "subject_to_id.json", {"alpha": 0})
    model = JEIRTModel(n_subjects=1, in_dim=4, hidden=4, dim=4, dropout=0.0)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
    torch.save(model.state_dict(), je_dir / "je_irt_head.pt")


def _load_model(path: Path):
    spec = importlib.util.spec_from_file_location("je_irt_runtime_shape_model", path / "model.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to build module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    old_dummy = os.environ.get("V1_KFACTOR_DUMMY_ENCODER")
    os.environ["V1_KFACTOR_DUMMY_ENCODER"] = "1"
    with tempfile.TemporaryDirectory(prefix="je_irt_runtime_shape_") as tmp:
        runtime_dir = Path(tmp)
        _build_runtime_dir(runtime_dir)
        module = _load_model(runtime_dir)
        platform_input = {
            "benchmark": "bench",
            "condition": "cond",
            "subject_content": "Name: Alpha",
            "item_content": "What is 2+2?",
        }
        p = module.predict(platform_input)
        fallback = module._prior_probability_for_input(platform_input)
        if not module.JE_IRT_ACTIVE or module.PRIOR_ONLY:
            raise AssertionError("JE-IRT did not activate in runtime-shape check")
        if abs(p - 0.5) > 1e-6:
            raise AssertionError(f"expected JE-IRT probability near 0.5, got {p}")
        if abs(p - fallback) < 1e-3:
            raise AssertionError("platform-shaped input fell through to prior fallback")
        missing = dict(platform_input, subject_content="Name: Missing")
        if abs(module.predict(missing) - module._prior_probability_for_input(missing)) > 1e-12:
            raise AssertionError("missing JE-IRT subject did not fall back to prior")
    if old_dummy is None:
        os.environ.pop("V1_KFACTOR_DUMMY_ENCODER", None)
    else:
        os.environ["V1_KFACTOR_DUMMY_ENCODER"] = old_dummy
    print("PASS: JE-IRT runtime shape check passed")


if __name__ == "__main__":
    main()
