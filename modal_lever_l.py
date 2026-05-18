"""Modal runner for Lever L priors, residual training, and split-faithful eval."""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "eval-comp-lever-l"
VOLUME_NAME = "eval-comp-data"
WORKDIR = "/root/eval_comp"
DATA_DIR = f"{WORKDIR}/data"
GPU = "T4"

LOCAL_ROOT = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2",
        "pyarrow>=15",
        "pandas>=2.0",
        "numpy>=1.26",
        "scikit-learn>=1.4",
        "tqdm>=4.66",
    )
    .add_local_dir(str(LOCAL_ROOT / "scripts"), remote_path=f"{WORKDIR}/scripts")
    .add_local_dir(str(LOCAL_ROOT / "src"), remote_path=f"{WORKDIR}/src")
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


def _run(cmd: list[str]) -> None:
    import os
    import subprocess

    os.chdir(WORKDIR)
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


@app.function(image=image, gpu=GPU, volumes={DATA_DIR: volume}, timeout=2 * 60 * 60)
def train_residual(
    seed: int = 0,
    max_rows: int = 1500,
    per_category: int = 300,
    epochs: int = 8,
    max_train_rows: int | None = None,
) -> None:
    cmd = [
        "python",
        "scripts/train_kfactor_residual.py",
        "--joined",
        "data/joined.parquet",
        "--stage1",
        "data/stage1/kfactor_k4",
        "--stage2",
        "data/stage2/kfactor_mpnet_mlp_v1",
        "--emb",
        "data/embeddings/mpnet_v1",
        "--out",
        "data/stage2/kfactor_mpnet_residual_v1",
        "--priors-out",
        "data/stage2/priors_v1",
        "--seed",
        str(seed),
        "--max-rows",
        str(max_rows),
        "--per-category",
        str(per_category),
        "--epochs",
        str(epochs),
        "--device",
        "cuda",
    ]
    if max_train_rows is not None:
        cmd.extend(["--max-train-rows", str(max_train_rows)])
    _run(cmd)
    volume.commit()


@app.function(image=image, gpu=GPU, volumes={DATA_DIR: volume}, timeout=90 * 60)
def eval_split(seeds: str = "0,1,2", max_rows: int = 1500, per_category: int = 300) -> None:
    _run(
        [
            "python",
            "scripts/eval_split_faithful.py",
            "--joined",
            "data/joined.parquet",
            "--stage1",
            "data/stage1/kfactor_k4",
            "--stage2",
            "data/stage2/kfactor_mpnet_mlp_v1",
            "--emb",
            "data/embeddings/mpnet_v1",
            "--residual",
            "data/stage2/kfactor_mpnet_residual_v1",
            "--seeds",
            seeds,
            "--max-rows",
            str(max_rows),
            "--per-category",
            str(per_category),
            "--device",
            "cuda",
            "--out",
            "data/stage2/kfactor_mpnet_residual_v1/split_faithful_eval.json",
        ]
    )
    volume.commit()


@app.local_entrypoint()
def main(
    stage: str = "all",
    seeds: str = "0,1,2",
    seed: int = 0,
    max_rows: int = 1500,
    per_category: int = 300,
    epochs: int = 8,
    max_train_rows: int | None = None,
) -> None:
    if stage not in ("train", "eval", "all"):
        raise SystemExit(f"unknown stage {stage!r}; expected train|eval|all")
    if stage in ("train", "all"):
        train_residual.remote(
            seed=seed,
            max_rows=max_rows,
            per_category=per_category,
            epochs=epochs,
            max_train_rows=max_train_rows,
        )
    if stage in ("eval", "all"):
        eval_split.remote(seeds=seeds, max_rows=max_rows, per_category=per_category)
    print(">>> done")
