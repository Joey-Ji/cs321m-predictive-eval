"""Modal app for training K-factor head with advanced statistical features.

This extends the Stage 2 pipeline to include benchmark/condition statistics,
interaction effects, and Stage 1 parameter distributions as additional features.

------------------------------------------------------------------------------
ONE-TIME SETUP — push local data to the volume (if not already done):

    modal volume create eval-comp-data || true
    modal volume put eval-comp-data data/joined.parquet   joined.parquet
    modal volume put eval-comp-data data/items.parquet    items.parquet
    modal volume put eval-comp-data data/subjects.parquet subjects.parquet
    modal volume put eval-comp-data data/stage1           stage1

RUN THE PIPELINE:

    # Full pipeline: encode + train with advanced features
    modal run modal_advanced_features.py

    # Just encode embeddings (GPU)
    modal run modal_advanced_features.py --stage encode

    # Just train with advanced features (GPU recommended)
    modal run modal_advanced_features.py --stage train

    # Custom hyperparameters
    modal run modal_advanced_features.py --stage train --head mlp --hidden 256 --epochs 50 --lr 0.001

PULL RESULTS BACK LOCALLY:

    modal volume get eval-comp-data embeddings ./data/embeddings
    modal volume get eval-comp-data stage2     ./data/stage2

INSPECT VOLUME CONTENTS:

    modal volume ls eval-comp-data
    modal volume ls eval-comp-data stage2/kfactor_mpnet_advanced_v1
------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "eval-comp-advanced-features"
VOLUME_NAME = "eval-comp-data"
WORKDIR = "/root/eval_comp"
DATA_DIR = f"{WORKDIR}/data"
ENCODER = "sentence-transformers/all-mpnet-base-v2"
EMB_DIR = "data/embeddings/mpnet_v1"
GPU = "T4"  # Options: T4, A10G, A100

LOCAL_ROOT = Path(__file__).resolve().parent


def _cache_encoder() -> None:
    """Bake the encoder weights into the image so cold starts skip the HF download."""
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(ENCODER)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2",
        "sentence-transformers>=3.0",
        "transformers>=4.41",
        "pyarrow>=15",
        "pandas>=2.0",
        "numpy>=1.26",
        "scikit-learn>=1.4",
        "tqdm>=4.66",
        "huggingface-hub>=0.24",
    )
    .run_function(_cache_encoder)
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


@app.function(
    image=image,
    gpu=GPU,
    volumes={DATA_DIR: volume},
    timeout=2 * 60 * 60,
)
def encode_items(
    batch: int = 128,
    max_chars: int = 4000,
    encoder: str = ENCODER,
    out: str = EMB_DIR,
) -> None:
    """Encode item content to embeddings using GPU acceleration."""
    _run(
        [
            "python",
            "scripts/encode_items.py",
            "--joined", "data/joined.parquet",
            "--out", out,
            "--encoder", encoder,
            "--batch", str(batch),
            "--max-chars", str(max_chars),
        ]
    )
    volume.commit()


@app.function(
    image=image,
    gpu=GPU,  # Use GPU for faster training
    volumes={DATA_DIR: volume},
    timeout=60 * 60,  # 1 hour timeout
)
def train_advanced_head(
    head: str = "mlp",
    hidden: int = 256,
    emb: str = EMB_DIR,
    out: str = "data/stage2/kfactor_mpnet_advanced_v1",
    epochs: int = 30,
    lr: float = 1e-3,
    val_frac: float = 0.1,
    seed: int = 0,
) -> None:
    """Train K-factor head with advanced statistical features using GPU acceleration."""
    _run(
        [
            "python",
            "scripts/train_kfactor_head_advanced.py",
            "--joined", "data/joined.parquet",
            "--stage1", "data/stage1/kfactor_k4",
            "--emb", emb,
            "--out", out,
            "--head-type", head,
            "--hidden", str(hidden),
            "--epochs", str(epochs),
            "--lr", str(lr),
            "--val-frac", str(val_frac),
            "--seed", str(seed),
        ]
    )
    volume.commit()


@app.local_entrypoint()
def main(
    stage: str = "all",
    head: str = "mlp",
    hidden: int = 256,
    encoder: str = ENCODER,
    emb_out: str = EMB_DIR,
    emb: str = EMB_DIR,
    batch: int = 128,
    max_chars: int = 4000,
    epochs: int = 30,
    lr: float = 1e-3,
    out: str = "data/stage2/kfactor_mpnet_advanced_v1",
    val_frac: float = 0.1,
    seed: int = 0,
) -> None:
    """Run the requested stage(s). `stage` is one of: encode | train | all."""
    if stage not in ("encode", "train", "all"):
        raise SystemExit(f"unknown stage {stage!r}; expected encode|train|all")

    if stage in ("encode", "all"):
        print(">>> Encoding items to embeddings (GPU)...")
        encode_items.remote(batch=batch, max_chars=max_chars, encoder=encoder, out=emb_out)

    if stage in ("train", "all"):
        print(">>> Training K-factor head with advanced features (GPU)...")
        train_advanced_head.remote(
            head=head,
            hidden=hidden,
            emb=emb,
            out=out,
            epochs=epochs,
            lr=lr,
            val_frac=val_frac,
            seed=seed,
        )

    print(">>> Done! Results saved to Modal volume.")
    print(f">>> To download: modal volume get {VOLUME_NAME} stage2 ./data/stage2")
