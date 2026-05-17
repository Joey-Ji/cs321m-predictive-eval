"""Modal app for the Stage 2 K-factor pipeline.

Runs the three existing scripts in `scripts/` against data stored in a Modal
Volume:

    scripts/encode_items.py        (GPU, encodes ~70K items with mpnet-base-v2)
    scripts/train_kfactor_head.py  (CPU, trains the linear or MLP head)
    scripts/evaluate_stage2.py     (CPU, offline log-likelihood + AUC)

The Modal Volume `eval-comp-data` mirrors the local `data/` directory layout,
so the existing scripts run unmodified with their default arg paths.

------------------------------------------------------------------------------
ONE-TIME SETUP — push local data to the volume:

    modal volume create eval-comp-data || true
    modal volume put eval-comp-data data/joined.parquet   joined.parquet
    modal volume put eval-comp-data data/items.parquet    items.parquet
    modal volume put eval-comp-data data/subjects.parquet subjects.parquet
    modal volume put eval-comp-data data/stage1           stage1

RUN THE PIPELINE:

    modal run modal_stage2.py                  # encode + train + eval
    modal run modal_stage2.py --stage encode   # just encode
    modal run modal_stage2.py --stage encode --encoder BAAI/bge-large-en-v1.5 --emb-out data/embeddings/bge_large_v1
    modal run modal_stage2.py --stage train    # just train the default linear head
    modal run modal_stage2.py --stage train --head mlp --out data/stage2/kfactor_mpnet_mlp_v1
    modal run modal_stage2.py --stage train --head mlp --emb data/embeddings/bge_large_v1 --out data/stage2/kfactor_bge_mlp_v1
    modal run modal_stage2.py --stage eval     # just evaluate the default linear head
    modal run modal_stage2.py --stage eval --out data/stage2/kfactor_mpnet_mlp_v1

PULL RESULTS BACK LOCALLY (after running):

    modal volume get eval-comp-data embeddings ./data/embeddings
    modal volume get eval-comp-data stage2     ./data/stage2

INSPECT VOLUME CONTENTS:

    modal volume ls eval-comp-data
    modal volume ls eval-comp-data stage1/kfactor_k4
------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "eval-comp-stage2"
VOLUME_NAME = "eval-comp-data"
WORKDIR = "/root/eval_comp"
DATA_DIR = f"{WORKDIR}/data"
ENCODER = "sentence-transformers/all-mpnet-base-v2"
EMB_DIR = "data/embeddings/mpnet_v1"
GPU = "T4"

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
    volumes={DATA_DIR: volume},
    timeout=30 * 60,
)
def train_head(
    head: str = "linear",
    hidden: int = 256,
    emb: str = EMB_DIR,
    out: str = "data/stage2/kfactor_mpnet_linear_v1",
    epochs: int = 200,
    lr: float = 1e-3,
    val_frac: float = 0.1,
    seed: int = 0,
) -> None:
    _run(
        [
            "python",
            "scripts/train_kfactor_head.py",
            "--stage1", "data/stage1/kfactor_k4",
            "--emb", emb,
            "--out", out,
            "--head", head,
            "--hidden", str(hidden),
            "--epochs", str(epochs),
            "--lr", str(lr),
            "--val-frac", str(val_frac),
            "--seed", str(seed),
        ]
    )
    volume.commit()


@app.function(
    image=image,
    volumes={DATA_DIR: volume},
    timeout=30 * 60,
)
def evaluate(
    stage2: str = "data/stage2/kfactor_mpnet_linear_v1",
    emb: str = EMB_DIR,
    split: str = "item-cold",
    val_frac: float = 0.1,
    seed: int = 0,
) -> None:
    _run(
        [
            "python",
            "scripts/evaluate_stage2.py",
            "--joined", "data/joined.parquet",
            "--stage1", "data/stage1/kfactor_k4",
            "--stage2", stage2,
            "--emb", emb,
            "--split", split,
            "--val-frac", str(val_frac),
            "--seed", str(seed),
        ]
    )
    volume.commit()


@app.local_entrypoint()
def main(
    stage: str = "all",
    head: str = "linear",
    hidden: int = 256,
    encoder: str = ENCODER,
    emb_out: str = EMB_DIR,
    emb: str = EMB_DIR,
    batch: int = 128,
    max_chars: int = 4000,
    epochs: int = 200,
    lr: float = 1e-3,
    out: str = "data/stage2/kfactor_mpnet_linear_v1",
    split: str = "item-cold",
    val_frac: float = 0.1,
    seed: int = 0,
) -> None:
    """Run the requested stage(s). `stage` is one of: encode | train | eval | all."""
    if stage not in ("encode", "train", "eval", "all"):
        raise SystemExit(f"unknown stage {stage!r}; expected encode|train|eval|all")
    if stage in ("encode", "all"):
        print(">>> encode_items")
        encode_items.remote(batch=batch, max_chars=max_chars, encoder=encoder, out=emb_out)
    if stage in ("train", "all"):
        print(">>> train_head")
        train_head.remote(
            head=head,
            hidden=hidden,
            emb=emb,
            out=out,
            epochs=epochs,
            lr=lr,
            val_frac=val_frac,
            seed=seed,
        )
    if stage in ("eval", "all"):
        print(">>> evaluate")
        evaluate.remote(stage2=out, emb=emb, split=split, val_frac=val_frac, seed=seed)
    print(">>> done")
