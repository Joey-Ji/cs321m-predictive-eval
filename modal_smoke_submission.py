"""Modal smoke test that mimics the Codabench runtime.

Catches the kinds of bugs we hit in the first Codabench upload: predict()
raising because the encoder weights weren't in the platform's HF cache, or
because something at import time silently depended on network access.

Mimics the platform contract:
  - Linux/CPU container (no Mac MPS contamination).
  - Only the deps the submission can reasonably expect at runtime.
  - HF encoder weights pre-fetched into /app/hf_cache via the image build.
  - TRANSFORMERS_OFFLINE=1 + HF_HUB_OFFLINE=1 at runtime so any unexpected
    network call raises instead of silently downloading.
  - predict() exercised against real rows from data/joined.parquet (subjects
    and item text from the real distribution).

USAGE:
    modal run modal_smoke_submission.py                          # default zip + 20 rows
    modal run modal_smoke_submission.py --zip <path> --n 50      # explicit args

PASS criteria (all must hold over N rows):
    - predict() returns a Python float
    - 0 <= result <= 1
    - no exception
    - works with labeled=None, labeled=[], and labeled=<5 dicts>
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "eval-comp-smoke"
VOLUME_NAME = "eval-comp-data"
HF_CACHE = "/app/hf_cache"

LOCAL_ROOT = Path(__file__).resolve().parent


def _prefetch_encoder() -> None:
    """Pre-download the encoder into HF_CACHE — this is what the platform does."""
    import os

    os.environ["HF_HOME"] = HF_CACHE
    from sentence_transformers import SentenceTransformer

    SentenceTransformer("sentence-transformers/all-mpnet-base-v2", cache_folder=HF_CACHE)
    SentenceTransformer("BAAI/bge-large-en-v1.5", cache_folder=HF_CACHE)


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
        "huggingface-hub>=0.24",
    )
    .env({"HF_HOME": HF_CACHE})
    .run_function(_prefetch_encoder)
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
app = modal.App(APP_NAME)


@app.function(image=image, volumes={"/data": volume}, timeout=900)
def smoke(zip_bytes: bytes, n_rows: int = 20) -> None:
    import importlib.util
    import os
    import sys
    import tempfile
    import traceback
    import zipfile
    from pathlib import Path as P

    os.environ["HF_HOME"] = HF_CACHE
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    tmp = P(tempfile.mkdtemp())
    zpath = tmp / "sub.zip"
    zpath.write_bytes(zip_bytes)
    sub = tmp / "sub"
    sub.mkdir()
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(sub)

    print(f"[smoke] submission unzipped to {sub}")
    print(f"[smoke] HF_HOME={os.environ['HF_HOME']}  OFFLINE=1")
    print(f"[smoke] /app/hf_cache contents: {sorted(p.name for p in P(HF_CACHE).iterdir())[:5]}")

    sys.path.insert(0, str(sub))
    spec = importlib.util.spec_from_file_location("model", sub / "model.py")
    if spec is None or spec.loader is None:
        raise SystemExit("model.py spec failed")
    model = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(model)
    except Exception:
        print("[smoke] FAIL: model.py raised at import time")
        traceback.print_exc()
        raise

    print(f"[smoke] model.py imported (ENCODER_OK={getattr(model, 'ENCODER_OK', '?')})")

    import pyarrow.parquet as pq

    import random

    table = pq.read_table(
        "/data/joined.parquet",
        columns=["subject_content", "item_content", "benchmark", "condition", "label"],
    )
    n_total = table.num_rows
    rng = random.Random(0)
    # Sample from across the whole file (it's sorted by subject) so we exercise
    # multiple subjects/benchmarks/items, not just the first one.
    indices = rng.sample(range(n_total), k=min(n_rows * 4, n_total))
    rows = table.take(indices).to_pylist()
    seen: set[tuple[str, str, str]] = set()
    samples: list[dict] = []
    for r in rows:
        key = (str(r["subject_content"])[:80], str(r["benchmark"]), str(r["item_content"])[:80])
        if key in seen:
            continue
        seen.add(key)
        samples.append(r)
        if len(samples) >= n_rows:
            break
    bm = sorted({str(s["benchmark"]) for s in samples})
    sj = len({str(s["subject_content"])[:60] for s in samples})
    print(f"[smoke] sampled {len(samples)} rows  subjects={sj}  benchmarks={len(bm)}")

    fails = 0
    for i, row in enumerate(samples):
        inp = {
            "subject_content": str(row["subject_content"] or ""),
            "item_content": str(row["item_content"] or ""),
            "benchmark": str(row["benchmark"] or ""),
            "condition": str(row["condition"] or ""),
        }
        try:
            p = model.predict(inp)
        except Exception:
            print(f"  row {i}: predict() RAISED")
            traceback.print_exc()
            fails += 1
            continue
        ok_type = isinstance(p, float) and not isinstance(p, bool)
        ok_range = ok_type and 0.0 <= p <= 1.0
        flag = "OK" if (ok_type and ok_range) else "BAD"
        print(f"  row {i}: p={p!r}  type={type(p).__name__}  label={row['label']}  {flag}")
        if not (ok_type and ok_range):
            fails += 1

    print("[smoke] testing predict(labeled=None) and predict(labeled=[5 dicts])")
    inp = {k: str(samples[0][k] or "") for k in ("subject_content", "item_content", "benchmark", "condition")}
    for label_arg in (None, [], [{**{k: str(s[k] or "") for k in ("subject_content", "item_content", "benchmark", "condition")}, "label": int(s["label"])} for s in samples[:5]]):
        try:
            p = model.predict(inp, labeled=label_arg)
            ok = isinstance(p, float) and not isinstance(p, bool) and 0.0 <= p <= 1.0
            kind = type(label_arg).__name__ if label_arg is not None else "None"
            n = len(label_arg) if label_arg is not None else 0
            print(f"  labeled={kind}[{n}]  p={p!r}  {'OK' if ok else 'BAD'}")
            if not ok:
                fails += 1
        except Exception:
            print(f"  labeled={type(label_arg).__name__}: RAISED")
            traceback.print_exc()
            fails += 1

    if fails:
        raise SystemExit(f"SMOKE FAILED: {fails} issue(s) found")
    print("[smoke] PASS")


@app.local_entrypoint()
def main(zip: str = "submissions/v1_kfactor.zip", n: int = 20) -> None:
    path = (LOCAL_ROOT / zip).resolve() if not Path(zip).is_absolute() else Path(zip)
    if not path.exists():
        raise SystemExit(f"submission zip not found: {path}")
    print(f"[local] uploading {path.name} ({path.stat().st_size / 1024:.1f} KB) and running smoke ({n} rows)")
    smoke.remote(path.read_bytes(), n)
