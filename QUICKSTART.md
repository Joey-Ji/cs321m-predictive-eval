# QUICKSTART

A walkthrough for getting from a fresh clone of this repo to a working
Codabench submission ZIP for the **Predictive AI Evaluation Challenge**,
without redoing the heavy training that we already ran on Modal.

`README.md` documents the *competition* contract (input schema, runtime
policy, GPU tiers). This file is the *project* guide: how to use what we
have, what's pushed, what isn't, and what to run.

---

## TL;DR

If you only want to rebuild the current `v1_kfactor` submission ZIP from
pre-trained artifacts that *someone else on the team* has produced:

```bash
uv sync                                    # 1. install deps
make data                                  # 2. download HF dataset (~600 MB)
make kfactor-export                        # 3. export Stage 1 from committed parquets
# 4. drop pre-trained Stage 2 artifacts into data/stage2/kfactor_mpnet_linear_v1/
#    (see "Stage 2 artifacts" below)
python scripts/build_submission.py v1_kfactor   # 5. build the ZIP
modal run modal_smoke_submission.py \
  --zip submissions/v1_kfactor.zip --n 20  # 6. smoke gate before any upload
```

You then upload `submissions/v1_kfactor.zip` to Codabench (1 scored
submission per UTC day per team).

If you need to reproduce Stage 2 from scratch on your own Modal account,
see the **Full reproduction** section.

---

## 1. What's in this repo (pushed) vs local-only

Git ignores `data/`, `*.pt`, `*.npy`, `*.bin`, `*.safetensors`, `*.zip`,
`.venv/`, `.hf_cache/`, `.library/` (see `.gitignore`). So roughly:

### Pushed (you get on `git clone`)

- **All source code**:
  - `scripts/` — Stage 1 training, encoding, Stage 2 head training,
    evaluation, submission build, local smoke test.
  - `src/` — feature encoding, K-factor model, validation helpers.
  - `stage_1/k_factor_irt/` — Yiheng's K=4 MIRT trainer
    (`fit_k_factor_irt.py`) plus a 2-layer MLP variant.
  - `submissions/v1_kfactor/model.py`, `submissions/v1_irt/model.py`
    — the live submission runtimes (the things Codabench actually executes).
  - `tests/`, `templates/`, `sample_code_submission/`.
- **Build glue**: `Makefile`, `pyproject.toml`, `uv.lock`,
  `.python-version`, `.github/workflows/contract.yml`.
- **Modal scripts**: `modal_stage2.py` (encode/train/eval),
  `modal_smoke_submission.py` (network-isolated upload gate).
- **Stage 1 K-factor handoff artifacts** (the one place we did commit
  trained outputs, because they're small and serve as the contract bridge
  between Yiheng's Stage 1 and the Stage 2 head):
  - `stage_1/k_factor_irt/artifacts/k4_full_train/subject_capabilities.parquet`
    (55 KB, 909 subjects)
  - `stage_1/k_factor_irt/artifacts/k4_full_train/item_parameters.parquet`
    (4.0 MB, 70,873 items)
  - `stage_1/k_factor_irt/artifacts/k4_full_train/manifest.json`
- **Competition spec**: `Predictive Evaluation Challenge.pdf`.

### Not pushed (must be downloaded or reproduced)

| Path | What it is | How to get it |
|---|---|---|
| `data/joined.parquet`, `data/items.parquet`, `data/responses.parquet`, `data/subjects.parquet`, `data/benchmarks.parquet` | Public HF training data (~600 MB joined) | `make data` |
| `data/stage1/kfactor_k4/{subject_state.pt, subject_name_to_id.json, item_targets.pt, item_to_id.json, subject_to_id.json, manifest.json}` | Stage 1 runtime tensors, derived from the committed parquets above | `make kfactor-export` |
| `data/embeddings/mpnet_v1/{item_embeddings.npy, item_side_features.npy}` | mpnet-base-v2 embeddings of all 70,873 items + benchmark/condition one-hot side features | Modal: `modal run modal_stage2.py --stage encode` |
| `data/stage2/kfactor_mpnet_linear_v1/{head.pt, head_meta.json, target_scaler.json, side_feature_meta.json, calibration.json, metrics.json}` | Trained linear head + calibration | Modal: `modal run modal_stage2.py --stage train` |
| `submissions/v1_kfactor.zip` | Uploadable ZIP | `python scripts/build_submission.py v1_kfactor` |
| `.hf_cache/` | Local HF model cache | Populated on first `SentenceTransformer(...)` call |

The Stage 1 K=4 IRT training itself (the step that *produced*
`subject_capabilities.parquet` and `item_parameters.parquet`) is **not
something you need to re-run** — those parquets are committed under
`stage_1/k_factor_irt/artifacts/k4_full_train/` and `make kfactor-export`
reads them. The trainer (`stage_1/k_factor_irt/fit_k_factor_irt.py`) is
checked in for transparency / re-use, not as part of the standard reproduction
path.

---

## 2. Prerequisites

- Python `3.12` (pinned in `.python-version`).
- [`uv`](https://github.com/astral-sh/uv) for env / dep management.
- [Modal](https://modal.com) account, only if you need to retrain Stage 2
  or run the upload smoke gate. Both encoding and the smoke gate use the
  `eval-comp-data` Modal Volume and an L4/T4 GPU image, so a Modal
  account is required for any end-to-end repro from scratch.
- ~5 GB free disk for `data/` + caches.
- Hugging Face token only if the public dataset becomes gated; today it's
  open at `aims-foundations/measurement-db`.

---

## 3. Path A — rebuild current submission, no retraining

This is the path you want if you trust the team's last-trained Stage 2
artifacts and just need a fresh ZIP.

### 3.1 Install

```bash
uv sync
```

### 3.2 Get the public training data

```bash
make data
```

Produces `data/{joined,items,responses,subjects,benchmarks}.parquet`.
`joined.parquet` (~360 MB) is the only one you'll touch directly for most
flows. Downloads from the public HF repo
`aims-foundations/measurement-db`.

### 3.3 Materialize Stage 1 runtime artifacts

```bash
make kfactor-export
```

This reads the committed parquets at
`stage_1/k_factor_irt/artifacts/k4_full_train/` and joins them against
`data/subjects.parquet` (on `display_name`) to write
`data/stage1/kfactor_k4/`:

- `subject_state.pt` — `subject_bias`, `subject_u` (K=4 latent factors),
  fallback values.
- `subject_name_to_id.json` — display-name → row-index map. **This was
  the silent-bug surface fixed in PR #4**: keys are byte-identical to what
  the runtime extracts from `subject_content`'s `Name:` line.
- `item_targets.pt`, `item_to_id.json` — per-item `(v_1..v_K, z)`
  targets for Stage 2 supervision.
- `manifest.json` — counts and provenance.

Sanity-check (CI runs this on every PR against a fixture):

```
K-factor subject lookup hit-rate: 909/909 (100.0%) against data/joined.parquet
PASS - K-factor contract is satisfied.
```

### 3.4 Drop Stage 2 artifacts in place

The Stage 2 head is **not** committed. You need to either:

1. **Have a teammate `modal volume get` and hand you a tarball**, then
   extract into `data/stage2/kfactor_mpnet_linear_v1/`. The expected
   filenames are:

   ```text
   data/stage2/kfactor_mpnet_linear_v1/
     head.pt
     head_meta.json
     target_scaler.json
     side_feature_meta.json
     calibration.json
     metrics.json
   ```

2. **Or pull them from the team's Modal volume** if you have access:

   ```bash
   modal volume get eval-comp-data stage2 data/stage2
   ```

3. **Or retrain** (see Path B below).

### 3.5 Build the submission ZIP

```bash
python scripts/build_submission.py v1_kfactor
```

This produces `submissions/v1_kfactor.zip` (~2.25 MB) with the flat
layout Codabench expects:

```text
model.py
models.txt
head.pt, head_meta.json, target_scaler.json, side_feature_meta.json,
calibration.json
subject_state.pt, subject_name_to_id.json, subject_to_id.json,
item_targets.pt, item_to_id.json, manifest.json
```

`scripts/build_submission.py` flattens names — it pulls Stage 1 files
from `data/stage1/kfactor_k4/` and Stage 2 files from
`data/stage2/kfactor_mpnet_linear_v1/` into the ZIP root. The runtime
`submissions/v1_kfactor/model.py` reads them with no path prefix.

### 3.6 Smoke gate (REQUIRED before any Codabench upload)

```bash
modal run modal_smoke_submission.py --zip submissions/v1_kfactor.zip --n 20
```

This spins up a Linux/CPU container with `HF_HOME=/app/hf_cache`
(encoder pre-fetched at image-build time), `TRANSFORMERS_OFFLINE=1`,
`HF_HUB_OFFLINE=1`, mimicking Codabench's network-isolated runtime. It
loads the zip, runs `predict()` on 20 diverse rows, and exercises both
`labeled=None` and `labeled=[...]` paths.

**Do not upload to Codabench unless this exits 0.** Local Mac smoke
(`make smoke-test SUB=submissions/v1_kfactor`) is necessary but not
sufficient — it doesn't catch the HF-cache / offline-mode failure mode
that bit the first upload (see commit `fe599b0`).

### 3.7 Upload to Codabench

Upload `submissions/v1_kfactor.zip` via the Codabench web UI. **Cap is 1
scored submission per UTC day per team**, so every upload is a slot
spent.

---

## 4. Path B — full reproduction from scratch (Modal required)

You need this if:

- you want to retrain the Stage 2 head (e.g. with a different head /
  encoder / split), or
- you want to verify the team's Stage 2 artifacts byte-for-byte from
  source.

### 4.1 One-time Modal setup

```bash
modal volume create eval-comp-data || true
modal volume put eval-comp-data data/joined.parquet   joined.parquet
modal volume put eval-comp-data data/items.parquet    items.parquet
modal volume put eval-comp-data data/subjects.parquet subjects.parquet
modal volume put eval-comp-data data/stage1           stage1
```

This mirrors your local `data/` layout into the `eval-comp-data` volume
so the existing `scripts/*.py` run unmodified on Modal.

### 4.2 Encode items (T4 GPU, ~few minutes)

```bash
modal run modal_stage2.py --stage encode
```

Produces `data/embeddings/mpnet_v1/{item_embeddings.npy, item_side_features.npy}`
on the volume (70,873 x 768 embedding matrix, 70,873 x 231 side-feature
matrix).

To use a different encoder (e.g. `BAAI/bge-large-en-v1.5`, Lever D in
the v2 plan), edit the `ENCODER` constant at the top of
`modal_stage2.py` and re-run.

### 4.3 Train the K-factor head (CPU)

```bash
modal run modal_stage2.py --stage train
```

Produces `data/stage2/kfactor_mpnet_linear_v1/{head.pt, head_meta.json,
target_scaler.json, side_feature_meta.json, calibration.json,
metrics.json}` on the volume. Default is a linear head, 200 epochs, val
fraction 0.1.

### 4.4 Evaluate offline

```bash
modal run modal_stage2.py --stage eval
```

Reports base / calibrated mean log-likelihood and AUC on the item-cold
val split. Baseline numbers we've seen with the current pipeline (from
`current-state.md` §3):

```text
base mll      = -0.600
calibrated    = -0.533    (offline Platt; overfits — see §4c)
AUC           = 0.775
```

### 4.5 Pull results back locally

```bash
modal volume get eval-comp-data embeddings ./data/embeddings
modal volume get eval-comp-data stage2     ./data/stage2
```

Then continue from §3.5 above (build ZIP, smoke, upload).

---

## 5. Where to look for more context

- **`README.md`** — competition contract: input schema, `predict()`
  signature, network-isolation policy, GPU tiers. Read first if you're
  touching `model.py` or `labeling.py`.
- **`Predictive Evaluation Challenge.pdf`** — organizers' canonical
  spec; sections to know are 2.2/3.4 (adaptive labeling), 3.3 (submission
  contract), 4.6 (network isolation), 2.1/3.1 (scoring).
- **`Makefile`** — every reproduction step is encoded as a `make`
  target; `make help` enumerates them.
- **`scripts/build_submission.py`** — authoritative source of what
  filenames go into the ZIP and from where.
- **`modal_stage2.py`** docstring — the end-to-end Modal pipeline.
- **`modal_smoke_submission.py`** — the upload gate; mimics the
  Codabench runtime as closely as we can.

The `.library/` directory holds a local-only knowledge base (not pushed)
with deeper design notes, plans, and session summaries. If you have
access to it, `main/plans/design.md` and `main/current-state.md` are the
two highest-value reads.

---

## 6. Common gotchas

- **`make kfactor-export` uses bare `python`**, not `uv run python`,
  inconsistent with the rest of the Makefile. If your default `python`
  isn't the project's `.venv` Python, run it as
  `uv run python scripts/export_kfactor_stage1.py ...` (full args in the
  Makefile target).
- **The submission `model.py` reads artifact files by bare filename**
  (`head.pt`, not `data/stage2/.../head.pt`) because the ZIP layout is
  flat. If you copy artifacts manually, make sure `build_submission.py`
  picks them up — it scans the `SUBMISSION_DEFAULT_INCLUDES` paths in
  `scripts/build_submission.py`.
- **Stage 1's `subject_name_to_id.json` is keyed by `display_name`, not
  `subject_id`** (PR #4 fix). The runtime extracts the name from
  `subject_content`'s first `Name:` line. If you regenerate it from a
  different join, hit-rate against `joined.parquet` must be 100%.
- **`V1_KFACTOR_DUMMY_ENCODER=1`** makes the runtime use a deterministic
  hash-based "encoder" so you can smoke-test the submission without
  loading mpnet. Required for `make test-kfactor` (the contract test
  uses fixture artifacts).
- **HF cache resolution** in the submission runtime follows
  `HF_HOME` -> `TRANSFORMERS_CACHE` -> `/app/hf_cache` -> submission-local
  `.hf_cache`. On Codabench the second-to-last wins because the platform
  pre-fetches into `/app/hf_cache`.
