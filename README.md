# eval_comp — Predictive AI Evaluation Challenge

Stanford CS 321M team submission. 3 people, 12-day sprint, deadline **2026-05-22**.

> Quick links: [`plan.md`](plan.md) — strategy & calendar · [`research-report.md`](research-report.md) — why this pipeline · [`STARTER_KIT.md`](STARTER_KIT.md) — original competition spec.

---

## What we're building

A `predict(input, labeled=None) -> float` that given four text fields (benchmark, condition, subject_content, item_content) returns the probability that the AI subject answers the item correctly. **Test items are new**; test subjects are seen in training. Metric: mean log-likelihood.

**Our pipeline** mirrors Truong et al. 2025 (ICML; arXiv:2503.13335) — the AIMS lab's own reference paper for this exact competition:

1. **Stage 1.** Fit Rasch IRT on training response matrix → `θ_s` per subject, `b_i` per item.
2. **Stage 2a.** Sentence-transformer + small head learns `item_text → b̂`. Generalizes to new items.
3. **Stage 2b.** Lookup `θ̂_s` for known subjects (test subjects are warm).
4. **Stage 3.** Combine: `P = σ(θ̂ − b̂)`. Clip, return as Python float.

Plus: adaptive labeling for online Platt calibration, optional LLM-judge ensemble. Details: see `plan.md §3` and `research-report.md`.

---

## Quickstart

```bash
# 1. Install deps (uv-managed)
uv sync

# 2. Download training data once (needs HF auth + disk)
make data

# 3. For parallel dev BEFORE real Stage 1 fits:
make irt-mock          # synthetic Stage 1 outputs in data/irt/

# 4. Run the full v1 pipeline locally
make irt               # Stage 1: Rasch IRT fit (overwrites data/irt/)
make encode            # Stage 2a part 1: encode items
make head              # Stage 2a part 2: train content head
make submission NAME=v1_irt
make smoke-test SUB=submissions/v1_irt
```

`make help` lists all targets.

---

## Project structure

```
eval_comp/
├── README.md              ← you are here
├── plan.md / .html        ← 12-day calendar, design decisions, meeting agenda
├── research-report.md     ← why the IRT pipeline, with sources
├── STARTER_KIT.md         ← original competition spec (data format, submission contract)
├── Makefile               ← workflow shortcuts
├── pyproject.toml         ← uv-managed deps
│
├── src/                   ← shared library code
│   └── validation.py      ← item-cold-start splits, log-likelihood + AUC scoring
│
├── scripts/               ← pipeline scripts (one per stage)
│   ├── download_data.py   ← HF dataset → data/joined.parquet
│   ├── eda.py             ← one-page data summary
│   ├── fit_irt.py         ← Stage 1: Rasch/2PL IRT
│   ├── mock_irt.py        ← Stage 1 mocks for parallel dev
│   ├── encode_items.py    ← Stage 2a part 1: sentence-transformer encoding
│   ├── train_content_head.py  ← Stage 2a part 2: text → IRT params regressor
│   ├── build_submission.py    ← package a submission dir + state into a ZIP
│   └── smoke_test.py      ← local CPU test of a submission's predict()
│
├── submissions/           ← one subdir per submission version (+ built ZIPs)
│   ├── smoke_test/        ← constant 0.5 baseline (verifies pipeline)
│   └── v1_irt/            ← Rasch + linear content head
│
├── tests/                 ← lightweight contract tests
│   └── check_contract.py  ← verifies Stage 1 outputs match the schema Stage 2 expects
│
├── data/                  ← gitignored — generated artifacts
│   ├── joined.parquet     ← four-field training rows (from download_data.py)
│   ├── irt/               ← Stage 1 outputs (theta, b, log_a, lookups)
│   ├── embeddings/        ← Stage 2a part 1 cache
│   └── head/              ← Stage 2a part 2 weights
│
├── sample_code_submission/  ← starter-kit reference, unchanged
└── templates/               ← starter-kit reference, unchanged
```

---

## Pipeline at a glance

```
download_data.py
        │
        ▼
data/joined.parquet ─────────────────┐
        │                            │
        ▼                            ▼
   scripts/eda.py            fit_irt.py  OR  mock_irt.py
                                      │
                                      ▼
                            data/irt/{theta,b,log_a}.pt
                            data/irt/{subject,item}_to_id.json
                                      │
                                      │              ┌─── encode_items.py ──→ data/embeddings/
                                      │              │
                                      └──────────────┴──→ train_content_head.py
                                                                  │
                                                                  ▼
                                                          data/head/{head.pt, head_meta.json}
                                                                  │
                                                                  ▼
                                                      build_submission.py NAME=v1_irt
                                                                  │
                                                                  ▼
                                                      submissions/v1_irt.zip → upload
```

Stage 1 ↔ Stage 2 contract: see `plan.md §0.5.9`. Two-line summary: Stage 2 reads `data/irt/b.pt` (regression targets) and `data/irt/item_to_id.json` (alignment). Everything else is consumed by the submission at inference time.

---

## Ownership

| Area | Files | Owner |
|---|---|---|
| Data + validation | `scripts/download_data.py`, `scripts/eda.py`, `src/validation.py` | _TBD_ |
| Stage 1 (IRT) | `scripts/fit_irt.py`, `scripts/mock_irt.py` | _TBD_ |
| Stage 2a (content head) | `scripts/encode_items.py`, `scripts/train_content_head.py`, `submissions/v1_irt/model.py` | **you (Joey)** |
| LLM-judge / adaptive labeling (Day 5+) | `submissions/v2_*/labeling.py`, `submissions/v2_*/model.py` | _TBD_ |
| Report | `report/` (when we create it) | _TBD_ |

Fill in names after the first team meeting. Decisions tracked in `plan.md §0.5.4`.

---

## Daily workflow

1. `git pull --rebase`
2. Make changes on a branch (or push directly to main if the team agreed — see `plan.md §0.5.4`).
3. Before committing model/script changes:
   - `make test` — sanity-checks the Stage 1↔2 contract.
   - For submissions: `make smoke-test SUB=submissions/<name>` — verifies `predict()` returns proper floats.
4. Before submitting to Codabench:
   - Validate offline on the cold-start split (see `src/validation.py`).
   - Confirm offline score beats current best. **Submission discipline rule:** no Codabench submit without an offline gain.

---

## Submission contract (Codabench)

ZIP must contain at the top level:

- `model.py` (required) — defines `predict(input: dict, labeled: list[dict] | None = None) -> float`
- `labeling.py` (optional) — defines `acquisition_function(input: dict) -> float`
- `models.txt` (optional) — HF repos to pre-fetch (max 5; 300B params total)
- `requirements.txt` (optional) — extra pip deps (organizer-controlled; usually disabled)
- Any auxiliary state files (`.pt`, `.npy`, `.json`)

Full details: `STARTER_KIT.md`. Submission contract spec: `Predictive Evaluation Challenge.pdf §3.3`.

---

## Gotchas (read before debugging)

1. **`predict()` must return a native Python `float`**, not `numpy.float32` or `torch.Tensor`. CSV serialization fails otherwise. Wrap with `float(...)`.
2. **Module-level init for everything heavy.** Encoders, weights, calibration caches load at import time. Loading inside `predict()` re-loads on every round.
3. **No outbound network calls at test time.** Everything in the ZIP or declared in `models.txt`.
4. **`acquisition_function` must never raise, time out, or return NaN/inf** — if it does, the whole round falls back to random sampling.
5. **`subject_content` is text, not a dict.** Parse the `Name:` line; treat metadata lines as optional. Don't rely on raw-string equality for the subject lookup.
6. **Filter binary labels.** Some training rows have continuous-scored labels; the `download_data.py` script filters these by default.

---

## When in doubt

- **What approach are we using?** → `research-report.md` §10 ("Recommendation")
- **What's the schedule?** → `plan.md §4` (12-day calendar)
- **What should I work on today?** → `plan.md §0.5.8` (task breakdown)
- **How does my piece connect to the others?** → `plan.md §0.5.9` (Stage 1↔2 contract)
- **What's the meeting agenda?** → `plan.md §0.5`
- **How do I add a new HF model?** → add repo to `submissions/<name>/models.txt` (max 5 total)
- **My script broke something.** → `make test` and post the error in the team channel.
