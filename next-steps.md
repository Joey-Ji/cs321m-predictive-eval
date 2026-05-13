# Next Steps — Adapting Stage 2 to Yiheng's K-factor IRT (PR #1)

Working plan for moving forward after PR #1 lands. Focus: what Joey owns, and how to wire the scaffold so the three of us can keep working in parallel without stepping on each other.

> Author: Joey (via working session) · Date: 2026-05-12 · Lives on: `dev` branch (or feature branch `working/next-steps`)

---

## 1. TL;DR

PR #1 (`devYaoYH`) ships a real Stage 1: a K-factor MIRT model with formula `logit = S_i + U_i · V_j + Z_j` (K=4, train loss 0.2919, accuracy 86.5%). It uses a different output schema than the 1PL/2PL scaffold I drafted — Parquet files under `stage_1/k_factor_irt/artifacts/`, not `.pt` tensors under `data/irt/`. **My Stage 2a scaffolding is now out of date and needs adaptation.**

The work splits into three buckets:

- **Schema reconciliation** (today) — agree with Yiheng on the canonical Stage 1 output format and document it.
- **Stage 2a adaptation** (1–2 days) — refit the content head to predict (V_j, Z_j) instead of just b_i; refactor `submissions/v1_irt/model.py` to do the bilinear combine.
- **Scaffold-for-parallelism** (~half a day in parallel with the above) — introduce a thin adapter layer so future Stage 1 changes (K=8, model swap, etc.) don't break Stage 2.

---

## 2. What changed in PR #1

### 2.1 The model

```
logit_ij = S_i + U_i · V_j + Z_j
```

| Symbol | Shape | Meaning |
|---|---|---|
| `S_i` | scalar | Per-subject overall bias (this subject's general capability) |
| `U_i` | K-dim vector | Per-subject latent capability vector |
| `V_j` | K-dim vector | Per-item loading vector (which factors this item taps) |
| `Z_j` | scalar | Per-item difficulty bias |

K = 4 in the current run. The dot product `U_i · V_j` is the interaction term — high when a subject's strengths align with what an item demands. This is essentially MIRT (multidimensional IRT) with separate subject and item bias terms.

Compared to the 1PL Rasch scaffold I built:
- 1PL: 1 scalar per subject (`θ_s`), 1 scalar per item (`b_i`)
- K-factor: **1 scalar + K-vector per subject, K-vector + 1 scalar per item**

### 2.2 The new artifact layout

```
stage_1/k_factor_irt/
├── README.md
├── fit_k_factor_irt.py
├── fit_mlp_irt.py
└── artifacts/k4_full_train/
    ├── README.md
    ├── manifest.json
    ├── subject_capabilities.parquet     # rows: subject_id, S_i (scalar), U_i (K-dim list)
    └── item_parameters.parquet          # rows: item_id, V_j (K-dim list), Z_j (scalar)
```

This is incompatible with my earlier scaffold which expected `.pt` tensors under `data/irt/`. **Adopting the new layout (parquet + manifest) is the right move** — it's more portable, ships better with Yiheng's branch, and the manifest gives us a place to store metadata (K, model type, version).

### 2.3 What this breaks in our current code

| File | What breaks | What needs to change |
|---|---|---|
| `scripts/train_content_head.py` | Reads `data/irt/b.pt`; head out_dim=1 or 2 | Read parquet via adapter; head out_dim = K+1 (predict V_j and Z_j) |
| `submissions/v1_irt/model.py` | Combines `σ(θ − b̂)`; expects scalar everything | Combine `σ(S + U·V̂ + Ẑ)`; need to load S and U too |
| `scripts/mock_irt.py` | Produces `.pt` tensors with 1PL/2PL shapes | Produce parquet files matching Yiheng's schema |
| `tests/check_contract.py` | Checks `.pt` schema | Check parquet schema + manifest |
| `scripts/build_submission.py` | Pulls from `data/irt/` and `data/head/` | Update default `--include` paths |

---

## 3. What Joey owns going forward

In dependency order. Items 1–2 are blocking; 3–5 can be parallelized with the other teammate's work.

### 3.1 Schema reconciliation (today)

- Review PR #1 end-to-end. Pay attention to:
  - Exact column types in the two parquet files (are vectors stored as Python lists, Arrow lists, or fixed-size lists? Affects how we read them).
  - What's in `manifest.json` (does it include K, encoder version, training hyperparams?).
  - Whether subject IDs are stored raw or normalized.
- Sync with Yiheng on:
  - Is the parquet schema stable going forward? Or will K change between runs?
  - Should the manifest include K explicitly so Stage 2 can configure itself?
  - Where do future variants (e.g., K=8) write their artifacts? Convention: `stage_1/k_factor_irt/artifacts/k<K>_<run_name>/`.
- Document the canonical schema explicitly in `stage_1/k_factor_irt/artifacts/<run>/README.md` (Yiheng already has a stub).

### 3.2 Build the Stage 1 adapter (today)

Single source of truth for reading Stage 1 outputs across all of Stage 2's code. New file: `src/stage1_adapter.py`. Suggested API:

```python
@dataclass
class Stage1Manifest:
    k: int                         # number of latent factors
    model_type: str                # "k_factor", "1pl", "2pl", ...
    artifact_dir: Path
    encoder_version: str | None    # if relevant for downstream

@dataclass
class SubjectParams:
    subject_to_id: dict[str, int]  # normalized name -> id
    S: torch.Tensor                # [n_subjects]
    U: torch.Tensor                # [n_subjects, K]

@dataclass
class ItemParams:
    item_to_id: dict[str, int]
    V: torch.Tensor                # [n_items, K]
    Z: torch.Tensor                # [n_items]

def load_manifest(artifact_dir: Path) -> Stage1Manifest: ...
def load_subjects(artifact_dir: Path) -> SubjectParams: ...
def load_items(artifact_dir: Path) -> ItemParams: ...
```

Why this matters: every Stage 2 file that previously imported `data/irt/*.pt` now imports through this adapter. When Stage 1 changes (K=4 → K=8, or a different model formula), only the adapter needs updating — not the head trainer, not the inference path, not the tests.

### 3.3 Adapt `train_content_head.py`

- Read item targets via `load_items()` — get `V` (K-dim) and `Z` (scalar) for every training item.
- Head output dim = `K + 1`. Concat as `(V_hat_dim_0, ..., V_hat_dim_{K-1}, Z_hat)`.
- Loss: MSE on the full (K+1)-dim target. Consider weighting Z slightly higher than V dims if validation says so (Z is more directly interpretable).
- Validate on item-cold-start split (existing logic, just per-dim metrics).
- Save `head.pt` + updated `head_meta.json` with `out_dim`, `target_layout: ["V_0", ..., "V_{K-1}", "Z"]`, K, encoder name.

### 3.4 Adapt `submissions/v1_irt/model.py`

- At module init: load encoder, head, manifest, subject params.
- In `predict()`:
  1. Normalize subject_content → look up `(S_i, U_i)`. Fallback to population mean if unknown.
  2. Encode item_content → run head → split output into `V_hat` (first K dims) and `Z_hat` (last dim).
  3. Combine: `logit = S_i + np.dot(U_i, V_hat) + Z_hat`.
  4. Return `float(clip(sigmoid(logit), 0.02, 0.98))`.
- `models.txt` keeps the current sentence-transformer.

### 3.5 Update mocks + contract test

- `scripts/mock_irt.py`: produce parquet files in the new schema. Add `--k` flag. Generate random (S, U, V, Z) with realistic scales.
- `tests/check_contract.py`: replace `.pt`-tensor checks with parquet-schema checks + manifest validation.
- `make irt-mock` should produce a working v1_irt submission against synthetic Stage 1 outputs end-to-end.

### 3.6 End-to-end run + submit

- Pull Yiheng's branch (or wait for merge), point at `stage_1/k_factor_irt/artifacts/k4_full_train/`.
- `make encode` → `make head` → `make submission NAME=v1_irt` → `make smoke-test SUB=submissions/v1_irt`.
- Verify offline log-likelihood beats Day-0 floor (`ln 0.5 ≈ −0.693`).
- Submit to Codabench.

Expected: this gets us our first real (non-trivial) leaderboard number. The K-factor model is more expressive than 1PL/Rasch; if the content head can hit even modest R² on V_j and Z_j, we should see a meaningful jump above 0.5.

---

## 4. How to make the scaffold parallel-friendly

Right now we have three latent workstreams: Stage 1 (Yiheng), Stage 2a (Joey), and the LLM-judge / adaptive-labeling track (third teammate, TBD). To keep them unblocked:

### 4.1 The adapter pattern (the key decoupling)

`src/stage1_adapter.py` decouples Stage 2 from Stage 1's choice of model. As long as the adapter API is stable, Yiheng can:
- Iterate on K (4, 6, 8) without breaking Stage 2.
- Try MLP-based item embedders (`fit_mlp_irt.py`) without breaking Stage 2.
- Try regularization variants without breaking Stage 2.

Joey can:
- Iterate on encoder choice (mpnet, bge-large) without touching Stage 1 code.
- Iterate on head architecture (linear, MLP, residual) without touching Stage 1.

Third teammate can:
- Build LLM-judge submissions that don't even consume Stage 1 outputs directly; ensemble at the probability level.

### 4.2 Submissions directory pattern

One subdir per submission variant, completely independent:

```
submissions/
├── smoke_test/             # constant 0.5 (floor)
├── v1_irt_k4/              # K=4 MIRT + linear head        (Joey, first real submission)
├── v2_irt_k4_mlp/          # K=4 MIRT + MLP head           (Joey, iteration)
├── v3_irt_k8/              # K=8 MIRT + MLP head           (Joey, after Yiheng's K=8)
├── v4_llm_judge_zs/        # zero-shot Qwen2.5-7B judge    (teammate C)
├── v4_llm_judge_fs/        # few-shot judge w/ labeled     (teammate C)
└── final_ensemble/         # log-mixture of best 2-3       (jointly)
```

Each subdir is self-contained: model.py + state files. `build_submission.py` packages whichever one you specify. Nobody blocks anyone.

### 4.3 Manifest-driven configuration

Store K, model type, training run, encoder, etc. in `manifest.json`. Stage 2 reads this and configures itself. Avoids hardcoded constants in Stage 2 code.

### 4.4 Tests as contract gates

`tests/check_contract.py` should fail loudly if the parquet schema drifts. CI runs on every PR; nobody can merge a schema break without seeing it.

### 4.5 Submission discipline rule (existing, keep)

No Codabench submit without an offline-validation improvement on the current best. Prevents teammates from racing each other on Codabench and burning the 1000-total cap.

---

## 5. Risk map

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Yiheng iterates Stage 1 schema again | medium | medium | Adapter layer absorbs it; only `src/stage1_adapter.py` changes |
| Content head can't predict K-dim V_j accurately | medium | high | Train and inspect per-dim R² before committing to the full pipeline; consider Z-only ablation as fallback |
| Subject-name normalization disagrees between Stage 1 and Stage 2 | medium | high | Pin a single `normalize_subject(subject_content) -> str` in the adapter; both stages use it |
| K-factor model overfits with K=4 on test items | medium | medium | Compare validation log-likelihood with K=1 (Rasch); if K=4 doesn't help, fall back |
| Two teammates push to same submissions/ subdir | low | medium | One owner per submission directory; document in README |
| Schema-break PR slips through review | low | high | `make test` is CI-gated; cannot merge red |

---

## 6. Where Joey starts (today)

In order, smallest commits first:

1. **Pull Yiheng's branch locally; inspect the parquet schemas + manifest.json.** Document any surprises.
2. **Write `src/stage1_adapter.py`** with the API in §3.2. Test it interactively against Yiheng's artifacts.
3. **Update `scripts/mock_irt.py`** to write parquet in the new schema. Run `make irt-mock` and verify it produces valid output.
4. **Update `tests/check_contract.py`** for the parquet schema. Run `make test`; should pass against the mock.
5. **Update `scripts/train_content_head.py`** to consume the adapter; head out_dim = K+1.
6. **Update `submissions/v1_irt/model.py`** for the bilinear combine.
7. **Run end-to-end against Yiheng's real K=4 artifacts**, smoke-test, submit.

Each step is a separate PR. CI on PRs catches schema breaks early.

---

## 7. Coordination questions for next sync

- Should `src/stage1_adapter.py` and the parquet schema live under `stage_1/` (Yiheng's directory) or `src/` (shared library)? Recommend `src/` so both teams can import it without depending on each other's paths.
- Will Yiheng keep `fit_k_factor_irt.py` and `fit_mlp_irt.py` as separate scripts, or consolidate? Doesn't block Stage 2 either way.
- Who's owning the LLM-judge track? Day 5 go/no-go criteria from earlier plan still apply: skip if Stage 1+2a beats Day-2 floor by a comfortable margin.
- Should we move `scripts/fit_irt.py` (my old 1PL/2PL scaffold) into a `legacy/` subdir, or just delete? Recommendation: keep it for now as a sanity-check baseline (smaller, faster to debug); revisit on Day 4.
