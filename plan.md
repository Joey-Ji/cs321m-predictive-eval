# Predictive AI Evaluation Challenge — Plan

This is the canonical, agent-readable plan. For a fast scannable version, open `plan.html` in a browser.

---

## 0. Status (updated 2026-05-10)

**Logistics resolved:**

- **Team:** 3 people. Combined Modal budget = $1,500 (3 × $500 personal coupons; cannot be pooled formally but can be applied to shared work).
- **Codabench:** team registered.
- **Deadline:** Friday 2026-05-22. **12 days from today.** First team meeting still to be scheduled.
- **Submission cap:** 50 scored submissions per team per UTC day; **1,000 total over the competition window.** This resolves the PDF/README discrepancy — we have plenty of headroom for fast iteration but should still treat each submission as expensive (1000-budget shared across 3 people × 12 days = ~28/day average if we want to leave a buffer; we won't actually use anywhere near that).
- **Compute:** primarily Modal. Local GPUs not part of the plan.

**Still TBD (will resolve at first team meeting):**

- Report ownership / sections.
- Modeling preference: full LLM-judge ensemble vs. cheaper IRT + content head.

**Pacing implication:** the original "phase by week" framing is now a **12-day sprint**. See §4 for the dated calendar.

---

## 0.5. Design decisions for first team meeting

Based on the research synthesis (`research-report.md` / `research-report.html`), here is what the meeting should produce. **Headline finding:** the AIMS lab itself published a reference paper for this competition — **Truong et al., ICML 2025 (arXiv:2503.13335)** — implementing Rasch IRT + linear probe on Llama-3-8B embeddings, with reported column-holdout AUC ≈ 0.804 in the AIMS textbook Chapter 3. Mirror that pipeline.

### 0.5.1 Pre-meeting prep (priority order, you/teammates before meeting)

1. **Upload `submissions/smoke_test.zip` to Codabench.** Confirms pipeline + records floor (~−0.693 mean log-likelihood).
2. **Locate `torch_measure` package** on the course infrastructure (Codabench starter, course portal, Canvas). The spec says organizers provide it. If found, use it for Stage 1 instead of reimplementing — saves a day, signal-positive in report.
3. **Check the public Codabench leaderboard.** Has the organizer baseline dropped yet? Top score?
4. **Try to kick off the data download.** `python scripts/download_data.py` needs `pip install datasets huggingface_hub pyarrow` and HF login.
5. **Send teammates the 1-line read-ahead:** "Skim `plan.html` and `research-report.html` (5 min total). Pipeline recommendation is in §1 of the research report."
6. **Confirm everyone's logistics:** Modal coupon redeemed, on Codabench team roster, no exam/travel conflicts in the 12-day window.

### 0.5.2 Pipeline recommendation (confirm at meeting)

| Phase | Approach | Days | Submissions |
|---|---|---|---|
| Primary | Rasch IRT + bge-large/mpnet content head + temperature scaling + per-round Platt | 1–5 | NCF baseline (D2) → IRT-content (D4) → +adaptive (D5) |
| Ensemble | Local LLM-judge (Qwen2.5-7B) + Venn-Abers calibration | 5–8 | Standalone judge (D7) → ensemble v1 (D7–8) → few-shot (D8) |
| Final | Log-mixture ensemble; weights tuned on validation | 9–11 | Best ensemble locked D9; report draft by D10 |

Rationale: this is the AIMS lab's own published pipeline. Naturally calibrated; cleanest report story (decomposition into ability × difficulty); fastest at inference; realistic to beat 0.804 column-holdout AUC with calibration discipline + ensembling.

### 0.5.3 What we explicitly drop

LightFM/FM/FFM, 3PL IRT, GNNs (transductive or inductive), transformer-from-scratch, FT-Transformer. See `research-report.md` §4 for one-line justifications.

### 0.5.4 Decisions the meeting MUST produce

1. **Roles** (4 areas, 3 people): (a) data + validation harness, (b) IRT + content head, (c) LLM-judge + adaptive labeling, (d) report. Someone owns two; report owner can also own a technical area.
2. **Pipeline confirm:** mirror Truong et al. as primary? (Default: yes — the research is unambiguous.)
3. **LLM-judge in/out:** decide now, or defer to Day 5? (Default: defer to Day 5 once IRT validation numbers are in. Use the Day 6 go/no-go criteria below.)
4. **Working agreements:** repo workflow (branches/PR), comm channel/cadence, submission discipline rule (no Codabench submission without an offline-validation improvement on best so far).
5. **Modal coordination:** whose credit pays for which jobs (training jobs on owner's credit; ad-hoc inference burns are anyone's).
6. **Next sync:** when. Suggest: short async daily, full sync at Day 5 around the LLM-judge decision.

### 0.5.5 Stop-loss / decision criteria (binding)

- **Day 4:** IRT + content-head must beat Day 2 NCF on validation log-likelihood. If not, debug the IRT fit before adding components.
- **Day 6 LLM-judge go/no-go:** GO only if (a) IRT pipeline is within 0.05 mean log-likelihood of the leaderboard top, AND (b) we have an engineer with bandwidth. Otherwise deepen IRT (hierarchical priors, benchmark-conditional difficulty, MIRT exploration).
- **Day 10 hard gate:** report draft must exist by EOD. Modeling iteration freezes; remaining time is calibration + ensemble tuning + writing.

### 0.5.6 Calibration discipline (everyone agrees, applies to every submission)

- Label smoothing (ε=0.05–0.10) in BCE training loss.
- Temperature scaling on held-out training fold (single scalar T, optimize log-loss).
- Per-round in `predict()`: global Platt on all 25 revealed labels (default); per-category temperature only if bucket has ≥3 of each class, shrunk toward the global Platt.
- Clip outputs to [0.02, 0.98].

### 0.5.7 What NOT to discuss at the meeting (defer)

- Specific IRT priors / regularization choices.
- LLM-judge prompt format details.
- Report section structure beyond "who's writing it."
- Choice of bge-large vs. mpnet vs. Llama-3-8B encoder — empirical, decide on Day 2.
- Per-category vs. global Platt threshold — Day 5 conversation once validation numbers exist.

### 0.5.8 Concrete task breakdown — replication backbone vs. score-higher levers

**Decision (committed):** mirror the Truong et al. pipeline. Below is the actual code work.

**Backbone (~300 LoC; mandatory; lands ~0.804 if executed cleanly):**

| # | Task | File | Owner | Day |
|---|---|---|---|---|
| 1 | Download + EDA | `scripts/download_data.py`, `scripts/eda.py` | Eng A | 1 (done — needs run) |
| 2 | Validation harness | `src/validation.py` | Eng A | 1 (done) |
| 3 | Fit Rasch IRT on training matrix → `data/theta.pt`, `data/item_params.pt`, `data/subject_to_id.json` | `scripts/fit_irt.py` | Eng B | 2 |
| 4 | Extract item embeddings (start with mpnet) → `data/item_embeddings.npy` | `scripts/encode_items.py` | Eng B | 2 |
| 5 | Train content head: embedding → predicted `b` (linear first; MLP on iter) | `scripts/train_content_head.py` | Eng B | 3 |
| 6 | Wire all stages into `predict()` | `submissions/v1_irt/model.py` | Eng B | 3 |
| 7 | Build + submit ZIP | `scripts/build_submission.py` | Eng A | 3 |

**Score-higher levers (the actual work that beats 0.804):**

| Lever | Expected lift (Δ mean log-likelihood) | Cost | Owner | When |
|---|---|---|---|---|
| **L1. Adaptive labeling + online Platt** in `predict()` (the competition's named lever; Truong et al. doesn't use it) | +0.01-0.03 | 1 day | Eng C | Day 5 |
| **L2. Calibration discipline** (temperature scaling on held-out + label smoothing in BCE + clip [0.02, 0.98]) | +0.02-0.05 | 0.5 day | Eng B | Days 4-5 |
| **L3. Per-benchmark specialization** (per-benchmark difficulty regressor OR per-benchmark temperature) | +0.01-0.03 | 1 day | Eng B | Day 6 |
| **L4. Side features** (concatenate benchmark + condition embedding to the head input) | +0.005-0.02 | 0.5 day | Eng B | Day 4 |
| **L5. Ensemble** (3-5 diverse predictors via log-mixture; weights tuned on validation) | +0.03-0.08 | 1 day | Eng A | Days 9-10 |
| **L6. LLM-judge as ensemble component** | +0.01-0.05 *if it works* | 3 days | Eng C | Days 6-8 (gated; see §0.5.5) |
| L7. EM iteration between Stage 1 IRT and Stage 2a head (matches reference exactly) | +0.005-0.02 | 0.5 day | Eng B | Day 6 |
| L8. Encoder upgrade (mpnet → bge-large-en-v1.5 or Llama-3-8B) | +0.01-0.03 | 0.5 day | Eng B | Day 6 |
| L9. Trace-feature exploration (item-trace length/coherence from `*_traces.parquet`) | uncertain | 1 day | stretch | Day 8+ |

L1, L2, L5 are the high-floor wins everyone should land. L3, L4, L7, L8 are tuning. L6 is the high-variance bet. L9 is genuine exploration.

**Replication choices we're making (not actually choices):**
- Start encoder = `all-mpnet-base-v2` (768-d, fast, in starter `models.txt`). Upgrade encoder is Day 6 swap.
- Start head = linear probe (matches reference). MLP only if validation supports it.
- Start training = one-shot (fit IRT once, then fit predictor on its outputs). EM iteration is Day 6 upgrade.

### 0.5.9 Stage 1 ↔ Stage 2 contract (pin this now)

Stage 1 and Stage 2 can be developed in parallel if both teams agree on the file schema *now* and never break it. Stage 2 reads only two things from Stage 1; everything else is internal to a stage.

**Stage 1 output contract** (produced by `scripts/fit_irt.py` and `scripts/mock_irt.py`):

| File | Shape / type | Meaning | 1PL consumer | 2PL consumer |
|---|---|---|---|---|
| `data/irt/theta.pt` | `float32 [n_subjects]` | Per-subject ability θ | submission only | submission only |
| `data/irt/b.pt` | `float32 [n_items]` | Per-item difficulty b | **Stage 2 (target)** + submission | **Stage 2 (target dim 1)** + submission |
| `data/irt/log_a.pt` | `float32 [n_items]` | Per-item log-discrimination (zeros for 1PL) | submission only (used as constant) | **Stage 2 (target dim 2)** + submission |
| `data/irt/subject_to_id.json` | `dict[str, int]` (normalized name → id) | Subject lookup | submission only | submission only |
| `data/irt/item_to_id.json` | `dict[str, int]` (item_id → id) | **Aligns embeddings ↔ IRT params** | **Stage 2** | **Stage 2** |
| `data/irt/fit_log.json` | metadata | Diagnostics | safe to ignore | safe to ignore |

**1PL vs 2PL — what changes for Stage 2:**

| Aspect | 1PL (Rasch) — v1 default | 2PL — v2 candidate (Day 4 ablation) |
|---|---|---|
| `train_content_head.py` flag | `--targets b` | `--targets b+log_a` |
| Head output dim | 1 | 2 |
| Targets | `b.pt` only | `b.pt` and `log_a.pt` jointly |
| `head_meta.json["target_order"]` | `["b"]` | `["b", "log_a"]` |
| Inference combine | `σ(θ − b̂)` | `σ(exp(log_â) · (θ − b̂))` |
| Submission directory | `submissions/v1_irt/` (already built) | `submissions/v2_irt_2pl/` (build when needed) |

The submission `model.py` reads `head_meta.json["target_order"]` at module init and selects the right inference path. v1's `model.py` is hardcoded for the 1PL path; v2 (when we build it) parses `target_order == ["b", "log_a"]` and uses per-item predicted log_a.

**Stage 2 output contract** (produced by `scripts/train_content_head.py`):

| File | Shape / type | Meaning | Consumed by |
|---|---|---|---|
| `data/head/head.pt` | torch state_dict | Trained content-head weights | submission |
| `data/head/head_meta.json` | dict with keys `head_type`, `hidden`, `in_dim`, `encoder` | Architecture spec needed to instantiate head at inference | submission |
| `data/embeddings/item_embeddings.npy` | `float32 [n_items, d]` | Per-item embeddings (cached) | Stage 2 internal; not strictly needed at submission time |

**Integration point:** the submission `model.py` (e.g., `submissions/v1_irt/model.py`) is the only place Stage 1 + Stage 2 outputs combine. Stages don't talk to each other directly.

**Parallel development pattern:**

1. Both teams agree on the contract above (literally read this section in the meeting).
2. Stage 2 dev runs `python scripts/mock_irt.py` once to populate `data/irt/` with synthetic-but-correctly-shaped outputs. Builds + tests entire content-head + submission pipeline against the mocks.
3. Stage 1 dev fits real IRT, overwrites `data/irt/`. Schema is unchanged so Stage 2 just re-runs `train_content_head.py` against real targets.
4. Day 3 integration: build `submissions/v1_irt/` against real outputs from both stages. Submit.
5. Iterating on Stage 1 hyperparameters (1PL → 2PL, regularization sweeps) only requires re-running `train_content_head.py` afterwards; Stage 2 code never changes.

**Schema-change protocol (in case the contract has to evolve):**

- Adding new files to `data/irt/` is safe (Stage 2 ignores them).
- Renaming or changing shape of `b.pt` or `item_to_id.json` is breaking — needs both teams' explicit agreement and a sync update to `train_content_head.py` and `model.py`.
- If Stage 1 wants to predict 2PL with two parameters (b, log_a) per item that Stage 2 should jointly regress, that's a contract change: Stage 2 head output dim becomes 2, both scripts need updating. Plan ahead.

### 0.5.10 3-line meeting follow-up template

After the meeting, paste this in the team channel:

```
Roles: Data+val=X · IRT+content=Y · LLM-judge/adaptive=Z · Report=?
LLM-judge decision: [in / out / Day 5 go-no-go]
Next sync: [date/time]
```

---

## 1. The competition in one paragraph

For each test row we get four text fields — `benchmark`, `condition`, `subject_content` (the AI being tested), `item_content` (the question) — and must return `P(subject answers item correctly)` as a `float` in `[0, 1]`. Test items are new (item-side cold-start); test subjects are seen in training. Metric is mean log-likelihood of true labels under our predictions ("negative log-loss, higher is better"), with AUC-ROC as secondary. Grading is 50% leaderboard + 50% 4-page NeurIPS-style report. Beating the organizer baseline (released ~3 weeks in) guarantees at least 80% of the leaderboard half.

### 1.1 Subject vs item, made concrete

- **Subject** = the AI model being tested. Examples: `GPT-4`, `Claude-3-Opus`, `Llama-3-70B`. Same models appear in train and test.
- **Item** = a single benchmark question. Test items are new — many from benchmarks that did not exist when training data was collected.
- **Cold-start regime:** subjects known, items new. Classical matrix completion has nothing to fit on these item columns; the only handle is item text + subject metadata + benchmark + condition.

### 1.2 The PGE pipeline the spec hands us

The spec frames this as a three-stage Prediction-Guided Evaluation pipeline:

1. **Stage 1.** Fit a measurement model (e.g., IRT) on training responses. Recover latent subject ability `θ_s` and item parameters `(b_i, a_i)`.
2. **Stage 2a.** Learn `item_text → (â, b̂)`. The only stage that *must* generalize at test time, because every test item is new.
3. **Stage 2b.** Learn `subject_metadata → θ`. For known subjects (our regime), a lookup of the Stage 1 estimates is sufficient.
4. **Stage 3.** Combine: `P(correct) = σ(â · (θ̂ − b̂))`. No parameters of its own.

---

## 2. State of the world (as of 2026-05-10)

### 2.1 Repository contents

```
eval_comp/
  README.md                            # data-loading recipe + submission contract
  Predictive Evaluation Challenge.pdf  # 11-page spec (source of truth)
  sample_code_submission/
    model.py        # returns 0.5 (placeholder)
    labeling.py     # returns 0.0 (placeholder)
  templates/
    hf_submission/
      model.py      # HF model-loading boilerplate, predict() returns 0.5
      models.txt    # currently lists sentence-transformers/all-mpnet-base-v2
    labeling_addon/
      labeling.py   # acquisition_function returns 0.0 (placeholder)
```

### 2.2 What is and isn't done

- **Done:** repo initialized, starter kit pulled, spec read end-to-end.
- **Not done:** team registration, Codabench submission, training-data download, EDA, validation harness, any actual model.
- **Unknown:** team composition, exact deadlines, daily submission cap, Modal credit redemption status.

### 2.3 Time situation

Spec dated 2026-04-20; today is 2026-05-10; final deadline is **2026-05-22 (Friday)**. **12 days remain.** The organizer baseline dropped (or drops) ~3 weeks after start, so we should be close to or just past baseline release — confirming this is on the first-meeting agenda.

---

## 3. Recommended approach

Three model families, layered, plus an active-learning channel.

### 3.1 Stage-1 IRT (per-subject ability)

Fit a 1PL or 2PL IRT model on the training response matrix. Initialize `θ_s` for every subject and `(b_i, log a_i)` for every item as `nn.Parameter`s, optimize binary cross-entropy on observed responses. The `torch_measure` package wraps this; rolling our own is fine.

After Stage 1, every training item has its text *and* fitted `(â, b̂)` — i.e., a supervised regression dataset for Stage 2a.

### 3.2 Stage-2a content head (item text → IRT params)

- **Encoder:** frozen `sentence-transformers/all-mpnet-base-v2` (already in `templates/hf_submission/models.txt`).
- **Head:** small MLP, e.g. `Linear(768→256), ReLU, Linear(256→256), ReLU, Linear(256→2)` predicting `(b̂, log â)`.
- **Loss:** MSE against Stage 1 outputs.
- **Optional features:** concatenate one-hot or embedded `benchmark` and `condition` to the item embedding before the head.

### 3.3 Stage-2b subject lookup

For known subjects, lookup `θ̂_s` from Stage 1. Build a normalized key from the `Name:` line of `subject_content` (the spec is explicit that the raw string is not stable). Fall back to population mean (`θ = 0`) for unrecognized names.

### 3.4 Stage 3 combine

`P(correct) = σ(â · (θ̂ − b̂))`. Cast to native Python `float` before returning.

### 3.5 Optional: LLM-as-judge ensemble

A 7B-class instruction-tuned model (Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct) declared in `models.txt`. Format the four fields into a `Benchmark / Condition / Subject / Item / Answer:` prompt; read next-token log-probs of `yes` and `no`; return `p(yes) / (p(yes) + p(no))`. Ensemble with the IRT-content predictor by log-mixture (weights tuned on validation).

### 3.6 Adaptive labeling

Ship `labeling.py` with a diversity acquisition function. Two viable variants:

- **k-means diversity (§3.6 of the spec):** offline-fit centroids on training-set embeddings; per-round score = "how under-represented is this candidate's cluster so far?" Requires module-level state (a `Counter`).
- **Online farthest-point sampling:** no offline artifact; module-level list of seen embeddings; score = distance to nearest seen.

Use the revealed labels in `predict()` for **online Platt scaling**: fit a 1- or 2-parameter logistic regression on `(raw_score, label)` pairs once at the first `predict()` call of each round, cache it on a module variable, pass every subsequent prediction through it.

### 3.7 Why this layering

- IRT-decomposition factors the problem into ability (lookup) × difficulty (predicted from text) → probability. Cleaner than NCF for the report.
- LLM-judge complements it with a direct measurement; the ensemble usually beats either alone.
- Adaptive labeling is the spec's named lever for the leaderboard; using it for calibration is the highest-EV use of the 25-label budget.

---

## 4. Concrete next steps — 12-day calendar

Dates are aspirational targets, not contracts. Each row should land *something* submittable by end of day.

| Date | Day | Goal | Submission |
| --- | --- | --- | --- |
| 2026-05-10 (Sat) | 0 | Smoke-test ZIP prepared; project scaffolded; data downloader written; first team meeting; agree on ownership. | Smoke-test submission (constant 0.5). |
| 2026-05-11 (Sun) | 1 | Data downloaded; EDA done; validation harness with item-cold-start split; per-benchmark majority-class baseline. | Per-benchmark majority-class. |
| 2026-05-12 (Mon) | 2 | NCF baseline trained on Modal; weights baked into ZIP. | NCF (§3.4). |
| 2026-05-13 (Tue) | 3 | IRT Stage 1 fit; subject ability lookup table built; Stage 2a content head training started. | (None — IRT submission lands tomorrow.) |
| 2026-05-14 (Wed) | 4 | IRT + content head end-to-end; Stage 2b lookup wired in; Stage 3 sigmoid combine. | IRT-content predictor. |
| 2026-05-15 (Thu) | 5 | Adaptive labeling: diversity acquisition function + online Platt calibration in `predict()`. Local sweep over training inputs to confirm no NaN/raise. | IRT + adaptive labeling. |
| 2026-05-16 (Fri) | 6 | Decision point on LLM-judge: if go, declare 7B model in `models.txt`, write prompt template, run on Modal. If no-go, deepen IRT (hierarchical priors, benchmark-conditional difficulty). | LLM-judge or IRT-v2. |
| 2026-05-17 (Sat) | 7 | LLM-judge integration + log-mixture ensemble weights tuned on validation. | Ensemble v1. |
| 2026-05-18 (Sun) | 8 | Few-shot variant: render adaptive-labeled examples through judge prompt. Ablation: zero-shot vs. few-shot. | Ensemble v2 (few-shot). |
| 2026-05-19 (Mon) | 9 | Failure-mode analysis on validation (per-benchmark, per-subject error). Pick ensemble weights and freeze. | Best ensemble locked. |
| 2026-05-20 (Tue) | 10 | Report draft: method, data, experiments + ablations, failure modes. | Final tuning submission if any. |
| 2026-05-21 (Wed) | 11 | Report polish; reproduce final-code Gradescope upload from the best Codabench submission. | Reserve. |
| 2026-05-22 (Thu) | 12 | **Final submission day.** Codabench + Gradescope locked in. | **Final.** |

**Checkpoints / kill criteria:**

- If Day 2 NCF doesn't beat 0.5 by a meaningful margin (mean log-likelihood materially better than −0.693), debug the data pipeline before continuing — something is wrong upstream.
- If Day 4 IRT doesn't beat NCF on validation, freeze the IRT path and ensemble the two; don't let either monoculture drive.
- If Day 6 the LLM-judge looks expensive or doesn't move validation, drop it and use the time for stronger ablations + a better report.
- Report draft must exist by Day 10. The report is 50% of the grade — under-investing in it is the biggest risk to the final score.

---

## 5. Engineering conventions

Non-negotiable for any code shipped to Codabench:

- **Module-level init for everything heavy.** Encoders, model weights, IRT parameters, calibration caches load at import time. Loading inside `predict()` re-loads on every round.
- **`predict()` returns Python `float`,** not `numpy.float32` or `torch.Tensor`. The spec says CSV serialization fails otherwise. Wrap with `float(...)`.
- **No outbound network calls.** Anything wanted at test time goes in the ZIP or in `models.txt` (cap: 5 repos, 300B params total, 1000 GB).
- **Defensive `subject_content` parsing.** Treat it as text, not a stable dict serialization. Optional metadata lines may be missing. Lookup by normalized name extracted from the `Name:` line.
- **`acquisition_function` must never raise, time out, or return NaN/inf** for any candidate. If it does, the whole round falls back to random sampling. Sweep it locally over all training inputs before submitting.
- **Use the public `benchmark_id`** (not the human-readable `name` column).
- **Filter labels appropriately.** Some response tables are continuous/scored; for a binary correctness model, threshold or filter.
- **`acquisition_function` state is module-level only.** Containers are destroyed between rounds; nothing persists. Use globals for in-round state, design artifacts so per-round bootstrap is cheap.

---

## 6. Risks

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| 12-day window slips (illness, exams, unforeseen) | Medium | High | NCF baseline by Day 2 = always have *something* on the leaderboard. Don't gate on the LLM-judge. |
| Report under-invested (50% of grade) | High | High | Draft must exist by Day 10. Treat ablations as report material, not just signal. |
| IRT fit unstable on sparse subjects | Medium | Medium | Hierarchical priors per benchmark; fall back to per-benchmark logistic regression. |
| Embeddings under-represent reasoning difficulty | High | Medium | Add `benchmark`/`condition` as side features; ensemble with LLM-judge. |
| LLM-judge too slow per round on Modal | Medium | Medium | Smaller model (3B) or quantize (bitsandbytes/AWQ); module-level batch queue; kill if Day 7 not converging. |
| Offline validation drifts from hidden test | Medium | High | Item-cold-start AND benchmark-cold-start splits; treat leaderboard as final arbiter. |
| Acquisition function NaN/raise → silent random fallback | Low | Medium | Local sweep over all public training inputs; clamp + sanity-check return values. |
| 1,000 total submission cap eaten early by churn | Low | Medium | Gate every submission on offline validation improvement; ban "let me just try this" submits. |
| Coordination cost across 3 teammates eats time | Medium | Medium | First-meeting deliverable: who owns what (data/IRT/LLM-judge/report). |

---

## 7. Blockers — resolved + open

**Resolved 2026-05-10:**

| # | Item | Answer |
| --- | --- | --- |
| 1 | Team composition | 3 people. |
| 2 | Codabench registration | Done. |
| 3 | Modal $500 credit | Redeemed (×3 = $1,500 effective). |
| 4 | Deadline | 2026-05-22. First team meeting still to schedule. |
| 5 | Submission cap | 50/team/day, **1,000 total**. |
| 6 | Compute | Primarily Modal. |

**Still open (target: first team meeting):**

| # | Item | Why it matters |
| --- | --- | --- |
| 7 | Report ownership | Report is 50% of grade. Need a designated owner + section assignments by Day 1 so drafting starts in parallel with modeling. |
| 8 | LLM-judge yes/no (soft) | Affects Day 6 fork. Can defer the decision until Day 5 once IRT validation numbers are in. |

**First-meeting agenda I recommend:**

1. Confirm baseline-release date and current public leaderboard score.
2. Agree on per-person ownership: (a) data + validation harness, (b) IRT + content head, (c) LLM-judge + adaptive labeling, (d) report (one person, with input from others).
3. Decide LLM-judge in/out by Day 5 if not now.
4. Agree on submission rules: every Codabench submission must beat current best on offline validation, no exceptions.
5. Modal coordination: who runs what jobs on whose credit; rough $-budget per phase.

---

## 9. Progress log

### Day 0 — 2026-05-10

- Project scaffold created: `data/`, `src/`, `scripts/`, `submissions/`, `.gitignore`.
- Smoke-test submission built: `submissions/smoke_test.zip` (model.py + labeling.py at top level, both unchanged from `sample_code_submission/`). **Ready for any teammate to upload to Codabench.**
- `pyproject.toml` + `.python-version` (3.12) + `uv.lock`. Env at `.venv/`. Deps: torch 2.11, transformers 5.8, sentence-transformers 5.4, datasets 4.8, scikit-learn 1.8, pyarrow 24, numpy 2.4.
- `scripts/smoke_test.py` written and **passing** on `submissions/smoke_test/`: `predict()` returns native float in [0,1] across three synthetic inputs, with `labeled=None`, with a labeled list of length 2. `acquisition_function()` returns a finite native float (won't trigger the random fallback).
- `scripts/download_data.py` **executed.** ~172 MB pulled (16 response tables + 3 registry tables; traces skipped). Wrote `data/{responses,items,subjects,benchmarks,joined}.parquet`. Total 547 MB on disk, 5.36M raw responses → 4.44M binary rows after `--binary-only` filter.
- `scripts/eda.py` **executed.** Output saved to `data/eda_summary.txt`. Headlines: 909 subjects, 70,873 items, 16 benchmarks, 215 conditions, P(label=1)=0.6529. Key findings:
  - **subject_content is name-only.** No Organization/Parameters/Released/Family in data. Stage 2b stays as pure name → θ̂ lookup.
  - **Two degenerate benchmarks:** `ultrafeedback` (101k rows, all=1) and `mtbench` (1.1k rows, all=1). Drop or relabel before training (~2.3% of rows).
  - **rewardbench is 10% of all data** (446k binary rows, P=0.71). Keep.
  - **Massive difficulty spread per benchmark** (cybench 9.6% → mmbench 80.7%). Strong prior that IRT will beat NCF.
  - **215 conditions is too many for one-hot.** Mostly MMBench skill tags. Decide on Day 2: embed or prefix-collapse.
  - **Item length p99 = 2457 chars, max = 24,770.** Most fit in 512-token encoder context; top 1% will get truncated.
  - **Half the benchmarks are multimodal/agentic** (ai2d, mathvista, mmbench, agentdojo, androidworld, swebench). `item_content` for these may not encode the actual image — need to spot-check what we get.
- `src/validation.py` written. Item-cold-start split, benchmark-cold-start split, mean log-likelihood + ROC-AUC scoring helpers. Sanity-checked locally:
  - `mean_log_likelihood` of constant 0.5 = ln 0.5 = −0.6931 (matches our floor for the constant-0.5 smoke test).
  - `auc_roc` returns 1.0 for a perfect ranker, 0.0 for the inverted ranker.
  - `item_cold_start_split` produces train/val partitions with no item overlap.

### Day 0 — afternoon update (2026-05-10)

- **Research synthesis** — three parallel researchers + evaluator pass produced `research-report.md` / `research-report.html`. Headline: AIMS lab itself published the canonical reference (Truong et al., ICML 2025, arXiv:2503.13335), reporting column-holdout AUC ≈ 0.804. Mirror that pipeline.
- **Design-decisions agenda** — added §0.5 to this plan (and matching `#decisions` section in plan.html) with pre-meeting prep, pipeline recommendation table, decisions to produce, stop-loss criteria, calibration discipline, and a 3-line follow-up template.
- **Concrete task breakdown** — added §0.5.8 with 7-task replication backbone + 9 score-higher levers ranked by ROI.
- **IRT pipeline scaffolding** — wrote (syntax-checked, runnable once data lands):
  - `scripts/fit_irt.py` — Rasch/2PL fit using PyTorch + Adam on BCE; outputs `data/irt/{theta,b,log_a}.pt` + lookup JSONs.
  - `scripts/encode_items.py` — sentence-transformer encoding with deduplication; defaults to `all-mpnet-base-v2`; writes `data/embeddings/item_embeddings.npy`.
  - `scripts/train_content_head.py` — linear probe or MLP regressor on (embedding → b); validates on item-cold-start split; writes `data/head/head.pt` + `head_meta.json`.
  - `scripts/build_submission.py` — packages `submissions/<name>/` + `data/{irt,head}/*.pt|*.json|*.npy` into a flat ZIP.
  - `submissions/v1_irt/model.py` — full inference path: encoder + head + θ lookup + sigmoid combine + clip to [0.02, 0.98]. Honors the `predict(input, labeled)` contract.
  - `submissions/v1_irt/models.txt` — declares the encoder repo.

### What's blocked on the team / user

- **Upload `submissions/smoke_test.zip` to Codabench.** Confirms the pipeline scores submissions and gives us our floor (~−0.693 mean log-likelihood).
- **Run `scripts/download_data.py`** somewhere with HF auth + disk space. Then `scripts/eda.py` to see real numbers.
- **Locate `torch_measure`** package on course infrastructure — if found, swap into `scripts/fit_irt.py` instead of our hand-rolled version.
- **First team meeting** — agenda in §0.5; decisions list in §0.5.4.

---

## 10. Where to start (Day 0 work, in flight today)

Tasks I'm prepping autonomously while logistics finalize:

1. **Smoke-test ZIP.** `submissions/smoke_test.zip` containing `sample_code_submission/{model.py,labeling.py}` unchanged. Ready for any teammate to upload.
2. **Project scaffold.** `src/`, `data/`, `scripts/`, `submissions/` directories. Light `.gitignore` so HF cache and Modal artifacts stay out of git.
3. **Data downloader script.** `scripts/download_data.py` using the explicit-files loader from `README.md`, skipping `*_traces.parquet`. Runs once, caches to `data/`.
4. **EDA skeleton.** `scripts/eda.py` (or `.ipynb`) producing the one-page summary: #subjects, #items, #benchmarks, label balance overall + per-benchmark, `subject_content` field coverage.
5. **Validation harness skeleton.** `src/validation.py`: item-cold-start split, mean-log-likelihood + AUC-ROC scoring helper.

Everything depends on the validation harness — that's why it's first after the data download. Without it, every Codabench submission is a blind shot at the daily budget.

Updates to this plan will be appended to `plan.md` and `plan.html` as work progresses (a "Progress log" section will be added once Day 0 work lands).
