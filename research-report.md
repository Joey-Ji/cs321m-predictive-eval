# Research Report — Modeling Approaches for the Predictive AI Evaluation Challenge

**Prepared:** 2026-05-10 — for first team-meeting design decision.
**Method:** Three parallel researcher agents (classical psychometrics; neural/LLM predictors; PGE literature + competition strategy) synthesized and evaluated.
**Confidence:** High on the primary recommendation; medium on encoder/architecture details; some citations corrected after evaluator review.

---

## 1. Executive Summary

The competition organizers (Stanford AIMS Lab, led by Prof. Sanmi Koyejo with **Sang Truong** on the teaching team) have published their own reference pipeline: **Truong et al. 2025, "Reliable and Efficient Amortized Model-based Evaluation"** (ICML 2025; arXiv:2503.13335). It implements Rasch IRT on training responses + a linear probe on Llama-3-8B embeddings predicting item difficulty for new items. The AIMS textbook (Chapter 3, "Learning") reports column-holdout (item cold-start) AUC ≈ **0.804** for this approach — this is both the canonical reference architecture and our explicit empirical target.

**Recommendation:** mirror the Truong et al. pipeline as the primary submission; develop a local LLM-as-judge in parallel as an ensemble component; commit to a disciplined calibration regime throughout.

---

## 2. Landscape Overview

Approaches sit in three families:

1. **Latent-factor models (IRT family).** 1PL/Rasch, 2PL, 3PL, MIRT. Naturally calibrated probability outputs. Requires a text→item-parameters regressor to handle cold-start. The AIMS reference and the entire educational-testing literature on text-based item difficulty (R2DE, BERT-IRT, AutoIRT) sit here.
2. **Pure neural predictors.** Sentence-transformer + MLP, NCF (NeuMF), FT-Transformer, GNNs, transformers from scratch. Flexible; tends to require explicit calibration; loses the IRT structural decomposition.
3. **LLM-as-judge.** Local 7B-class instruction model; read p(yes)/p(no) from logprobs. High ceiling, high latency, miscalibrated by default.

Plus three approaches that are NOT recommended for this regime: matrix factorization (LightFM, FM, FFM); GNNs; transformer-from-scratch on response data. See §6.

---

## 3. Detailed Findings

### 3.1 IRT family

| Model | Form | Item params | Subjects/item needed | Notes |
| --- | --- | --- | --- | --- |
| 1PL (Rasch) | σ(θ_s − b_i) | 1 (b) | 50+ with priors; 100-250 ideal | AIMS reference; preferred for limited subjects, lower overfitting |
| 2PL | σ(a_i · (θ_s − b_i)) | 2 (a, b) | 250-500 | More expressive; standard choice when data permits |
| 3PL | c + (1−c)·σ(a·(θ−b)) | 3 (a, b, c) | 1000+ | NOT RECOMMENDED — guessing parameter inappropriate for LLMs |
| MIRT | σ(a_i · (θ_s − b_i)), θ vector | many | many | Worth considering only if multi-skill structure evident |

Subjects-known + items-new is unusual: typical psychometric setting has the cold-start on the student side. Here, θ_s comes from Stage 1 fitting with potentially thousands of training responses per subject — high-quality, stable inputs to Stage 3.

Tooling: `py-irt` (Lalor & Rodriguez, INFORMS J. Computing 2022; arXiv:2203.01282), `IRTorch`, `deepirtools`. Plus organizer-provided `torch_measure` (course-internal; not on public PyPI/GitHub — see action item §8).

### 3.2 Amortized IRT + content head — THE primary approach

This is what Truong et al. 2025 implement, what the spec describes as "Stages 1, 2a, 2b, 3," and what every related literature (R2DE, BERT-IRT, AutoIRT) converges on.

- **Stage 1:** Fit Rasch (or 2PL) on training response matrix. Recover θ̂_s for every subject and (b̂_i, optionally â_i) for every training item.
- **Stage 2a:** Train regressor: item text embedding → predicted (b̂, â). Reference architecture from Truong et al. = **Llama-3-8B embeddings (4096-d) → linear probe + EM iteration**. Practical alternatives: bge-large-en-v1.5 (335M, MTEB 64.23, 1024-d), all-mpnet-base-v2 (110M, MTEB 57.78, 768-d, already in starter kit's `models.txt`). Add side features: benchmark, condition, item length, structural metadata.
- **Stage 2b:** For known subjects, lookup θ̂_s from Stage 1.
- **Stage 3:** P = σ(â · (θ̂ − b̂)).

Key empirical anchor: AIMS textbook reports column-holdout AUC ≈ 0.804 for this pipeline (verify exact split/metric definition in textbook Chapter 3 before quoting in report). Difficulty-prediction R² in related literature ranges 0.19-0.62 depending on dataset; even R²=0.3 substantially improves log-loss over a flat difficulty prior.

### 3.3 Pure content MLP / NCF (the §3.4 spec baseline)

- Frozen sentence-transformer + small MLP, BCE on response label.
- NeuMF variant: combine GMF (Hadamard product) path + MLP (concat) path. He et al. 2017 (WWW; arXiv:1708.05031) shows the union beats either alone.
- Same cold-start handling as IRT-amortized (text encoder generalizes).
- Lacks IRT's structured decomposition into ability × difficulty. Less interpretable; weaker report story.
- Two-tower vs. interaction: ContextGNN (arXiv:2411.19513, Nov 2024 preprint) shows interaction models outperform two-tower by ~20% on average on standard rec benchmarks (this gap shrinks under cold-start).
- Best encoder: bge-large-en-v1.5 if open-source preferred; e5-mistral-7b-instruct if compute permits. all-mpnet-base-v2 is the spec's default.

### 3.4 LLM-as-judge

- Frame as yes/no. Read next-token logits of "yes"/"no". Return p(yes)/(p(yes)+p(no)).
- Candidate models: Qwen2.5-7B-Instruct (MMLU 74.2, MT-Bench 87.5; arXiv:2412.15115), Llama-3.1-8B-Instruct, Gemma-2-9b-it.
- Latency: with vLLM batching, ~30-60s for 1000 calls on L4/A100. Sequential = ~43× slower.
- Raw chat-LLM logits are miscalibrated (arXiv:2402.13213). **Venn-Abers post-hoc calibration** is the recommended fix (Giovannotti & Gammerman, COPA 2024; arXiv:2407.01122 — temperature-invariant; substantial ECE reduction on Llama 2 7B yes/no questions).
- Few-shot variant: render labeled examples as in-context shots. "Calibrate Before Use" (Zhao et al., ICML 2021; arXiv:2102.09690) shows up to 30% absolute accuracy gain from in-context calibration, but few-shot adds latency proportional to shot count.
- Best framing: ensemble component, not primary.

### 3.5 Rubric-based demand profiles ("General Scales")

- Zhou et al., 2025 preprint (arXiv:2503.06378). 18 cognitive demand rubrics (attention, comprehension, reasoning, knowledge, etc.) annotated by GPT-4o.
- Reportedly outperforms raw embedding regression for OOD prediction.
- For our competition: would need offline GPT-4o annotation of training items (allowed) + a local LLM at test time (Qwen-72B?) to annotate test items the same way (since outbound network is forbidden at test). Significant inference cost.
- Worth a stretch goal in last 3-4 days only if the IRT-content path is bottlenecking on item-text features.

### 3.6 IRT × neural hybrids in the LLM era

Three 2025 papers all converge on the IRT + neural item embedding architecture:

- **Truong et al. (ICML 2025)** — the AIMS reference; Llama-3-8B embeddings + linear probe + EM.
- **Chen et al., "Learning Compact Representations of LLM Abilities via IRT"** (arXiv:2510.00844, 2025 preprint) — Mixture-of-Experts over sentence-transformer embeddings; reportedly SOTA on LLM routing + benchmark accuracy prediction.
- **PSN-IRT** (arXiv:2505.15055) — Pseudo-Siamese network for IRT on 11 LLM benchmarks (41,871 items); AAAI 2026 oral.

The cross-validation across these independent groups is strong evidence we're picking the right architecture family.

---

## 4. Approaches NOT recommended

| Approach | Why not |
|----------|---------|
| LightFM / FM / FFM | Not probability-calibrated by default; degrades when test items have no feature overlap; no advantage over IRT-amortized given subjects-known |
| 3PL IRT | Guessing parameter inappropriate for LLMs; needs N>1000 per item |
| GNNs (transductive) | Cannot generate embeddings for cold-start item nodes |
| GNNs (inductive) | Degenerates to content MLP without graph structure on item side |
| Transformer from scratch on response data | Insufficient data (~200K-2M rows) vs. transfer from pretrained sentence encoders |
| FT-Transformer | Marginal gain over plain MLP; not worth the complexity overhead |

---

## 5. Calibration strategy

Mean log-likelihood (the metric) punishes miscalibration heavily. Disciplined calibration is the highest-EV non-modeling work.

- **Training-time:** Label smoothing (ε=0.05-0.10) baked into BCE loss. Focal loss (Mukhoti et al., NeurIPS 2020) reduces ECE vs. plain BCE.
- **Post-hoc on held-out training fold:** Temperature scaling (Guo et al., ICML 2017; arXiv:1706.04599) — single scalar, optimize via cross-validated log-loss. The consensus best for small calibration sets. Avoid isotonic regression at N<100.
- **Per-round on revealed labels (the K=5/category recipe):** Each round we get exactly 25 labels, structured as 5 per category × 5 categories.
  - **Default — global Platt:** fit one logistic on all 25 (raw_score, label) pairs. Always works; low-variance.
  - **Per-category shifts (only if data permits):** within each benchmark, fit a per-category temperature *only if* the bucket has ≥3 labels of each class (otherwise the Platt overfits or degenerates). Shrink per-category fits toward the global Platt with a Bayesian prior (e.g., posterior mean weighted by bucket size).
  - **Order of operations inside `predict()`:** on first call of round, fit calibrators on `labeled`; cache in module global; pass every subsequent prediction through.
- **Output clipping:** clip predictions to [0.02, 0.98] to avoid catastrophic log(ε) on confident-wrong predictions.
- **For LLM-judge specifically:** Venn-Abers calibration on yes/no logits (Giovannotti & Gammerman, COPA 2024).

---

## 6. Active-labeling / acquisition

The K=5/category, 25-total label budget per round.

- **Round 1:** Diversity sampling. K-means centroids fit offline + per-round under-representation score (§3.6 of spec); OR online farthest-point sampling.
- **Round 2+:** Fisher-information / CAT criterion — select items where predicted difficulty b̂ is closest to current ability estimate θ̂ (maximizes information per label under Rasch).
- **Why diversity first:** uncertainty sampling requires a calibrated model; we don't have one at round 1.
- 2025 reference: CUSAL (arXiv:2510.03162) — calibrated uncertainty hybrid; outperforms BALD at early rounds.

---

## 7. Compute budget

| Budget item | Cost | Headroom on $1500 |
|---|---|---|
| Llama-3-8B embedding extraction over training set | ~$3 (rough; rerun once EDA gives row count) | trivial |
| Bge-large embedding extraction | ~$1 | trivial |
| Rasch/2PL IRT fit (CPU or single GPU) | <$1 | trivial |
| MLP head training | <$1 | trivial |
| LoRA fine-tune of 7B model (3-5 epochs, A100-40GB) | ~$10-15/run | 30-100 runs |
| Per-submission inference budget | ~$1-2/round | ample |

Modal pricing (May 2026): L4=$0.80/hr, A100-40GB=$2.10/hr, A100-80GB=$2.50/hr, H100=$3.95/hr. **Budget is not the binding constraint; submission cap (50/day, 1000 total) is.**

---

## 8. Day-0 action: locate `torch_measure`

The spec (`README.md` line 245) states: *"The organizers provide the `torch_measure` package to facilitate measurement model implementation."* It is not on public PyPI or GitHub — almost certainly course-internal (Codabench starter, course portal, or Canvas).

**Action:** before reimplementing IRT in PyTorch ourselves, find this package. Using the organizer-blessed package is signal-positive in the report and saves a day of plumbing. If after one focused search it can't be found, fall back to `py-irt` or `IRTorch`.

---

## 9. What wins similar competitions

- **NeurIPS 2020 Education Challenge** (predict student answer correctness, AUC primary): top teams used ensembles of 100+ models; target encoding + ensemble dominated.
- **Riiid! Kaggle 2021** (780K students, 13K questions, AUC): ensemble of 50+ models won (SAINT+ transformer winner; ensembles of DKT/SAKT/SAINT/LGBM in top 3).
- **Eedi 2024 Kaggle** (cold-start misconception items, MAP@25): Qwen2.5-32B + LoRA as retriever+reranker won; CoT reasoning on item text was the key transferable trick.
- Cross-cutting patterns: feature engineering ≈70% of improvement; ensembles routinely beat single best (3-8% log-loss gain typical from diverse 3-5-model log-mixture); calibration dominates when log-loss is the metric.

---

## 10. Recommendation

### 10.1 Primary pipeline (commit at meeting)

**Mirror Truong et al. 2025: amortized IRT + content head.**

- Stage 1: Rasch IRT (start here for stability; upgrade to 2PL if data supports per-item discrimination estimation).
- Stage 2a: bge-large-en-v1.5 (or all-mpnet-base-v2 — already in starter kit; faster smoke-test) → 2-layer MLP regressor → predicted (b̂_i, log â_i). Optionally concatenate benchmark/condition embeddings.
- Stage 2b: Subject ability lookup from Stage 1 by normalized name.
- Stage 3: P = σ(â · (θ̂ − b̂)).
- Calibration: temperature scaling on held-out training fold + per-round Platt via `labeled` argument inside `predict()` (see §5 for the K=5 recipe).

**Why this over alternatives:**
1. Mirrors the AIMS lab's own published method (Truong et al. ICML 2025) — the organizers explicitly designed the competition for this pipeline (`torch_measure`, the §3.4 NCF worked example, the labeling.py adaptive channel).
2. Naturally calibrated (the IRT logistic is a proper probability model).
3. Cleanest report story (decomposition into ability × difficulty; explicit Stage 1/2a/2b/3 mapping).
4. Fastest at inference (linear/MLP after embedding).
5. Realistic to beat 0.804 column-holdout AUC with thoughtful ablations + calibration discipline + ensembling.

### 10.2 Secondary pipeline (parallel branch, days 5-8)

**Local LLM-as-judge (Qwen2.5-7B-Instruct) with Venn-Abers calibration.**

- Declared in `models.txt`, vLLM-batched inference, p(yes)/p(no) logits.
- Few-shot variant via `labeled` examples (good ablation for the report).
- Goal: ensemble component, not standalone.

### 10.3 Final ensemble (days 9-11)

Log-mixture of (a) IRT-content predictor and (b) LLM-judge with weights tuned on validation. Standard log-loss competition pattern; near-universal +3-8% improvement over best single.

### 10.4 Stretch (only if days 7-9 free up)

Rubric-based regressor (§3.5): GPT-4o-annotated rubric features offline → calibrated regressor → local-LLM annotation at test time. High implementation cost; only pursue if the IRT pipeline plateaus.

### 10.5 Stop-loss / decision criteria

Make these binding so the meeting decision doesn't drift:

- **Day 4 checkpoint:** if IRT + content-head validation log-likelihood is *below* the NCF baseline (Day 2), debug the IRT fit before adding more components.
- **Day 6 LLM-judge go/no-go:** go ahead only if (a) the IRT pipeline is within 0.05 mean log-likelihood of the leaderboard top, AND (b) we have an engineer with bandwidth. Otherwise deepen IRT (hierarchical priors, benchmark-conditional difficulty, MIRT exploration).
- **Day 10 hard gate:** report draft must exist by end of day. Modeling iteration freezes; remaining time is calibration + ensemble tuning + writing.

### 10.6 Stretch features worth knowing about

- The `*_traces.parquet` files in the HF dataset (per `README.md` lines 137-139) contain raw model outputs. Trace length / coherence may correlate with item difficulty (subjects that emit long incoherent traces vs. short confident ones). Worth a one-paragraph experiment in the report if time permits.

---

## 11. Confidence

- **High confidence:** IRT-amortized as primary; calibration discipline (esp. temperature + clipping); what NOT to use; ensemble pattern. Three independent researchers converged on this; canonical reference paper exists (Truong et al.).
- **Medium confidence:** Choice of sentence transformer (Llama-3-8B vs. bge-large vs. mpnet — empirical question); LLM-judge ROI (depends on test distribution); 2PL vs. Rasch (depends on per-item subject counts).
- **Lower confidence:** Active-labeling strategy details with K=5; per-category vs. global Platt threshold; whether rubric features pay off in 12 days.

---

## 12. Gaps + caveats

- `torch_measure` not findable on public PyPI/GitHub. Day-0 action item to locate via course infrastructure.
- "PGE" terminology not found in published papers; appears to be course-internal language for what the textbook calls "Prediction-Powered Evaluation" / "amortized factor model."
- 0.804 column-holdout AUC anchor needs precise locator (textbook chapter/section + split definition) before quoting in the report.
- Direct head-to-head benchmarks of (sentence-transformer + MLP) vs. LLM-judge vs. IRT-amortized on the same dataset are scarce in the literature.
- Calibration with N=25 is under-studied relative to large-N regimes.

---

## 13. Sources

**Canonical references (load-bearing for the recommendation):**

- **Truong et al.** "Reliable and Efficient Amortized Model-based Evaluation." ICML 2025; arXiv:2503.13335. *AIMS lab canonical paper.*
- **AIMS textbook**, Chapter 3 (Learning). [aimslab.stanford.edu/textbook](https://aimslab.stanford.edu/textbook)
- **Stanford CRFM blog** (June 2025). "Reliable and Efficient Amortized Model-based Evaluation." [crfm.stanford.edu/2025/06/04/reliable-and-efficient-evaluation](https://crfm.stanford.edu/2025/06/04/reliable-and-efficient-evaluation.html)

**IRT and content-head literature:**

- **Lalor & Rodriguez.** py-irt. INFORMS J. Computing 2022; arXiv:2203.01282.
- **Benedetto et al.** "R2DE: NLP for IRT parameter estimation." LAK 2020; arXiv:2001.07569.
- **Sharpnack et al.** "AutoIRT." 2024 preprint; arXiv:2409.08823.
- **BERT-IRT.** BEA 2024 workshop. 10× pilot reduction with BERT + IRT.
- **Chen et al.** "Learning Compact Representations of LLM Abilities via IRT." 2025 preprint; arXiv:2510.00844.
- **PSN-IRT.** AAAI 2026 oral; arXiv:2505.15055.

**Calibration:**

- **Guo et al.** "On Calibration of Modern Neural Networks." ICML 2017; arXiv:1706.04599.
- **Mukhoti et al.** "Calibrating Deep Neural Networks using Focal Loss." NeurIPS 2020.
- **Giovannotti & Gammerman.** "Calibrated Large Language Models for Binary Question Answering." COPA 2024; arXiv:2407.01122.

**LLM-judge / NCF / encoders:**

- **He et al.** "Neural Collaborative Filtering." WWW 2017; arXiv:1708.05031.
- **Qwen Team.** Qwen2.5 technical report. 2024 preprint; arXiv:2412.15115.
- **bge-large-en-v1.5.** BAAI HuggingFace model card. MTEB avg 64.23.
- **Zhao et al.** "Calibrate Before Use." ICML 2021; arXiv:2102.09690.

**Competition / strategy / related:**

- NeurIPS 2020 Education Challenge results (arXiv:2104.04034).
- Eedi Mining Misconceptions Kaggle 2024.
- Modal pricing: [modal.com/pricing](https://modal.com/pricing).
- General Scales: Zhou et al., 2025 preprint; arXiv:2503.06378.
- ContextGNN: 2024 preprint; arXiv:2411.19513.
- tinyBenchmarks: arXiv:2402.14992.
- CUSAL active learning: arXiv:2510.03162.

---

## 14. Appendix — what NOT to discuss at the meeting

To keep the meeting on time, defer these:

- Specific IRT priors / regularization choices.
- LLM-judge prompt format details.
- Report section structure beyond "who's writing it."
- Calibration recipe (global vs. per-category Platt) — Day 5 conversation once we have validation numbers.
- Choice of bge-large vs. mpnet — empirical, decide on Day 2.
