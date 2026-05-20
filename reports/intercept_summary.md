# Online Intercept Summary

## Phase 0 Implementation

Implemented a prior-only global online intercept in `submissions/v1_kfactor/model.py`. In `PRIOR_ONLY` mode, `predict()` now adds `_online_intercept_delta(labeled)` to the canonicalized prior logit before sigmoid/clipping. The delta uses one Newton step at delta=0 with gradient `sum(y - p)`, Hessian `sum(p * (1 - p)) + lam`, and hard clipping to `[-clip, clip]`.

Config is loaded from flat `intercept_config.json` with defaults `lam=50.0`, `clip=0.15`. The cache is keyed by the existing labeled-list hash shape. The calibration, online Platt, Lever F per-subject shifts, residual MLP, and locked prior artifacts were not modified.

Validation:

| Check | Result |
| --- | --- |
| `python tests/test_online_intercept.py` | PASS |
| `python tests/test_canonicalization_v2.py` | PASS |
| `python tests/test_modal_eval_submission.py` | PASS |
| `python -m py_compile ...` | PASS |
| `git diff --check` | PASS |

## Phase 1 Proxy

Literal Modal execution was attempted with `modal run modal_eval_submission.py ... --out reports/intercept_control_canon_v2.json`, but the sandbox could not connect. The required network escalation was rejected because it would upload the submission zip to Modal. I used `scripts/local_eval_submission.py`, which calls the same evaluator helper functions locally on `joined.parquet`. Because these zips are `PRIOR_ONLY`, the fallback sets `V1_KFACTOR_DUMMY_ENCODER=1`; encoder embeddings are not used by `_raw_logit` in the prior path.

Control `v1_kfactor_canon_v2.zip`: -0.453164 +/- 0.003791 over seeds `0,1,2,3,4,5,6,7,8,9`.

| lam | clip | MLL mean | MLL std | delta vs control | sign consistency |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 0.05 | -0.453570 | 0.003648 | -0.000406 | 4/10 |
| 25 | 0.10 | -0.453997 | 0.003752 | -0.000833 | 4/10 |
| 25 | 0.15 | -0.454097 | 0.003920 | -0.000933 | 4/10 |
| 25 | 0.25 | -0.454097 | 0.003920 | -0.000933 | 4/10 |
| 50 | 0.05 | -0.453487 | 0.003678 | -0.000322 | 4/10 |
| 50 | 0.10 | -0.453527 | 0.003738 | -0.000362 | 4/10 |
| 50 | 0.15 | -0.453527 | 0.003738 | -0.000362 | 4/10 |
| 50 | 0.25 | -0.453527 | 0.003738 | -0.000362 | 4/10 |
| 100 | 0.05 | -0.453311 | 0.003729 | -0.000147 | 4/10 |
| 100 | 0.10 | -0.453311 | 0.003729 | -0.000147 | 4/10 |
| 100 | 0.15 | -0.453311 | 0.003729 | -0.000147 | 4/10 |
| 100 | 0.25 | -0.453311 | 0.003729 | -0.000147 | 4/10 |

## Phase 1 Decision

STOP. No candidate beat control by `+0.005` MLL, and no candidate reached the `7/10` sign-consistency gate. The least-bad candidates were `lam=100` with any tested clip, at `-0.453311 +/- 0.003729`, which is `-0.000147` below control with `4/10` positive seed diffs.

## Phase 2 Final Zip

Not built. The decision rule says not to ship when the grid is a clean null. Candidate zips were generated for evaluation under `submissions/v1_kfactor_intercept_lam*_clip*.zip` and are gitignored, but there is no `submissions/v1_kfactor_online_intercept.zip` handoff artifact from this run.

## Phase 3 Smoke

No final zip was produced, so final smoke is not applicable. The smoke script did pass against representative candidate zips; the saved best-candidate smoke is `reports/intercept_smoke_lam100_clip0.05.json`, confirming `PRIOR_ONLY=True`, config loading, empty-label equivalence, positive/negative label directionality, cold-subject global fallback, clip bounding, and no `labeling.py` in the manifest.

## Files Changed

| File | Description |
| --- | --- |
| `submissions/v1_kfactor/model.py` | Adds config-loaded, cached prior-only online intercept and applies it only in the `PRIOR_ONLY` branch. |
| `submissions/v1_kfactor/intercept_config.json` | Default intercept config artifact. |
| `scripts/build_submission.py` | Adds `intercept_config.json` to runtime state, excludes `labeling.py` for `v1_kfactor`, and supports generated intercept config overrides for grid builds. |
| `modal_eval_submission.py` | Adds `--out` support and returns per-seed raw results in the output payload. |
| `scripts/local_eval_submission.py` | Local fallback wrapper around the same corrected-proxy evaluator helpers. |
| `scripts/smoke_online_intercept.py` | In-process smoke for online-intercept zips. |
| `tests/test_online_intercept.py` | Unit tests for empty labels, Newton formula, clip threshold, and skipped non-binary labels. |
| `reports/intercept_*.json` | Control, grid, and smoke outputs. |
| `reports/intercept_grid_summary.json` | Machine-readable grid summary and decision. |

## Surprises

The online intercept was slightly negative on this corrected-proxy fallback even with heavy shrinkage. Several clip values tie exactly because the unclipped Newton step did not exceed those clip thresholds for the sampled rounds. AUC is unchanged across all variants, which is expected for a global per-round logit shift.
