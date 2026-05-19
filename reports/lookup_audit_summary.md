# Lookup Normalization Audit

## Artifacts

- Pre-fix debug zip: `submissions/v1_kfactor_lookup_audit.zip`
- Fixed debug zip: `submissions/v1_kfactor_lookup_audit_fixed.zip`
- Both zips were packed with runtime state from the locked best zip, `../../../submissions/v1_kfactor_priors_only_tmp.zip`, so the audit compares lookup code only.
- No Codabench upload was performed.

## Phase 0 Training-Format Counter Table

Local proxy-helper audit over the corrected proxy sampling path, 3 seeds x 5,000 predictions:

| outcome | count | pct |
|---|---:|---:|
| hit_subject_category | 15000 | 100.000% |
| hit_subject_benchmark | 0 | 0.000% |
| hit_subject | 0 | 0.000% |
| hit_benchmark_condition_only | 0 | 0.000% |
| hit_benchmark_only | 0 | 0.000% |
| fell_to_global | 0 | 0.000% |
| prior_none | 0 | 0.000% |

Sample buffer excerpts: none. No training-format rows fell below `hit_subject_benchmark`.

## Phase 0 Synthetic Confusion Table

Pre-fix audit on 200 dense `subject_category` cells (`n >= 20`), using the locked prior payload:

| transformation | hit_subject_category | hit_subject_benchmark | hit_subject | hit_benchmark_condition_only | not_applicable |
|---|---:|---:|---:|---:|---:|
| control_original | 200 | 0 | 0 | 0 | 0 |
| subject_lowercase_prefix | 0 | 0 | 0 | 200 | 0 |
| subject_uppercase_prefix | 0 | 0 | 0 | 200 | 0 |
| subject_no_prefix | 200 | 0 | 0 | 0 | 0 |
| subject_leading_whitespace | 200 | 0 | 0 | 0 | 0 |
| subject_trailing_newline | 200 | 0 | 0 | 0 | 0 |
| subject_unicode_nfd | 200 | 0 | 0 | 0 | 0 |
| benchmark_titlecase | 0 | 0 | 200 | 0 | 0 |
| benchmark_trailing_space | 0 | 0 | 200 | 0 | 0 |
| condition_trailing_space | 0 | 200 | 0 | 0 | 0 |
| condition_pipe_spaces | 0 | 8 | 0 | 0 | 192 |

Phase 0 decision: proceed. Plausible Codabench-format variants caused deterministic misses from existing dense cells.

## Phase 1 Diagnosis

The dense prior keys existed under canonical training-time forms, but predict-time lookup did not canonicalize several raw input variants. Lowercase or uppercase `name:` prefixes missed the case-sensitive `Name:` regex and normalized to the full string (`name: <display>`), causing subject lookup failure and falling to benchmark-condition-only. Title-cased or trailing-space benchmarks missed benchmark-specific keys and fell to subject-only. Conditions with trailing whitespace or spaces around `|` missed subject-category keys and fell to subject-benchmark. Existing prior keys were inspected before the fix: all 16 benchmark keys were already stripped/lowercase with no collisions, and condition `strip()` plus `|` whitespace normalization changed/collided with 0 existing condition keys. Condition case was not lowercased because 136 existing condition strings contain uppercase criterion text.

## Phase 2 Fix

Surgical runtime changes:

- `NAME_LINE` now uses `re.IGNORECASE` in both runtime and training helper normalization.
- Runtime benchmark lookup key is `strip().lower()`.
- Runtime condition lookup key is `strip()` with whitespace around `|` collapsed to a bare pipe.
- No priors were regenerated because existing training keys are unchanged by these transforms.

Post-fix synthetic table:

| transformation | hit_subject_category | not_applicable |
|---|---:|---:|
| control_original | 200 | 0 |
| subject_lowercase_prefix | 200 | 0 |
| subject_uppercase_prefix | 200 | 0 |
| subject_no_prefix | 200 | 0 |
| subject_leading_whitespace | 200 | 0 |
| subject_trailing_newline | 200 | 0 |
| subject_unicode_nfd | 200 | 0 |
| benchmark_titlecase | 200 | 0 |
| benchmark_trailing_space | 200 | 0 |
| condition_trailing_space | 200 | 0 |
| condition_pipe_spaces | 8 | 192 |

Post-fix training-format audit remained 15,000/15,000 `hit_subject_category`.

## Phase 3 Proxy MLL

Modal proxy, identical args: `--m-categories 5 --k 5 --max-rows 5000 --per-category 1000 --seeds 0,1,2`.

| zip | mean MLL | std | seed MLLs |
|---|---:|---:|---|
| locked original | -0.454697 | 0.004744 | -0.458337, -0.449331, -0.456422 |
| fixed | -0.454697 | 0.004744 | -0.458337, -0.449331, -0.456422 |

Decision: the bug is real synthetically, but the proxy cannot see leaderboard impact because proxy inputs are already training-formatted. The fixed zip is proxy-neutral within +/-0.005. User should decide whether to upload.

Cell-count bucket breakdown from the same sampled proxy rows:

| bucket | original MLL mean | fixed MLL mean | n mean |
|---|---:|---:|---:|
| missing | n/a | n/a | 0.0 |
| n<20 | -0.301522 | -0.301522 | 276.3 |
| 20<=n<50 | -0.298680 | -0.298680 | 300.3 |
| 50<=n<100 | -0.534649 | -0.534649 | 384.7 |
| n>=100 | -0.469190 | -0.469190 | 4038.7 |

Modal run notes:

- Original run: `https://modal.com/apps/junyiji3/main/ap-tVsdiboAqeZznCPWh7oUHd`
- Final fixed run: `https://modal.com/apps/junyiji3/main/ap-hsRLOy8KKRN3XSlPXhEfum`

## Surprises

- The current local `data/stage2/priors_v1/runtime_priors.json` is not byte-equivalent to the locked best zip's `runtime_priors.json`. A fixed zip built from current local priors scored `-0.460341 +/- 0.004689`, a regression unrelated to the normalization fix. The final handoff zips were therefore repacked with the locked zip's prior payload.
- This checkout's `modal_eval_submission.py` does not expose the documented `--out` flag, so local JSON reports were generated via the same proxy helper functions, while Modal logs provide the remote MLL comparison.
- One Modal worker for the final fixed run was killed with SIGKILL and retried automatically; the retry completed successfully with the parity scores above.
