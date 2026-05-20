# Canonicalization V2 Summary

## Phase 0 Baseline Audit

Expanded audit against `/Users/joey/VScode/StanfordCS/CS 321M/eval_comp/submissions/v1_kfactor_lookup_audit_fixed.zip`, `n=200`, `min_count=20`, `seed=0`.

| Field | hit_subject_category | Miss path counts | Misses |
| --- | ---: | --- | ---: |
| subject | 1600 | hit_benchmark_condition_only=2600 | 2600 |
| benchmark | 2138 | hit_subject=262 | 262 |
| condition | 2164 | hit_subject_benchmark=236 | 236 |
| total | 5902 | hit_benchmark_condition_only=2600, hit_subject=262, hit_subject_benchmark=236 | 3098 |

High-miss mutators: 13 subject variants at 200/200 misses each, `condition_upper` at 200/200, `benchmark_insert_separator` at 126/200, underscore-to-space/hyphen at 68/200 each, `condition_none_to_empty` at 28/200, and `condition_lower` at 8/200.

Decision: proceed. Baseline misses exceeded the 50-miss threshold.

## Phase 1 Rules Shipped

| Rule | Justification |
| --- | --- |
| Subject prefix parser for `Name`, `Subject`, `Model`, `display_name` with optional markdown bullet/quote and ASCII/full-width colon | Recovers plausible Codabench subject wrappers without touching internal model-name tokens. |
| Subject NFKC + surrounding quote unwrap + trailing `.`, `,`, `;` removal only for extracted prefixed names | Handles Unicode and sentence-style wrappers while avoiding hyphen/underscore/space subject merges. |
| Benchmark NFKC + strip/lower + whitespace collapse | Safe extension of PR #19 normalization. |
| Benchmark separator-compaction aliases from existing prior keys only | Recovers hyphen/underscore/space/no-separator benchmark variants without hardcoded speculative aliases. |
| Condition NFKC + pipe spacing + internal whitespace collapse | Safe extension of PR #19 pipe normalization. |
| Condition case alias from existing prior keys only | Recovers case variants while returning the original locked condition key. |
| Empty condition aliases (`""`, `none`, `null`, `n/a`, `na`, `-`) to locked canonical `none` | Locked priors contain `none` and not `""`; this recovers blank/null Codabench condition forms. |

Skipped: aggressive condition synonym aliases (`cot`/`chain-of-thought`, `0-shot`/`zero-shot`, `few-shot`/`nshot`) because the locked condition vocabulary does not contain those forms and broad replacements inside criterion text are higher risk.

## Collision Checks

Source: `/Users/joey/VScode/StanfordCS/CS 321M/eval_comp/data/stage2/priors_v1_locked/runtime_priors.json`.

| Check | Collision buckets |
| --- | ---: |
| subject NFKC/quote/trailing-punctuation/lower | 0 |
| benchmark NFKC/strip/lower/whitespace | 0 |
| benchmark separator compact | 0 |
| condition NFKC/pipe/whitespace | 0 |
| condition lowercase alias | 0 |
| condition empty alias signature | 0 |

No whitelist merges were used.

## Phase 2 Post-Fix Audit

Expanded audit against `submissions/v1_kfactor_canon_v2.zip`, same sample and locked priors.

| Field | hit_subject_category | Misses |
| --- | ---: | ---: |
| subject | 4200 | 0 |
| benchmark | 2400 | 0 |
| condition | 2400 | 0 |
| total | 9000 | 0 |

## Phase 3 Proxy

Literal `modal run modal_eval_submission.py ...` was attempted but could not connect in the sandbox. The required network escalation was rejected because it would upload the submission zip to Modal. As a safer substitute, I ran the same evaluator helper functions locally on `joined.parquet` with `V1_KFACTOR_DUMMY_ENCODER=1`; this submission is prior-only, so encoder outputs are not used by `predict()`.

| Zip | MLL mean | MLL std |
| --- | ---: | ---: |
| locked baseline | -0.4546967372 | 0.0047441576 |
| candidate | -0.4546967372 | 0.0047441576 |
| delta | 0.0000000000 |  |

Result: proxy-neutral in the local corrected-proxy equivalent. Literal Modal validation remains blocked.

## Phase 4 Smoke

`python scripts/smoke_canon_v2.py --zip submissions/v1_kfactor_canon_v2.zip`

Known-subject variants produced identical probabilities across subject prefix/quote/punctuation variants, benchmark separator/case variants, condition case variants, and `none` aliases. Cold subject returned the clipped global prior.

Audit counters: `hit_subject_category=21`, `hit_benchmark_condition_only=1`, all other outcomes `0`. The one benchmark-condition-only outcome is the cold-subject probe; its returned probability matched the global prior.

## Files Changed

| File | Description |
| --- | --- |
| `submissions/v1_kfactor/model.py` | Canonicalization V2 runtime normalization for subject, benchmark, and condition lookup keys. |
| `scripts/lever_l_utils.py` | Mirrored subject parser for source-of-truth parity. |
| `scripts/audit_lookup_synthetic.py` | Expanded adversarial mutator matrix with per-field and aggregate path counts. |
| `scripts/smoke_canon_v2.py` | In-process zip smoke for the Canonicalization V2 variants. |
| `tests/test_canonicalization_v2.py` | Unit tests for variant normalization and collision safety over locked prior keys. |
| `reports/canon_v2_baseline_audit.json` | Phase 0 expanded baseline audit. |
| `reports/canon_v2_postfix_audit.json` | Phase 2 expanded post-fix audit. |
| `reports/canon_v2_collision_check.json` | Collision check report for shipped rules. |
| `reports/canon_v2_modal.json` | Local corrected-proxy equivalent plus literal Modal blocker note. |
| `reports/canon_v2_smoke.json` | Phase 4 smoke JSON output. |

Handoff zip: `submissions/v1_kfactor_canon_v2.zip`.

## Surprises

The strongest remaining baseline misses were subject formatting variants, not benchmark aliases. The locked proxy remained exactly neutral after the fix, matching the expected proxy-blind pattern.

## PR URL

Not opened. The task guardrail says not to push/open a PR until Phase 3 is complete; literal Modal validation is blocked by the external-upload restriction.
