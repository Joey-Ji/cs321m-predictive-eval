# Item Residual Summary

## 1. Phase 0

- OOF item targets built from `data/joined.parquet` with 5 item-disjoint folds, locked Canon V2 priors, `lambda_item=10.0`, training clip `+/-0.50`.
- Delta distribution: mean `0.023541`, std `0.293253`, min/max `-0.500000 / 0.500000`.
- Tails: `27,088` items with `|delta| > 0.25`; `19,116` clipped at `+/-0.50`.
- Leakage spot-check: sampled items were assigned to one fold and absent from the corresponding prior-fit training rows.

## 2. Phase 0 Decision

Proceed. Target std is above the `0.05` stop threshold.

## 3. Phase 1

Weighted ridge on `[item_content mpnet embedding, benchmark one-hot, condition one-hot]`.

Best holdout row-BCE setting:

| alpha | w | row MLL | gain | weighted RMSE | corr |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.50 | -0.513452 | +0.009472 | 0.375248 | 0.322518 |

## 4. Phase 1 Decision

Proceed to full-data artifact. Best holdout gain exceeded the `+0.003` MLL gate.

## 5. Phase 3

Modal baseline, 10 seeds, identical corrected proxy args:

- `v1_kfactor_canon_v2.zip`: `-0.453164 +/- 0.003791`

Modal candidate package probe:

- seed 0 only completed due serial encoder cost: candidate `-0.456158`; same-seed baseline from the Modal run was `-0.458337`, gain `+0.002179`.

Cached-embedding corrected proxy fallback, 10 seeds:

| model | MLL mean | MLL std |
|---|---:|---:|
| baseline priors | -0.453164 | 0.003791 |
| item residual | -0.450326 | 0.003981 |
| gain | +0.002838 | 0.000428 |

Per-seed sign consistency: `10/10` seeds improved.

## 6. Phase 3 Decision

Stop. Mean gain `+0.002838` is below the required `+0.005` MLL gate, even though all seeds improved. Treat this as a null finding, not a recommended upload.

## 7. Phase 4

Smoke output from `scripts/smoke_item_residual.py`:

- `PRIOR_ONLY=True`
- `ITEM_RESIDUAL_OK=True`
- `weight_w=0.5`
- runtime delta clip `+/-0.25`
- dense-cell composition matched `sigmoid(prior_logit + w * delta)`
- unseen item and cold subject paths produced finite bounded deltas

## 8. Files Changed

- Added OOF target generation, ridge training, cached proxy eval, local eval fallback, and item-residual smoke scripts.
- Added runtime item-residual loader/inference in `submissions/v1_kfactor/model.py`.
- Added build allowlist entries for `item_residual_model.pt` and `item_residual_meta.json`.
- Added tests in `tests/test_item_residual.py`.
- Reports: Phase 0, Phase 1, Phase 3 cached proxy, smoke, and this summary.

## 9. Surprises Or Cautions

- The residual generalizes directionally but not strongly enough: 10/10 cached-proxy seeds improve, but the average gain is below the predeclared gate.
- Full Modal candidate validation is impractical with serial predict-time mpnet encoding. A one-seed package probe completed after embedding prewarm; the 10-seed decision used the cached local proxy.
- `submissions/v1_kfactor_item_residual_v1.zip` exists as a candidate artifact for inspection only. Do not upload based on this run.

## 10. PR URL

https://github.com/Joey-Ji/cs321m-predictive-eval/pull/22
