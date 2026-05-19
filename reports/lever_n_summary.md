# Lever N Reliability Gate Results

## Phase 0 Split-Faithful Bucket Check

| Signal | Overall MLL | count<5 | 5<=count<20 | count>=20 |
|---|---:|---:|---:|---:|
| priors-only | -0.528500 +/- 0.028557 | -0.646543 +/- 0.058883 | -0.391736 +/- 0.278819 | -0.522292 +/- 0.033572 |
| K-factor base | -0.556812 +/- 0.001334 | -0.520365 +/- 0.274862 | -0.535477 +/- 0.291172 | -0.557235 +/- 0.015947 |
| K-factor - priors | -0.028312 +/- 0.027249 | +0.126178 +/- 0.241734 | -0.143740 +/- 0.023527 | -0.034943 +/- 0.027242 |

Decision: proceed by the specified rule, because K-factor beat priors by more than +0.01 MLL in `count<5`.

## Phase 3 Corrected Modal Proxy Grid

Same-build priors-only baseline is the `(0,0)` disabled-gate sanity run. The first grid used the raw K-factor base logit.

| T_low | T_high | Overall MLL | count<5 | 5<=count<20 | count>=20 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | -0.444766 +/- 0.004199 | -0.413383 +/- 0.002098 | -0.117543 +/- 0.006971 | -0.452665 +/- 0.005174 |
| 3 | 10 | -0.519163 +/- 0.003021 | -1.027374 +/- 0.032666 | -0.106763 +/- 0.006199 | -0.452665 +/- 0.005174 |
| 5 | 20 | -0.538161 +/- 0.003356 | -1.184804 +/- 0.039765 | -0.095649 +/- 0.005403 | -0.452665 +/- 0.005174 |
| 10 | 30 | -0.537926 +/- 0.003323 | -1.184804 +/- 0.039765 | -0.065725 +/- 0.003260 | -0.452714 +/- 0.005169 |

Decision: null result. Every nonzero gate sharply regresses overall MLL and the `count<5` bucket. No nonzero setting should be uploaded.

## Phase 3 Calibrated K-Factor Retry

After changing `_kfactor_base_logit()` to return `_apply_calibration(_base_logit_parts(input)[0])`, the nonzero gates improved relative to the raw-logit gate but still missed the disabled-gate baseline by a wide margin.

| T_low | T_high | Overall MLL | count<5 | 5<=count<20 | count>=20 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | -0.444766 +/- 0.004199 | -0.413383 +/- 0.002098 | -0.117543 +/- 0.006971 | -0.452665 +/- 0.005174 |
| 3 | 10 | -0.489428 +/- 0.003015 | -0.781324 +/- 0.018042 | -0.119514 +/- 0.007112 | -0.452665 +/- 0.005174 |
| 5 | 20 | -0.500526 +/- 0.002892 | -0.872561 +/- 0.021924 | -0.122573 +/- 0.007331 | -0.452665 +/- 0.005174 |
| 10 | 30 | -0.500486 +/- 0.002896 | -0.872561 +/- 0.021924 | -0.113765 +/- 0.006700 | -0.452713 +/- 0.005185 |

Decision: real null. Calibration reduced the sparse-bucket damage but did not produce a candidate worth uploading.

## Artifacts

- Phase 0 raw JSON: `reports/lever_n_phase0_buckets.json`
- Modal proxy raw JSON:
  - `reports/lever_n_modal_t0_0.json`
  - `reports/lever_n_modal_t3_10.json`
  - `reports/lever_n_modal_t5_20.json`
  - `reports/lever_n_modal_t10_30.json`
  - `reports/lever_n_modal_cal_t0_0.json`
  - `reports/lever_n_modal_cal_t3_10.json`
  - `reports/lever_n_modal_cal_t5_20.json`
  - `reports/lever_n_modal_cal_t10_30.json`
- In-process smoke raw JSON:
  - `reports/lever_n_smoke_t3_10.json`
  - `reports/lever_n_smoke_null.json`
  - `reports/lever_n_smoke_cal_null.json`
- Safe disabled-gate sanity zip: `submissions/v1_kfactor_lever_n_t0_0.zip`
- Safe disabled-gate handoff zip: `submissions/v1_kfactor_lever_n_null.zip`
- Calibrated-code disabled-gate handoff zip: `submissions/v1_kfactor_lever_n_cal_null.zip`

No nonzero Lever N upload candidate was selected.
