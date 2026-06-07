# v1_subject_ability_v5 — team-best submission (-0.59)

## What the model does

For each `(subject, item, benchmark, condition)` query, the model estimates the subject's
ability purely from the anchor labels revealed that round and maps it to a pass probability.
First it computes a Laplace-smoothed per-subject accuracy, converts it to an ability logit, and
shrinks it toward the group's trimmed-mean ability with strength `~1/(1 + n/SHRINK_N0)`. It then
fits a per-benchmark logistic calibration `p = sigmoid(rho * z + beta0)` by penalized NLL over the
in-category anchors. Finally it blends the calibrated probability toward the benchmark base rate
until enough in-category anchors have been seen, and clips to `[0.02, 0.98]`. All state is derived
from the `labeled` argument at call time — there is no training, no disk I/O, and no network access;
it is CPU-only online inference.

## Files in the submission

- `model.py` — `predict(input, labeled)` entry point (the full inference logic above).
- `labeling.py` — `acquisition_function(input)`; returns a constant so the platform falls back to
  uniform-random label selection (steering acquisition did not help).
- `requirements.txt` — explicit runtime dependencies (`numpy`, `scipy`).
- `README.md` — this file.

## models.txt — not needed

There is **no** `models.txt`. That file is only required when a submission downloads HuggingFace
models at runtime. This submission uses no pretrained weights, no embeddings, and makes no network
calls, so there is nothing to declare. The submission ZIP is just the two `.py` files (plus
`requirements.txt`); no data files or checkpoints are bundled.

## Dependencies

- `numpy` (>= 1.26)
- `scipy` (>= 1.11) — used for `scipy.optimize.minimize` (L-BFGS-B calibration fit) and
  `scipy.special.expit`.

Both are present on the Codabench platform runtime (numpy directly; scipy ships with the platform's
scientific Python stack). They are declared explicitly in `requirements.txt`.

## Reproduce / verify

Build the submission ZIP (small; only `model.py`, `labeling.py`, `requirements.txt` are packed):

```
make submission NAME=v1_subject_ability_v5
```

Local CPU smoke test (prints `PASS`):

```
uv run python scripts/smoke_test.py submissions/v1_subject_ability_v5
```

Unit tests (added by a parallel worker):

```
uv run python tests/test_v1_subject_ability_v5.py
```

One-command build + smoke:

```
make submission-v5
```

## Leaderboard

This submission scored **-0.59** on the Codabench leaderboard — the team's best result.
