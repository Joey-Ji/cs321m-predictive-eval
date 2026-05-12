# Stage 1: K-Factor IRT

This directory contains the Stage 1 response-matrix model used to estimate:

- subject/model capability vectors `U_i`
- subject/model bias terms `S_i`
- item loading vectors `V_j`
- item difficulty/bias terms `Z_j`

The fitted probability model is:

```text
P(y_ij = 1) = sigmoid(S_i + U_i dot V_j + Z_j)
```

For the first pass we set `K=4`, so each subject has a 4-dimensional
capability vector and each item has a 4-dimensional loading vector.

## What This Stage Does

This is a transductive fit over public training responses. It estimates latent
parameters for subjects and training items that appear in the public response
matrix. It does not yet solve item cold-start for hidden leaderboard items.

The intended use is:

1. Fit `U`, `V`, and `Z` on the public binary response matrix.
2. Use fitted subject vectors `U` and subject biases `S` as known subject/model
   capability estimates.
3. Use fitted item parameters `(V, Z)` as targets for a later content model that
   predicts item parameters from `item_content` for unseen items.

## Data Preparation

Do not check downloaded data or generated fitted artifacts into git.

From the repository root, download and join the public training data:

```bash
uv run python scripts/download_data.py --out data
```

This writes:

```text
data/responses.parquet
data/items.parquet
data/subjects.parquet
data/benchmarks.parquet
data/joined.parquet
```

The downloader filters the joined table to binary labels by default. In our
run this produced `4,443,797` binary rows.

## Final K=4 Artifact Run

For Stage 2 target generation, fit on all binary rows. This intentionally uses
`--val-frac 0` because the goal is to produce the best pseudo-gold training-item
parameters available from the public response matrix.

From the repository root:

```bash
uv run python stage_1/k_factor_irt/fit_k_factor_irt.py \
  --joined data/joined.parquet \
  --k 4 \
  --epochs 20 \
  --batch-size 65536 \
  --lr 0.05 \
  --lr-factor 0.5 \
  --lr-patience 2 \
  --min-lr 0.0001 \
  --weight-decay 0.0001 \
  --smoothing 20 \
  --val-frac 0 \
  --out stage_1/k_factor_irt/outputs/k4_full
```

## Diagnostic K=4 Replication Run

Use this when changing hyperparameters. It keeps a row-level validation slice so
you can catch instability, but those validation rows should be included again
for the final artifact run above.

From the repository root:

```bash
uv run python stage_1/k_factor_irt/fit_k_factor_irt.py \
  --joined data/joined.parquet \
  --k 4 \
  --epochs 3 \
  --batch-size 65536 \
  --lr 0.05 \
  --warmup-epochs 0 \
  --lr-factor 0.5 \
  --lr-patience 2 \
  --min-lr 0.0001 \
  --weight-decay 0.0001 \
  --smoothing 20 \
  --val-frac 0.02 \
  --out stage_1/k_factor_irt/outputs/k4_quick
```

For slower starts, add for example:

```bash
--lr 0.02 --warmup-epochs 2 --warmup-start-factor 0.1
```

The output directory is ignored by git. It will contain:

```text
subject_capabilities.csv  # subject_id, subject_bias, u_0, u_1, u_2, u_3
item_parameters.csv       # item_id, v_0, v_1, v_2, v_3, z
fit_summary.json          # config, counts, train/validation loss history
model_state.pt            # PyTorch state dict and id vocabularies
```

## Metrics From The Initial Dense Quick Run

Before adding subject bias, sparse optimization, marginal initialization, and
the LR schedule, the initial dense run used a deterministic row-level validation
holdout. We used row-level validation because item-cold-start validation would
hold out item vectors and bias terms that this Stage 1 model is explicitly
trying to fit.

```text
n_rows                         4,443,797
n_train_rows                   4,354,670
n_val_rows                        89,127
n_subjects                           909
n_items                           70,873
train_positive_rate              0.652883
val_positive_rate                0.651744
best_val_log_loss                0.325357
best_val_mean_log_likelihood    -0.325357
```

Loss history:

```text
epoch 1 train_log_loss=0.429022 val_log_loss=0.345932
epoch 2 train_log_loss=0.325680 val_log_loss=0.329251
epoch 3 train_log_loss=0.311787 val_log_loss=0.325357
```

## Sparse Subject-Bias Diagnostic Results

After adding smoothed marginal initialization, sparse embeddings, `SparseAdam`,
batch-local L2, the LR schedule, and subject bias, the best diagnostic run so
far is:

```text
lr=0.05, warmup_epochs=0, weight_decay=0.0001, smoothing=20
best_val_log_loss                0.320203
best_val_mean_log_likelihood    -0.320203
best_epoch                              3
```

Two slower-start checks were close but did not beat it:

```text
lr=0.02, warmup_epochs=2, warmup_start_factor=0.1  best_val_log_loss=0.320989
lr=0.05, warmup_epochs=2, warmup_start_factor=0.1  best_val_log_loss=0.320360
```

Validation log loss is the average negative log probability assigned to the
true held-out label. Lower is better. Mean log likelihood is the same value with
the sign flipped, so higher is better and it is usually negative.

The factor dimensions are not individually identifiable: signs and rotations can
change without changing predictions. Interpret `S_i + U_i dot V_j + Z_j`, not
an individual raw coordinate as a stable semantic axis.

## Residual MLP IRT Experiment

`fit_mlp_irt.py` keeps the same transductive subject/item embeddings and adds a
nonlinear residual on top of the interpretable IRT backbone:

```text
base_ij = S_i + U_i dot V_j + Z_j
features_ij = [S_i, Z_j, U_i, V_j, U_i * V_j, |U_i - V_j|]
logit_ij = base_ij + MLP(features_ij)
```

The default MLP is:

```text
input_dim = 2 + 4K
Linear(input_dim -> 128), LayerNorm, GELU, Dropout(0.10)
Linear(128 -> 64),        LayerNorm, GELU, Dropout(0.10)
Linear(64 -> 1)
```

The final MLP layer is zero-initialized, so training starts exactly at the IRT
backbone and learns nonlinear corrections from there. This is still not a
cold-start item model: item IDs are learned embeddings. Use it to test
nonlinear headroom and to generate richer Stage 1 pseudo-targets, then use
Stage 2 to learn item parameters from text/metadata.

Diagnostic command:

```bash
uv run python stage_1/k_factor_irt/fit_mlp_irt.py \
  --joined data/joined.parquet \
  --k 16 \
  --epochs 10 \
  --batch-size 65536 \
  --lr 0.001 \
  --embedding-lr 0.02 \
  --mlp-lr 0.001 \
  --warmup-epochs 1 \
  --warmup-start-factor 0.25 \
  --lr-factor 0.5 \
  --lr-patience 2 \
  --min-lr 0.00001 \
  --weight-decay 0.0001 \
  --smoothing 20 \
  --hidden-dims 128,64 \
  --dropout 0.1 \
  --residual-scale 0.5 \
  --val-frac 0.02 \
  --early-stop-patience 4 \
  --out stage_1/k_factor_irt/outputs/mlp_k16_diag_emb002_mlp0001_res05
```

This run reached validation log loss `0.313188` at epoch 2. That is a useful
nonlinear sanity check and slightly better than the original K=4 recipe, but it
did not beat the tuned K=16 linear IRT run (`0.312561`). The MLP should be
treated as a richer Stage 1 overfit/pseudo-labeling model rather than evidence
that the item-ID-only nonlinear model generalizes better.

## Training Pipeline

The K-factor script:

1. Loads `subject_id`, `item_id`, and `label` from `data/joined.parquet`.
2. Keeps only labels in `{0, 1}`.
3. Encodes subjects and items into dense integer IDs.
4. Creates a deterministic row-level validation split.
5. Initializes:
   - `subject_bias = Embedding(n_subjects, 1, sparse=True)`
   - `subject_u = Embedding(n_subjects, K, sparse=True)`
   - `item_v = Embedding(n_items, K, sparse=True)`
   - `item_z = Embedding(n_items, 1, sparse=True)`
6. Initializes `subject_bias` and `item_z` from smoothed marginal correctness
   rates. The factor embeddings start from small random normal values.
7. Optimizes binary cross entropy with logits using `SparseAdam`, plus a small
   batch-local L2 penalty when `--weight-decay` is positive.
8. Optionally linearly warms the learning rate from
   `--warmup-start-factor * --lr` to `--lr` over `--warmup-epochs`.
9. Applies `ReduceLROnPlateau` to the validation loss when validation is enabled,
   otherwise to the training loss. The scheduler starts after warmup.

```text
logit_ij = subject_bias[i] + subject_u[i] dot item_v[j] + item_z[j]
```

10. Writes CSV parameter tables, a JSON summary, and a PyTorch checkpoint.

## Next Step

This Stage 1 fit should feed Stage 2: train a content model that maps item text
and metadata to fitted `(V_j, Z_j)` so the system can predict parameters for
new hidden items.
