# Stage 1: K-Factor IRT

This directory contains the Stage 1 response-matrix model used to estimate:

- subject/model capability vectors `U_i`
- item loading vectors `V_j`
- item difficulty/bias terms `Z_j`

The fitted probability model is:

```text
P(y_ij = 1) = sigmoid(U_i dot V_j + Z_j)
```

For the first pass we set `K=4`, so each subject has a 4-dimensional
capability vector and each item has a 4-dimensional loading vector.

## What This Stage Does

This is a transductive fit over public training responses. It estimates latent
parameters for subjects and training items that appear in the public response
matrix. It does not yet solve item cold-start for hidden leaderboard items.

The intended use is:

1. Fit `U`, `V`, and `Z` on the public binary response matrix.
2. Use fitted subject vectors `U` as known subject/model capability estimates.
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

## Quick K=4 Replication Run

From the repository root:

```bash
uv run python stage_1/k_factor_irt/fit_k_factor_irt.py \
  --joined data/joined.parquet \
  --k 4 \
  --epochs 3 \
  --batch-size 65536 \
  --lr 0.05 \
  --val-frac 0.02 \
  --out stage_1/k_factor_irt/outputs/k4_quick
```

The output directory is ignored by git. It will contain:

```text
subject_capabilities.csv  # subject_id, u_0, u_1, u_2, u_3
item_parameters.csv       # item_id, v_0, v_1, v_2, v_3, z
fit_summary.json          # config, counts, train/validation loss history
model_state.pt            # PyTorch state dict and id vocabularies
```

## Metrics From The First Quick Run

The initial run used a deterministic row-level validation holdout. We used
row-level validation here because item-cold-start validation would hold out item
vectors and bias terms that this Stage 1 model is explicitly trying to fit.

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

Validation log loss is the average negative log probability assigned to the
true held-out label. Lower is better. Mean log likelihood is the same value with
the sign flipped, so higher is better and it is usually negative.

The factor dimensions are not individually identifiable: signs and rotations can
change without changing predictions. Interpret `U_i dot V_j + Z_j`, not an
individual raw coordinate as a stable semantic axis.

## Training Pipeline

The script:

1. Loads `subject_id`, `item_id`, and `label` from `data/joined.parquet`.
2. Keeps only labels in `{0, 1}`.
3. Encodes subjects and items into dense integer IDs.
4. Creates a deterministic row-level validation split.
5. Initializes:
   - `subject_u = Embedding(n_subjects, K)`
   - `item_v = Embedding(n_items, K)`
   - `item_z = Embedding(n_items, 1)`
6. Optimizes binary cross entropy with logits:

```text
logit_ij = subject_u[i] dot item_v[j] + item_z[j]
```

7. Writes CSV parameter tables, a JSON summary, and a PyTorch checkpoint.

## Next Step

This Stage 1 fit should feed Stage 2: train a content model that maps item text
and metadata to fitted `(V_j, Z_j)` so the system can predict parameters for
new hidden items.
