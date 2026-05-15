# K=4 Full-Train Latent Export

This directory contains the Stage 1 plain K-factor IRT latents from the full
training-response fit:

```text
K=4 train_log_loss=0.2919 train_accuracy=86.50%
```

Files:

```text
subject_capabilities.parquet  subject_id, subject_bias, u_0, u_1, u_2, u_3
item_parameters.parquet       item_id, v_0, v_1, v_2, v_3, z
manifest.json                 run metadata and source output directory
```

Load example:

```python
import pandas as pd

subjects = pd.read_parquet(
    "stage_1/k_factor_irt/artifacts/k4_full_train/subject_capabilities.parquet"
)
items = pd.read_parquet(
    "stage_1/k_factor_irt/artifacts/k4_full_train/item_parameters.parquet"
)
```

Prediction form:

```text
logit_ij = subject_bias_i + U_i dot V_j + z_j
P(y_ij = 1) = sigmoid(logit_ij)
```
