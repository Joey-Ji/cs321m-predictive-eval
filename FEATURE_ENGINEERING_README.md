# Advanced Feature Engineering for Stage 2

**Branch:** `feature/stage2-advanced-features`
**Author:** Ruijia Guan
**Date:** May 20, 2025

## Overview

This branch adds **advanced statistical and metadata features** to the Stage 2 K-factor head training pipeline. Previously, Stage 2 only used:
- Text embeddings (mpnet/bge encoder)
- One-hot benchmark/condition encodings

Now we also include **15 additional features** computed from training data:

### New Features (15 dimensions)

#### Benchmark-level (5 features)
1. `bench_avg_correct`: Average correctness rate for this benchmark
2. `bench_item_count`: Number of training items
3. `bench_avg_length`: Average item content length
4. `bench_std_length`: Standard deviation of lengths
5. `bench_label_entropy`: Entropy of label distribution (difficulty variation)

#### Condition-level (3 features)
6. `cond_avg_correct`: Average correctness for this condition
7. `cond_item_count`: Number of items with this condition
8. `cond_benchmark_count`: Number of benchmarks using this condition

#### Interaction (2 features)
9. `interaction_avg_correct`: Correctness for (benchmark, condition) pair
10. `interaction_item_count`: Item count for this specific combination

#### Stage 1 Global (4 features)
11. `stage1_z_mean`: Global mean of item difficulty from Stage 1
12. `stage1_z_std`: Global std of item difficulty
13. `stage1_v_norm_mean`: Mean L2 norm of item loading vectors
14. `stage1_v_norm_std`: Std of loading vector norms

#### Item-level (1 feature)
15. `item_length`: Character count of this item's content

## Files Added/Modified

### New Files
- `src/advanced_features.py` - Feature extraction module
- `scripts/train_kfactor_head_advanced.py` - Enhanced training script
- `tests/test_advanced_features.py` - Unit tests
- `FEATURE_ENGINEERING_README.md` - This file

### Architecture

```
Training Data (joined.parquet)
       |
       v
AdvancedFeatureExtractor
       |
       +---> compute_benchmark_statistics()
       +---> compute_condition_statistics()
       +---> compute_interaction_statistics()
       +---> compute_stage1_parameter_statistics()
       |
       v
Feature Vector [15 dims]
       |
       +---> Concatenate with:
             - Embedding [768 dims]
             - Side features [~20 dims]
       v
Total Input [~803 dims] --> MLP Head --> (V, z) predictions
```

## Usage

### Step 1: Download Data (if not already done)

```bash
uv run python scripts/download_data.py --out data
```

This creates:
- `data/joined.parquet` (training rows)
- `data/items.parquet`, `data/subjects.parquet`, `data/benchmarks.parquet`

### Step 2: Prepare Stage 1 Outputs

Make sure you have Stage 1 outputs available:
```
data/stage1/kfactor_k4/
  - item_targets.pt
  - item_to_id.json
```

### Step 3: Prepare Embeddings

Make sure you have item embeddings:
```
data/embeddings/mpnet_v1/
  - item_embeddings.npy
  - item_id_order.json
  - item_side_features.npy
  - side_feature_meta.json
  - encoder_meta.json
```

### Step 4: Train with Advanced Features

```bash
uv run python scripts/train_kfactor_head_advanced.py \
  --joined data/joined.parquet \
  --stage1 data/stage1/kfactor_k4 \
  --emb data/embeddings/mpnet_v1 \
  --out data/stage2/kfactor_mpnet_advanced_v1 \
  --head-type mlp \
  --hidden 256 \
  --epochs 30 \
  --lr 0.001 \
  --val-frac 0.1 \
  --seed 0
```

### Step 5: Compare with Baseline

Train the baseline (without advanced features) for comparison:

```bash
uv run python scripts/train_kfactor_head.py \
  --stage1 data/stage1/kfactor_k4 \
  --emb data/embeddings/mpnet_v1 \
  --out data/stage2/kfactor_mpnet_baseline_v1 \
  --head-type mlp \
  --hidden 256 \
  --epochs 30 \
  --lr 0.001
```

### Step 6: Evaluate Both Models

Use the split-faithful evaluation to compare MLL scores:

```bash
# Baseline
python scripts/eval_split_faithful.py \
  --joined data/joined.parquet \
  --stage1 data/stage1/kfactor_k4 \
  --stage2 data/stage2/kfactor_mpnet_baseline_v1 \
  --emb data/embeddings/mpnet_v1 \
  --seeds 0,1,2 \
  --max-rows 1500 \
  --per-category 300

# Advanced features
python scripts/eval_split_faithful.py \
  --joined data/joined.parquet \
  --stage1 data/stage1/kfactor_k4 \
  --stage2 data/stage2/kfactor_mpnet_advanced_v1 \
  --emb data/embeddings/mpnet_v1 \
  --seeds 0,1,2 \
  --max-rows 1500 \
  --per-category 300
```

## Expected Improvements

### Hypothesis
Adding statistical features should help the model because:

1. **Benchmark difficulty**: Knowing that "GPQA" has 30% correct rate vs "MMLU" with 70% helps calibrate predictions
2. **Condition effects**: "zero-shot" vs "few-shot" can have consistent difficulty deltas
3. **Interaction patterns**: Some benchmarks are more sensitive to conditions than others
4. **Item complexity**: Longer/more complex items tend to be harder

### Success Criteria
- ✅ **Minimal**: Code runs without errors, produces valid predictions
- ✅ **Good**: Validation MSE decreases by 2-5% compared to baseline
- ✅ **Excellent**: MLL improves by ≥0.01 on split-faithful proxy evaluation

## Implementation Details

### Feature Extraction
All features are computed from **training data only** to avoid leakage:
- Statistics aggregated over training rows in `joined.parquet`
- Stage 1 parameters from fitted `item_targets.pt`
- For unseen benchmark/condition at test time, use **global mean fallback**

### Normalization Strategy
Features are **not normalized** before concatenation because:
- The MLP head can learn appropriate scaling
- Some features (counts) have clear absolute meaning
- Avoids needing to save/load normalization parameters

If performance is poor, consider adding feature normalization.

### Graceful Degradation
The `AdvancedFeatureExtractor` handles edge cases:
- Missing benchmark → use global average statistics
- Unseen condition → use global average statistics
- Empty statistics → use neutral defaults (0.5 for rates, 0 for counts)

## Testing

Run unit tests (requires numpy):
```bash
uv run python tests/test_advanced_features.py
```

Tests cover:
- Benchmark/condition/interaction statistics computation
- Feature extraction for known and unseen values
- Serialization (save/load) of feature extractor

## Integration with Existing Pipeline

### Compatibility
- ✅ Works with existing `encode_items.py` output
- ✅ Compatible with both `mpnet` and `bge` encoders
- ✅ Can be disabled by using original `train_kfactor_head.py`

### Submission Integration
To use in a submission:
1. Train with `train_kfactor_head_advanced.py`
2. Copy `advanced_feature_extractor.json` into submission ZIP
3. Modify `submissions/v1_kfactor/model.py` to:
   - Load the feature extractor
   - Extract features for each prediction input
   - Concatenate with embeddings before head inference

This requires updating the submission `predict()` function but the model weights remain the same.

## Next Steps

### Immediate
- [ ] Download data if not available
- [ ] Run training with advanced features
- [ ] Compare validation metrics with baseline
- [ ] If improvement ≥2%, integrate into submission

### Future Enhancements
- [ ] Add **temporal features** (release date → model capability proxy)
- [ ] Add **subject metadata features** (provider, family, params)
- [ ] Try **learned feature embeddings** instead of one-hot for benchmark/condition
- [ ] Experiment with **feature interactions** (benchmark × condition learned embeddings)
- [ ] Add **curriculum learning** (train on easy benchmarks first)

### Ablation Studies
To understand which features help most:
1. Train with all 15 features (baseline)
2. Train with only benchmark features (5 dims)
3. Train with only condition features (3 dims)
4. Train with only interaction features (2 dims)
5. Compare validation metrics

## Troubleshooting

### Issue: Training is slow
**Solution:** Use smaller `--hidden` dimension (128 instead of 256) or reduce `--epochs`

### Issue: Validation loss increases
**Potential causes:**
- Overfitting: reduce `--epochs` or increase dropout in `build_head()`
- Learning rate too high: try `--lr 0.0005`
- Features not helping: check feature statistics in saved JSON

### Issue: Features seem wrong
**Debug steps:**
1. Check `advanced_feature_extractor.json` after training
2. Verify `benchmark_stats` and `condition_stats` look reasonable
3. Run unit tests to validate extraction logic

### Issue: Cannot load data
**Check:**
- `data/joined.parquet` exists and is not corrupted
- Stage 1 outputs exist at specified path
- Embeddings directory has all required files

## Questions?

Contact Ruijia Guan or open an issue on the team repo.

## References

- Original Stage 2 pipeline: `scripts/train_kfactor_head.py`
- K-factor model: `src/kfactor.py`
- Feature encoding: `src/features.py`
