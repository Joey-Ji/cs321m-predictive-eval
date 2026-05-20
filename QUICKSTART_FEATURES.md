# 🚀 Quick Start: Feature Engineering Branch

**Current Branch:** `feature/stage2-advanced-features`
**Status:** ✅ Code Complete, Ready for Testing
**Commit:** `1ff0d6d`

## What's Done

✅ Created `src/advanced_features.py` - 15 new statistical features
✅ Created `scripts/train_kfactor_head_advanced.py` - Enhanced training script
✅ Created unit tests
✅ Committed to branch
✅ Comprehensive documentation in [FEATURE_ENGINEERING_README.md](FEATURE_ENGINEERING_README.md)

## What You Need to Do Next

### Option A: Test Locally (Recommended First)

```bash
# 1. Download data (if you haven't already)
uv run python scripts/download_data.py --out data

# 2. Check if Stage 1 outputs exist
ls data/stage1/kfactor_k4/item_targets.pt

# 3. Check if embeddings exist
ls data/embeddings/mpnet_v1/item_embeddings.npy

# 4. Train with new features (takes ~10 minutes)
uv run python scripts/train_kfactor_head_advanced.py \
  --joined data/joined.parquet \
  --stage1 data/stage1/kfactor_k4 \
  --emb data/embeddings/mpnet_v1 \
  --out data/stage2/kfactor_mpnet_advanced_test \
  --head-type mlp \
  --hidden 256 \
  --epochs 30 \
  --lr 0.001

# 5. Check outputs
ls data/stage2/kfactor_mpnet_advanced_test/
# Should see:
#   - head.pt (model weights)
#   - head_meta.json (config)
#   - advanced_feature_extractor.json (feature stats)
#   - metrics.json (validation results)
```

### Option B: Compare with Baseline

```bash
# Train baseline (no advanced features)
uv run python scripts/train_kfactor_head.py \
  --stage1 data/stage1/kfactor_k4 \
  --emb data/embeddings/mpnet_v1 \
  --out data/stage2/kfactor_mpnet_baseline \
  --head-type mlp \
  --hidden 256 \
  --epochs 30 \
  --lr 0.001

# Compare metrics.json from both outputs
cat data/stage2/kfactor_mpnet_baseline/metrics.json
cat data/stage2/kfactor_mpnet_advanced_test/metrics.json

# Look for:
# - target_val_mse (lower is better)
# - target_val_r2_per_dim (higher is better)
```

### Option C: Full Evaluation (If training works)

```bash
# Run split-faithful proxy evaluation
uv run python scripts/eval_split_faithful.py \
  --joined data/joined.parquet \
  --stage1 data/stage1/kfactor_k4 \
  --stage2 data/stage2/kfactor_mpnet_advanced_test \
  --emb data/embeddings/mpnet_v1 \
  --seeds 0,1,2 \
  --max-rows 1500 \
  --per-category 300

# Goal: MLL should improve by ≥0.01 compared to baseline
```

## If You Don't Have Data Yet

### Step 1: Download Data (~5-10 minutes)
```bash
uv run python scripts/download_data.py --out data
```

### Step 2: Check What Stage 1/Embeddings Exist
```bash
ls -R data/stage1/
ls -R data/embeddings/
```

If missing, you'll need to:
- Run Stage 1 training (see `stage_1/k_factor_irt/README.md`)
- Generate embeddings (see `scripts/encode_items.py`)

Or ask your teammate Joey for their data directories.

## Expected Output

When training completes successfully, you should see:

```
Loading training data...
  Loaded 4,443,797 training rows
Building advanced feature extractor...
  Feature dimension: 15
  Feature names: bench_avg_correct, bench_item_count, ...
Loading Stage 1 targets...
Loading embeddings and side features...
Aligning items...
  Aligned 70,873 items

Feature dimensions:
  Embedding:         768
  Side features:     18
  Advanced features: 15
  Total input:       801
  Output (k+1):      5

Dataset split:
  Train: 63,785 items
  Val:   7,088 items

Training mlp head for 30 epochs on cuda...
  epoch    1/30 train_mse_std=0.xxxxx val_mse_std=0.xxxxx
  ...
  epoch   30/30 train_mse_std=0.xxxxx val_mse_std=0.xxxxx

Validation metrics:
  MSE (standardized): 0.xxxxx
  MSE (original):     0.xxxxx
  R² per dimension:   ['0.xxx', '0.xxx', ...]

Done! Outputs saved to data/stage2/kfactor_mpnet_advanced_test
```

## Success Criteria

### ✅ Minimal Success
- Script runs without errors
- Outputs created in target directory
- Validation MSE is finite and reasonable

### ✅ Good Success
- Validation MSE is 2-5% lower than baseline
- R² values are ≥0.3 for most dimensions

### ✅ Excellent Success
- Split-faithful MLL improves by ≥0.01
- Ready to integrate into submission

## Troubleshooting

### Error: "ModuleNotFoundError"
**Fix:** Use `uv run python` instead of `python`

### Error: "No such file or directory: data/joined.parquet"
**Fix:** Run `uv run python scripts/download_data.py --out data`

### Error: "No such file or directory: data/stage1/kfactor_k4"
**Fix:** You need Stage 1 outputs. Ask teammate or run Stage 1 training.

### Error: "CUDA out of memory"
**Fix:** Add `CUDA_VISIBLE_DEVICES="" ` to force CPU training (slower but works)

### Training is very slow
**Fix:** Reduce `--epochs 10` for quick testing

## Next Actions After Testing

### If Features Help (MLL improves)
1. Create PR to merge into main
2. Integrate into submission (update `model.py`)
3. Test on Modal evaluation
4. Submit to Codabench

### If Features Don't Help Significantly
1. Try ablation studies (remove some feature groups)
2. Add feature normalization
3. Try different feature combinations
4. Document findings for team

### If You Want to Iterate
1. Modify `src/advanced_features.py` to add new features
2. Rerun training
3. Compare metrics
4. Keep what works

## Team Coordination

### Before Merging to Main
- [ ] Share results with team (MLL comparison)
- [ ] Get code review from Joey or teammate
- [ ] Verify doesn't break existing submissions
- [ ] Update main README if this becomes default

### Sharing Your Work
```bash
# Push branch to remote
git push origin feature/stage2-advanced-features

# Create GitHub PR
gh pr create --title "Feature Engineering: Add 15 statistical features to Stage 2" \
  --body "See FEATURE_ENGINEERING_README.md for details"
```

## Questions?

1. Read [FEATURE_ENGINEERING_README.md](FEATURE_ENGINEERING_README.md) for full details
2. Check unit tests: `uv run python tests/test_advanced_features.py`
3. Ask teammate or create GitHub issue

---

**Remember:** This is an experiment! Even if it doesn't improve MLL significantly, you've learned about:
- Feature engineering for neural models
- Stage 2 K-factor pipeline
- Evaluation methodology
- And you have clean, documented code to build on 🎉
