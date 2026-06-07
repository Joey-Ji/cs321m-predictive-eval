#!/bin/bash
# 训练高级特征模型 - 完整流程
# 使用方法: bash RUN_TRAINING.sh

set -e  # 出错时停止

echo "🚀 开始训练流程..."
echo ""

# 步骤1: 下载数据（如果还没下载）
if [ ! -f "data/joined.parquet" ]; then
    echo "📥 步骤1/4: 下载训练数据..."
    python3.11 scripts/download_data.py --out data
    echo "✅ 数据下载完成"
else
    echo "✅ 步骤1/4: 训练数据已存在"
fi
echo ""

# 步骤2: 检查Stage 1输出
if [ ! -f "data/stage1/kfactor_k4/item_targets.pt" ]; then
    echo "⚠️  步骤2/4: Stage 1输出不存在"
    echo "选项A: 从队友那里复制 data/stage1/kfactor_k4/"
    echo "选项B: 运行Stage 1训练（~30-60分钟）:"
    echo "  python3.11 stage_1/k_factor_irt/fit_k_factor_irt.py \\"
    echo "    --joined data/joined.parquet --k 4 --epochs 8 \\"
    echo "    --batch-size 65536 --lr 0.05 --val-frac 0 \\"
    echo "    --out data/stage1/kfactor_k4"
    exit 1
else
    echo "✅ 步骤2/4: Stage 1输出已存在"
fi
echo ""

# 步骤3: 检查embeddings
if [ ! -f "data/embeddings/mpnet_v1/item_embeddings.npy" ]; then
    echo "⚠️  步骤3/4: Embeddings不存在"
    echo "选项A: 从队友那里复制 data/embeddings/mpnet_v1/"
    echo "选项B: 生成embeddings（~10-20分钟）:"
    echo "  python3.11 scripts/encode_items.py \\"
    echo "    --joined data/joined.parquet \\"
    echo "    --encoder sentence-transformers/all-mpnet-base-v2 \\"
    echo "    --out data/embeddings/mpnet_v1"
    exit 1
else
    echo "✅ 步骤3/4: Embeddings已存在"
fi
echo ""

# 步骤4: 训练高级特征模型
echo "🎯 步骤4/4: 训练高级特征模型..."
echo "这将需要约15-30分钟（CPU）或5-10分钟（GPU）"
echo ""

python3.11 scripts/train_kfactor_head_advanced.py \
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

echo ""
echo "🎉 训练完成！"
echo ""
echo "查看结果:"
echo "  cat data/stage2/kfactor_mpnet_advanced_v1/metrics.json"
echo ""
echo "对比baseline（可选）:"
echo "  bash RUN_BASELINE.sh"
