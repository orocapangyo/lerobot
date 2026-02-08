#!/bin/bash
# X-VLA Training Script Template for H100
# Usage: ./scripts/train_xvla_h100.sh YOUR_HF_ID/dataset_name

DATASET_REPO_ID=${1:-"YOUR_HF_ID/bimanual_towel_fold"}
OUTPUT_DIR="outputs/train/xvla_h100"

echo "🏋️ Starting X-VLA Training on H100..."
echo "📊 Dataset: $DATASET_REPO_ID"

# Activate environment if not already active
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Training Command
# Optimized for H100 with bfloat16 and high performance settings
lerobot-train \
    --dataset.repo_id=$DATASET_REPO_ID \
    --policy.type=xvla \
    --policy.path="lerobot/xvla-base" \
    --policy.dtype=bfloat16 \
    --policy.action_mode=auto \
    --policy.train_soft_prompts=true \
    --policy.train_policy_transformer=true \
    --policy.freeze_vision_encoder=false \
    --policy.freeze_language_encoder=false \
    --batch_size=32 \
    --steps=30000 \
    --eval_freq=5000 \
    --save_freq=5000 \
    --output_dir=$OUTPUT_DIR \
    --job_name="xvla_h100_training"

echo "✅ Training session completed/interrupted."
echo "📂 Checkpoints saved in: $OUTPUT_DIR"
