#!/usr/bin/env bash
# 第 1 步：QLoRA 训练（与本次实验完全相同的超参：seed=42 / 1 epoch / bs=4 / grad_accum=16 /
#         lr=1e-5 / cosine / warmup=3% / max_length=1280 / group_by_length）
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUT_ROOT/checkpoints/foundation-sec-qlora-24class-balanced-30k-v1}"
MAX_LENGTH="${MAX_LENGTH:-1280}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
SEED="${SEED:-42}"
RUN_LOG="$OUTPUT_DIR/run.log"

mkdir -p "$OUTPUT_DIR"
log "QLoRA 训练启动：GPU=$CUDA_VISIBLE_DEVICES model=$MODEL_PATH"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv | tee -a "$RUN_LOG"

"$PYTHON" "$REPO_ROOT/code/train_qlora.py" \
  --model "$MODEL_PATH" \
  --train "$TRAIN_FILE" \
  --output "$OUTPUT_DIR" \
  --max-train-samples 30000 \
  --max-length "$MAX_LENGTH" \
  --per-device-train-batch-size "$TRAIN_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --seed "$SEED" \
  --gradient-checkpointing 2>&1 | tee -a "$RUN_LOG"

log "训练结束，adapter 与 run_config.json 位于 $OUTPUT_DIR"
log "参考量级：468 steps，单卡 RTX 4090 约 8,419 秒，train_loss=1.0930"
