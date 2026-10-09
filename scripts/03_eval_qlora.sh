#!/usr/bin/env bash
# 第 3 步：微调后推理（基座 + QLoRA adapter，adapter 不合并权重，推理时挂载）
#   默认用仓库内冻结的 adapter/；若刚训练完，用 ADAPTER_DIR=$OUT_ROOT/checkpoints/<run> 覆盖。
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

OUTPUT_DIR="${OUTPUT_DIR:-$OUT_ROOT/results/finetuned}"
mkdir -p "$OUTPUT_DIR"
log "微调后推理：model=$MODEL_PATH adapter=$ADAPTER_DIR"

"$PYTHON" "$REPO_ROOT/code/evaluate_baseline.py" \
  --model "$MODEL_PATH" \
  --adapter "$ADAPTER_DIR" \
  --data "$DATA_DIR/test.jsonl" \
  --output "$OUTPUT_DIR/predictions.jsonl" \
  --max-new-tokens 512 2>&1 | tee "$OUTPUT_DIR/evaluate.log"

log "微调后预测完成 → $OUTPUT_DIR/predictions.jsonl（1,300 条）"
