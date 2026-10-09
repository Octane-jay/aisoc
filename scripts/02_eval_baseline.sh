#!/usr/bin/env bash
# 第 2 步：零样本基线推理（Foundation-Sec-8B-Instruct，无 adapter）
#         贪心解码 do_sample=False，max_new_tokens=512，同一 Alpaca 提示词模板。
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

OUTPUT_DIR="${OUTPUT_DIR:-$OUT_ROOT/results/baseline}"
mkdir -p "$OUTPUT_DIR"
log "零样本基线推理：model=$MODEL_PATH"

"$PYTHON" "$REPO_ROOT/code/evaluate_baseline.py" \
  --model "$MODEL_PATH" \
  --data "$DATA_DIR/test.jsonl" \
  --output "$OUTPUT_DIR/predictions.jsonl" \
  --max-new-tokens 512 2>&1 | tee "$OUTPUT_DIR/evaluate.log"

log "基线预测完成 → $OUTPUT_DIR/predictions.jsonl（1,300 条）"
