#!/usr/bin/env bash
# 第 4 步：统一口径打分（基线与微调后使用同一解析、同一分词、同一模型、同一测试集）
#   rescore        宽松/严格叶子值 F1（旧报告口径）
#   score_sevenllm Rouge-L(jieba/whitespace) + 语义相似度 + 分组指标
#   coarse_embedding Coarse Semantic F1（报告主指标，τ=0.65/0.70/0.75/0.80）
#   variants       字段别名规范化 + embedding 的辅助口径
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

SCORE_DEVICE="${SCORE_DEVICE:-cuda}"
SEMANTIC_MODEL="${SEMANTIC_MODEL:-sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2}"

score_dir() {
  local dir="$1"
  [ -f "$dir/predictions.jsonl" ] || { log "跳过（无 predictions）：$dir"; return 0; }
  log "打分 $dir（device=$SCORE_DEVICE）"
  "$PYTHON" "$REPO_ROOT/code/rescore_baseline.py" \
    --data "$DATA_DIR/test.jsonl" --predictions "$dir/predictions.jsonl" --output "$dir/rescored.jsonl"
  "$PYTHON" "$REPO_ROOT/code/score_sevenllm.py" \
    --data "$DATA_DIR/test.jsonl" --predictions "$dir/predictions.jsonl" \
    --output "$dir/paper_metrics.jsonl" --semantic-model "$SEMANTIC_MODEL" \
    --cache-dir "$SEMANTIC_MODEL_CACHE" --device "$SCORE_DEVICE"
  "$PYTHON" "$REPO_ROOT/code/score_coarse_embedding.py" \
    --data "$DATA_DIR/test.jsonl" --predictions "$dir/predictions.jsonl" \
    --output "$dir/coarse_embedding.jsonl" --semantic-model "$SEMANTIC_MODEL" \
    --cache-dir "$SEMANTIC_MODEL_CACHE" --device "$SCORE_DEVICE"
  "$PYTHON" "$REPO_ROOT/code/score_structured_variants.py" \
    --data "$DATA_DIR/test.jsonl" --predictions "$dir/predictions.jsonl" \
    --output "$dir/variants.jsonl" --semantic-model "$SEMANTIC_MODEL" \
    --cache-dir "$SEMANTIC_MODEL_CACHE" --device "$SCORE_DEVICE"
}

score_dir "${BASELINE_DIR:-$OUT_ROOT/results/baseline}"
score_dir "${FINETUNED_DIR:-$OUT_ROOT/results/finetuned}"
log "打分完成"
