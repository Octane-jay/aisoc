#!/usr/bin/env bash
# 第 0 步：数据准备 + 数据可复现性校验
#   1) 解压随仓库分发的官方 train.jsonl.gz（若 raw/train.jsonl 不存在）
#   2) 规范化成 SEVENLLM-Dataset-processed（字段 id/category/instruction/input/output）
#   3) 用 seed=42、每类 1250 条重新生成 24 类均衡 30k 子集，并与仓库内冻结版本逐字节比对
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

# 原始数据解压到 OUT_ROOT，不污染仓库工作区
RAW_WORK="$OUT_ROOT/raw/SEVENLLM-Dataset"
mkdir -p "$RAW_WORK"
if [ -f "$RAW_DIR/train.jsonl" ]; then
  cp -n "$RAW_DIR/train.jsonl" "$RAW_WORK/train.jsonl"
elif [ -f "$RAW_WORK/train.jsonl" ]; then
  log "复用已解压的 $RAW_WORK/train.jsonl"
else
  log "解压 $RAW_DIR/train.jsonl.gz → $RAW_WORK/train.jsonl"
  gunzip -ck "$RAW_DIR/train.jsonl.gz" > "$RAW_WORK/train.jsonl"
fi
cp -n "$RAW_DIR/test.jsonl" "$RAW_WORK/test.jsonl" 2>/dev/null || true
for file in "$RAW_WORK/train.jsonl" "$RAW_WORK/test.jsonl"; do
  [ -f "$file" ] || { log "缺少原始数据：$file（见 REPRODUCE.md 第 3 节）"; exit 1; }
done

sha256sum "$RAW_WORK/train.jsonl" "$RAW_WORK/test.jsonl"

log "1/3 规范化字段 → $OUT_ROOT/SEVENLLM-Dataset-processed"
"$PYTHON" "$REPO_ROOT/code/prepare_dataset.py" \
  --source-dir "$RAW_WORK" \
  --output-dir "$OUT_ROOT/SEVENLLM-Dataset-processed"

log "2/3 重建 24 类均衡 30k 子集（seed=42，每类上限 1250）"
rm -rf "$OUT_ROOT/SEVENLLM-24class-balanced-30k-v1"
"$PYTHON" "$REPO_ROOT/code/prepare_sevenllm_balanced.py" \
  --source-dir "$RAW_WORK" \
  --output-dir "$OUT_ROOT/SEVENLLM-24class-balanced-30k-v1" \
  --model "$MODEL_PATH" \
  --seed 42 \
  --per-category 1250 \
  --max-length 1280

log "3/3 与仓库内冻结数据比对 sha256"
"$PYTHON" - "$DATA_DIR" "$OUT_ROOT/SEVENLLM-24class-balanced-30k-v1" <<'PY'
import hashlib, sys
from pathlib import Path

frozen, regen = Path(sys.argv[1]), Path(sys.argv[2])
ok = True
for name in ("train.jsonl", "test.jsonl"):
    a = hashlib.sha256((frozen / name).read_bytes()).hexdigest()
    b = hashlib.sha256((regen / name).read_bytes()).hexdigest()
    status = "一致" if a == b else "不一致！"
    ok &= a == b
    print(f"{name}: 冻结 {a[:16]} / 重建 {b[:16]} → {status}")
raise SystemExit(0 if ok else 1)
PY
log "数据准备完成，可直接进入 01_train_qlora.sh"
