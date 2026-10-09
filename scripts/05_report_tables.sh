#!/usr/bin/env bash
# 第 5 步：重算评测报告里的三大场景统计表（训练/测试构成 + 分场景指标）
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

"$PYTHON" "$REPO_ROOT/code/report_scenarios.py" \
  --data-dir "$DATA_DIR" \
  --baseline-dir "${BASELINE_DIR:-$OUT_ROOT/results/baseline}" \
  --finetuned-dir "${FINETUNED_DIR:-$OUT_ROOT/results/finetuned}" \
  --output "$OUT_ROOT/report/SCENARIO_TABLES.md" \
  --json-output "$OUT_ROOT/report/scenario_tables.json"
log "场景统计表 → $OUT_ROOT/report/SCENARIO_TABLES.md"
