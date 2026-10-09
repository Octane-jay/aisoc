#!/usr/bin/env bash
# 第 6 步：一致性校验
#   1) 数据 sha256 与 manifest.json 记录一致
#   2) 重跑得到的指标与仓库内冻结结果（results/）逐项比对
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

"$PYTHON" "$REPO_ROOT/code/verify_metrics.py" \
  --reference "$REPO_ROOT/results" \
  --candidate "${CANDIDATE_ROOT:-$OUT_ROOT/results}" \
  --data "$DATA_DIR"
