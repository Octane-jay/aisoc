#!/usr/bin/env bash
# 一键复现：数据 → 训练 → 基线推理 → 微调推理 → 打分 → 报告表 → 校验
# 耗量参考：数据准备 ~1 分钟；训练约 2.5 小时（单卡 4090）；两次推理各约 40 分钟；打分约 10 分钟。
# 建议在 tmux 里跑：tmux new -s poc 'bash scripts/run_all.sh 2>&1 | tee run_all.log'
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

bash 00_prepare_data.sh
bash 01_train_qlora.sh
bash 02_eval_baseline.sh
ADAPTER_DIR="${ADAPTER_DIR:-$OUT_ROOT/checkpoints/foundation-sec-qlora-24class-balanced-30k-v1}" bash 03_eval_qlora.sh
bash 04_score.sh
bash 05_report_tables.sh
bash 10_verify.sh
