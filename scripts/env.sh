#!/usr/bin/env bash
# 公共环境：换服务器时只需要在外部 export 下面几个变量，或改这一处默认值。
# 用法：source scripts/env.sh
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT

# 目录约定（全部相对可迁移，仓库里不写死任何机器绝对路径）
export WORK_ROOT="${WORK_ROOT:-$(dirname "$REPO_ROOT")}"           # 数据/模型/环境的父目录
export MODELS_ROOT="${MODELS_ROOT:-$WORK_ROOT/models}"              # 基座模型与句向量缓存
export ENV_ROOT="${ENV_ROOT:-$WORK_ROOT/env/aisoc}"                 # conda 环境（可选）
export MODEL_PATH="${MODEL_PATH:-$MODELS_ROOT/Foundation-Sec-8B-Instruct}"
export SEMANTIC_MODEL_CACHE="${SEMANTIC_MODEL_CACHE:-$MODELS_ROOT/sentence-transformers}"
export OUT_ROOT="${OUT_ROOT:-$WORK_ROOT/poc-runs}"                  # 重跑产物根目录

# 运行时
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ -z "${PYTHON:-}" ]; then
  if [ -x "$ENV_ROOT/bin/python" ]; then PYTHON="$ENV_ROOT/bin/python"; else PYTHON="python3"; fi
fi
export PYTHON

log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"; }
export -f log

export DATA_DIR="$REPO_ROOT/data/SEVENLLM-24class-balanced-30k-v1"
export RAW_DIR="$REPO_ROOT/data/raw/SEVENLLM-Dataset"
export ADAPTER_DIR="${ADAPTER_DIR:-$REPO_ROOT/adapter}"
