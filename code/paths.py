"""可移植路径解析。

原则：仓库里不出现任何机器绝对路径。所有默认值都从环境变量推导，
换服务器时只需要设置 MODELS_ROOT / ENV_ROOT / WORK_ROOT 之一。

环境变量约定（见 scripts/env.sh）：
  WORK_ROOT   项目父目录，默认取仓库目录的上一级
  MODELS_ROOT 基座模型与句向量模型缓存所在目录，默认 $WORK_ROOT/models
  ENV_ROOT    conda 环境目录，默认 $WORK_ROOT/env/aisoc
  SEMANTIC_MODEL_CACHE  sentence-transformers 缓存目录，默认 $MODELS_ROOT/sentence-transformers
"""

import os
from pathlib import Path


def repo_root():
    return Path(__file__).resolve().parent.parent


def work_root():
    return Path(os.environ.get("WORK_ROOT") or repo_root().parent)


def models_root():
    return Path(os.environ["MODELS_ROOT"]) if os.environ.get("MODELS_ROOT") else work_root() / "models"


def base_model_path():
    """基座模型目录：优先 MODEL_PATH，其次 MODELS_ROOT/Foundation-Sec-8B-Instruct。"""
    if os.environ.get("MODEL_PATH"):
        return Path(os.environ["MODEL_PATH"])
    return models_root() / "Foundation-Sec-8B-Instruct"


def semantic_cache_dir():
    """sentence-transformers 的缓存目录（离线机器上预置好该目录即可）。"""
    explicit = os.environ.get("SEMANTIC_MODEL_CACHE")
    if explicit:
        return Path(explicit)
    return models_root() / "sentence-transformers"
