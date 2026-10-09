#!/usr/bin/env bash
# 生成 FILE_HASHES.txt：仓库内所有非 .git 文件的 sha256，换机 clone 后可校验完整性。
# 用法：bash scripts/99_hash_repo.sh [输出文件]
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

OUTPUT="${1:-$REPO_ROOT/FILE_HASHES.txt}"
cd "$REPO_ROOT"
find . -type f -not -path './.git/*' -not -name 'FILE_HASHES.txt' -not -name '*.pyc' \
  -not -path './code/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum > "$OUTPUT"
log "已写出 $OUTPUT（$(wc -l < "$OUTPUT") 个文件）"
