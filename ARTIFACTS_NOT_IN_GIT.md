# 没有进库的东西（以及怎么再拿到）

判据：只要下面每一项都能"重新下载或重新跑出来"，这个仓库对本实验就是完整的。

| 类别 | 路径 / 内容 | 体积 | 为什么没进库 | 怎么补回来 |
| --- | --- | ---: | --- | --- |
| 基座模型 | `Foundation-Sec-8B-Instruct`（4 个 safetensors 分片） | 16.0 GB | 超过 GitHub 单文件 100 MB 硬限 | `huggingface-cli download fdtn-ai/Foundation-Sec-8B-Instruct --local-dir $MODELS_ROOT/Foundation-Sec-8B-Instruct`；指纹见 `REPRODUCE.md` §3.1 |
| 句向量模型 | `paraphrase-multilingual-MiniLM-L12-v2` | ~470 MB | 同上 | `huggingface-cli download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir $SEMANTIC_MODEL_CACHE/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2` |
| 中间数据 | `SEVENLLM-Dataset-processed/{train,test}.jsonl` | 189 MB + 2.8 MB | 纯中间产物，可重建 | `bash scripts/00_prepare_data.sh`（第 1 步） |
| 官方原始训练集（未压缩） | `SEVENLLM-Dataset/train.jsonl` | 187 MB | 单文件超限；已改为 `data/raw/SEVENLLM-Dataset/train.jsonl.gz`（40 MB）入库 | `gunzip -k` 即可，脚本已自动处理 |
| 训练中间 checkpoint | `checkpoint-250/`、`checkpoint-468/`（含 `optimizer.pt` 43 MB/个、`rng_state.pth`） | ~250 MB | 只用于断点续训；最终 adapter 已在 `adapter/` | 重新训练会再生成；`checkpoint-250` 的 `trainer_state.json` 数值与最终一致 |
| 失败/早期训练日志 | `outputs/foundation-sec-qlora-core-30k/run.log` 等 | 41 KB | 与结论无关，但记录了两次 `max_length` 失败（exit_code=143/1） | 见 `results/logs/qlora-train.run.log` 里保留的完整重试记录 |
| 分布外测试集 | `Primus-Instruct/{alert_explanation,cmd_analysis}.parquet`（各 100 条） | 0.5 MB | 本次正式报告未使用，仅 `plan.md` 列为后续计划 | `trendmicro-ailab/Primus-Instruct`（gated，需 HF token） |
| conda 环境 | `/data/hyc/env/aisoc` | 数 GB | 环境不复用，按 `requirements.txt` 重建即可 | `pip install -r requirements.txt` |

## 与本次实验无关、但仍然留在原机器上的东西

- `/data/hyc/poc/data/SEVENLLM-24class-balanced-30k-v1/`：与仓库内 `data/` 同名目录，内容一致（冻结副本）。
- `/data/hyc/poc/checkpoints/foundation-sec-qlora-24class-balanced-30k-v1/`：完整训练输出（含 optimizer 状态）。
- `/data/hyc/poc/outputs/baseline/`：基线推理与打分的原始输出，`results/baseline/` 是其副本。

## 冻结哈希

仓库内每个文件的 sha256 见 `FILE_HASHES.txt`（生成：`bash scripts/99_hash_repo.sh`）。
关键数据文件的 sha256 同时写进了 `data/SEVENLLM-24class-balanced-30k-v1/manifest.json`，
`scripts/10_verify.sh` 会自动比对。
