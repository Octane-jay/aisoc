# data/ 说明

本目录只保留**训练与评测真正需要**的文件，避免把 400 MB 中间产物带进仓库。

| 路径 | 内容 |
| --- | --- |
| `SEVENLLM-24class-balanced-30k-v1/train.jsonl` | **实际训练数据**，30,000 条，24 类 × 1,250 条 |
| `SEVENLLM-24class-balanced-30k-v1/test.jsonl` | **实际评测数据**，SEVENLLM 官方测试集全量 1,300 条（规范化后的字段） |
| `SEVENLLM-24class-balanced-30k-v1/manifest.json` | 选样清单：24 个共同类别、每类可用/抽中条数、**每个类别被抽中的样本 id 列表**、源文件 sha256、长度分布 |
| `SEVENLLM-24class-balanced-30k-v1/duplicate_audit.json` | 泄漏与去重审计：17 条与测试集 input 重复的训练样本（已剔除）、96 条重复任务、6,040 条训练集独有类别 |
| `SEVENLLM-24class-balanced-30k-v1/length_report.json` | token 长度分布（prompt/output/combined 的 p50~p99 与超长条数） |
| `raw/SEVENLLM-Dataset/train.jsonl.gz` | 官方原始训练集（91,401 条）压缩版，仅用于复现"选样"这一步 |
| `raw/SEVENLLM-Dataset/test.jsonl` | 官方原始测试集（1,300 条，与 HF 下载件逐字节一致） |
| `dataset_catalog.md` | SEVENLLM 28 个类别按业务分组的目录（日志分析 / 漏洞 / 告警 / 攻击链） |
| `selection_candidates.md` | 三大业务场景（告警研判 / 攻击链 / 日志漏洞）的类别映射与候选测试子集统计 |

## 字段

规范化后每条记录为 `{"id", "category", "instruction", "input", "output", "answer_type"}`：

- 相对官方原始数据，**丢掉了 `thought` 字段**（思维链不参与训练，避免把中间推理当监督信号）；
- `answer_type` 取 `structured`（`output` 是 JSON 对象/数组）或 `text`；
- 训练输入模板：`### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n`。

## 重新生成

```bash
bash scripts/00_prepare_data.sh     # 会重建 processed 与 30k 子集，并与本目录的冻结文件比对 sha256
```

`SEVENLLM-Dataset-processed/`（189 MB 的中间产物）不进仓库，由上面的脚本按需生成。
