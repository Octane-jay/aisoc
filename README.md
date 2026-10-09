# Foundation-Sec-8B-Instruct × SEVENLLM：QLoRA 安全垂类微调与复现包

网络安全垂类模型的 QLoRA 微调实验，以及**换服务器后可 100% 复现**的全部代码、数据与结果。

- 基座：`Foundation-Sec-8B-Instruct`（8B，Llama-3.1 架构）
- 训练数据：SEVENLLM-Dataset 官方 `train.jsonl` 中训练/测试共有的 24 个类别，按类均衡抽样 **30,000 条**（每类 1,250）
- 测试集：SEVENLLM 官方 `test.jsonl` **全量 1,300 条**（结构化抽取 600、开放生成 700）
- 微调：QLoRA（NF4 4-bit，LoRA `r=8`/`alpha=16`/`dropout=0.05`，覆盖注意力与 MLP 全部投影层）
- 随机种子：**42**；1 epoch；有效 batch 64；学习率 1e-5；cosine + 3% warmup；`max_length=1280`；共 468 steps
- 硬件与耗时：单卡 RTX 4090 48 GB，8,418.8 秒完成，`train_loss=1.0930`

## 结果（同一 1,300 条官方测试集，贪心解码）

| 指标 | 零样本基线 | QLoRA 微调后 | 变化 |
| --- | ---: | ---: | ---: |
| Coarse Semantic F1（τ=0.80，主） | 0.4418 | 0.5071 | +0.0653 |
| Coarse Semantic F1（τ=0.75） | 0.4686 | 0.5325 | +0.0639 |
| Loose Leaf F1（辅助） | 0.3367 | 0.3995 | +0.0628 |
| Exact Leaf F1（辅助） | 0.2247 | 0.2989 | +0.0742 |
| 结构化 JSON 合法率 | 96.50% | 97.83% | +1.33pp |
| 生成 Rouge-L（jieba/whitespace） | 0.2971 | 0.3887 | +0.0916 |
| 生成语义相似度 | 0.8514 | 0.8928 | +0.0414 |

分场景（告警研判 / 攻击链 / 日志漏洞）构成与指标见
[`results/report/SCENARIO_TABLES.md`](results/report/SCENARIO_TABLES.md)，
正式报告见 [`results/report/评测报告_本次实验_正式版.docx`](results/report/评测报告_本次实验_正式版.docx)。

> 结论仅限 SEVENLLM 官方测试分布。SEVENLLM 是 CTI 安全文本指令数据，
> 不等价于原始主机日志、网络流量或生产 SIEM 告警，不能直接外推为生产安全研判能力提升。

## 仓库结构

```
code/       训练、推理、打分、场景统计脚本（参数化，无硬编码路径）
scripts/    分步入口：00 数据 → 01 训练 → 02/03 推理 → 04 打分 → 05 报告表 → 10 校验
configs/    机器可读的实验参数（超参、种子、版本、指纹）
data/       冻结的 30k 训练集 + 官方测试集 + 选样清单；raw/ 为官方原始数据
adapter/    本次实验最终的 QLoRA adapter（可直接推理，无需重训）
results/    基线与微调后的逐样本预测、全部指标文件、日志、正式报告
REPRODUCE.md  换服务器复现手册（环境、下载、每步命令、期望结果、已知坑）
```

## 快速开始

```bash
export WORK_ROOT=/data/xxx           # 数据/模型/环境父目录
export MODELS_ROOT=$WORK_ROOT/models
export ENV_ROOT=$WORK_ROOT/env/aisoc

bash scripts/00_prepare_data.sh      # 数据校验（sha256 与冻结清单比对）
ADAPTER_DIR=$PWD/adapter bash scripts/03_eval_qlora.sh   # 用自带 adapter 直接推理
bash scripts/02_eval_baseline.sh
bash scripts/04_score.sh
bash scripts/05_report_tables.sh
bash scripts/10_verify.sh            # 与冻结结果逐项比对
```

完整流程（含重新训练）`bash scripts/run_all.sh`，细节与坑见 [REPRODUCE.md](REPRODUCE.md)。

## 数据与权重

| 内容 | 是否随库分发 |
| --- | --- |
| 30k 训练集、官方测试集、选样 manifest、场景统计 | 是 |
| QLoRA adapter（84 MB）+ tokenizer | 是 |
| 逐样本预测与全部指标文件 | 是 |
| SEVENLLM 原始 `train.jsonl`（91,401 条） | 压缩随库分发（`data/raw/`，40 MB） |
| 基座模型权重（16 GB）、句向量模型 | 否，按 `REPRODUCE.md` 第 3 节在新服务器上下载 |

数据来源：Hugging Face `Multilingual-Multimodal-NLP/SEVENLLM-Dataset`、
`fdtn-ai/Foundation-Sec-8B-Instruct`、`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`。
