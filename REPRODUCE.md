# 换服务器复现手册（Foundation-Sec-8B-Instruct × SEVENLLM QLoRA）

目标：**换一台机器后，照这份文档走一遍，就能重跑出 `results/` 里冻结的全部数字。**
所有路径都相对仓库或环境变量，不写死任何机器绝对路径。

对应实验报告：`results/report/评测报告_本次实验_正式版.docx`

---

## 0. 一分钟版

```bash
# 1) 取代码与数据
git clone <REPO_URL> poc && cd poc

# 2) 设好路径（只有这一步需要按新机器改）
export WORK_ROOT=/data/xxx            # 数据、模型、环境的父目录
export MODELS_ROOT=$WORK_ROOT/models  # 基座模型 + 句向量缓存
export ENV_ROOT=$WORK_ROOT/env/aisoc  # conda 环境（也可直接 export PYTHON=...）

# 3) 装环境 → 下模型（见第 2、3 节）

# 4) 只想验证评测链路（不用重训，用仓库里的 adapter，约 1 小时）
bash scripts/00_prepare_data.sh        # 数据校验，~1 分钟
bash scripts/02_eval_baseline.sh
ADAPTER_DIR=$PWD/adapter bash scripts/03_eval_qlora.sh
bash scripts/04_score.sh
bash scripts/05_report_tables.sh
bash scripts/10_verify.sh              # 与冻结结果逐项比对

# 5) 要完整复现训练（含训练，约 4 小时）
bash scripts/run_all.sh
```

重新训练时建议挂 tmux：`tmux new -s poc 'bash scripts/run_all.sh 2>&1 | tee run_all.log'`。

---

## 1. 仓库内容速查

| 目录 | 内容 | 是否必需 |
| --- | --- | --- |
| `code/` | 训练、推理、打分、场景统计的全部脚本 | 必需 |
| `scripts/` | 按顺序执行的入口脚本 + 环境变量约定 | 必需 |
| `data/SEVENLLM-24class-balanced-30k-v1/` | **冻结的训练集 30,000 条 + 官方测试集 1,300 条 + 选择清单 manifest** | 必需（不用重新选样） |
| `data/raw/SEVENLLM-Dataset/` | 官方原始 `train.jsonl.gz`（91,401 条）与 `test.jsonl`（1,300 条） | 用于重建选样，可不下 |
| `adapter/` | 本次实验最终 QLoRA adapter（84 MB）+ tokenizer，可直接推理 | 验证/演示用，重训可跳过 |
| `results/baseline/`、`results/finetuned/` | 两次推理的**逐样本预测**与全部分数文件（`*.jsonl` + `*.metrics.json`） | 必需（对照基准） |
| `results/report/` | 正式评测报告 docx + 三大场景统计表 | 必需 |
| `results/logs/` | 训练日志、评测日志、`training_args.bin`、`trainer_state.json` | 证据留档 |

**没有进库、也不必进库的**：基座模型权重（16 GB）、conda 环境、`SEVENLLM-Dataset-processed/`（中间产物，可由 `00_prepare_data.sh` 重建）、早期失败的训练日志。

---

## 2. 环境

本次实验实际使用的版本（`/data/hyc/env/aisoc`，Python 3.10.21）：

| 包 | 版本 | 用途 |
| --- | --- | --- |
| torch | 2.4.1+cu121 | 训练/推理 |
| transformers | 4.46.3 | 模型加载 |
| peft | 0.13.2 | LoRA / QLoRA |
| accelerate | 1.0.1 | Trainer |
| bitsandbytes | 0.50.2 | 4-bit NF4 量化 |
| datasets | 5.0.1 | 数据集封装 |
| sentence-transformers | 3.3.1 | 语义相似度 / Coarse Semantic F1 |
| tokenizers | 0.20.3 · huggingface-hub 0.36.2 · safetensors 0.8.0 | 推理依赖 |
| jieba | 0.42.1 | Rouge-L 中文分词 |
| scipy | 1.15.3 · numpy 2.2.6 | Hungarian 匹配 |

```bash
conda create -p "$ENV_ROOT" python=3.10 -y
conda activate "$ENV_ROOT"
pip install -r requirements.txt          # 见仓库根目录；国内可加 -i https://pypi.tuna.tsinghua.edu.cn/simple
python -c "import torch, peft, transformers; print(torch.cuda.is_available())"   # 必须为 True
```

> 版本冲突是这套实验最容易踩的坑：`transformers 5.x` 会让 `peft` 导入直接报
> `cannot import name 'Cache'`。务必锁到上表版本。

打分链路（`04_score.sh`）要在同一环境里跑，因为 `score_*.py` 复用了
`evaluate_baseline.py` 里的 JSON 解析与题型判定函数（`code/metrics_common.py`
只依赖标准库，`report_scenarios.py` 因此可以在没装 torch 的机器上算统计表）。

---

## 3. 模型与数据

### 3.1 基座模型（约 16 GB，不进库）

```bash
huggingface-cli download fdtn-ai/Foundation-Sec-8B-Instruct --local-dir "$MODELS_ROOT/Foundation-Sec-8B-Instruct"
```

本地指纹（用于确认拿到的是同一份权重）：

```
文件数 4 个分片，合计 16,081,516,129 bytes
config.json                    sha256 09b92d21ad46c9812b000ec94b0bb39b0fe7c180afc576799b012b0d4c62f73e
model.safetensors.index.json   sha256 c8d2635fd74bfa99d68ab5ae8305df7f68980772ddbd36fe8640cbf2f18bc37a
generation_config.json         sha256 70b5284d5524b4bf8a87a78b9a0b03a332c5730a5efffc904a531cd5d2ff9797
```

### 3.2 句向量模型（打分用，约 470 MB）

```bash
huggingface-cli download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --local-dir "$MODELS_ROOT/sentence-transformers/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
```

离线机器上只要保证 `SEMANTIC_MODEL_CACHE` 指向这个目录（默认
`$MODELS_ROOT/sentence-transformers`），并 `export HF_HUB_OFFLINE=1` 即可。

### 3.3 数据

| 文件 | 条数 | sha256 | 说明 |
| --- | ---: | --- | --- |
| 官方 `train.jsonl` | 91,401 | `804ab62cc47c609ff2c93b4ff26ca2ef0fcdcc05133f538ca90a18eeb1b380ae` | 源：HF `Multilingual-Multimodal-NLP/SEVENLLM-Dataset` |
| 官方 `test.jsonl` | 1,300 | `08a27277386b1f205ed0814d379bb0db9bdce8f39bde099020a1f8044d404d7b` | 同上 |
| 冻结训练集 `data/SEVENLLM-24class-balanced-30k-v1/train.jsonl` | 30,000 | `fe5305b85a0e68be26ebb16b5523fb6e986aff099589c77e251fe2268b569b2b` | 本实验实际训练数据 |
| 冻结测试集 `.../test.jsonl` | 1,300 | `1238da4e5c8699d9f504886818f6db9699630603973bfa662f27aedde0c58df8` | 本实验实际评测数据 |
| `data/raw/SEVENLLM-Dataset/train.jsonl.gz` | 91,401 | gz `a7214a4a9fcb7d6110a10c17e4ce66742e09c0cfd745fbaf224e25f7b40f04d5` | 已随库分发，省掉一次 187 MB 下载 |
| `data/raw/SEVENLLM-Dataset/test.jsonl` | 1,300 | `08a27277...`（同官方） | 已随库分发 |

重新训练/重新评测**只需要** `data/SEVENLLM-24class-balanced-30k-v1/` 两个文件，
它们在库里，无需下载。原始 91,401 条只用于验证选样过程（`00_prepare_data.sh` 第 3 步会
用 `seed=42`、每类上限 1,250 重新选一次，并与冻结文件逐字节比对 sha256）。

---

## 4. 全流程复现

每一步的期望产物、耗时量级如下（顺序不可颠倒；`OUT_ROOT` 默认 `$WORK_ROOT/poc-runs`）。

| 步骤 | 命令 | 产物 | 参考耗时 |
| --- | --- | --- | --- |
| 0 数据 | `bash scripts/00_prepare_data.sh` | `$OUT_ROOT/SEVENLLM-Dataset-processed/`、`$OUT_ROOT/SEVENLLM-24class-balanced-30k-v1/`，并打印 sha256 比对结果 | ~1 分钟 |
| 1 训练 | `bash scripts/01_train_qlora.sh` | `$OUT_ROOT/checkpoints/foundation-sec-qlora-24class-balanced-30k-v1/`（adapter + `run_config.json` + `run.log`） | 单卡 4090 约 2.5 小时 |
| 2 基线推理 | `bash scripts/02_eval_baseline.sh` | `$OUT_ROOT/results/baseline/predictions.jsonl`（1,300 条） | ~40 分钟 |
| 3 微调推理 | `ADAPTER_DIR=... bash scripts/03_eval_qlora.sh` | `$OUT_ROOT/results/finetuned/predictions.jsonl` | ~40 分钟 |
| 4 打分 | `bash scripts/04_score.sh` | 四个口径的 `*.jsonl` + `*.metrics.json` | ~10 分钟（GPU） |
| 5 报告表 | `bash scripts/05_report_tables.sh` | `$OUT_ROOT/report/SCENARIO_TABLES.md` | 秒级 |
| 6 校验 | `bash scripts/10_verify.sh` | 与 `results/` 冻结值逐项比对的差异表 | 秒级 |

不做训练、只想验证评测与指标链路时，把第 3 步的 `ADAPTER_DIR` 指向仓库自带的
`adapter/`，跳过第 1 步即可。

### 单步命令（不想用脚本时）

```bash
PY=${ENV_ROOT}/bin/python
D=data/SEVENLLM-24class-balanced-30k-v1

# 训练
$PY code/train_qlora.py --model "$MODEL_PATH" --train $D/train.jsonl \
  --output "$OUT_ROOT/checkpoints/run1" --max-train-samples 30000 --max-length 1280 \
  --per-device-train-batch-size 4 --gradient-accumulation-steps 16 --seed 42 --gradient-checkpointing

# 推理（基线 / 微调后）
$PY code/evaluate_baseline.py --model "$MODEL_PATH" --data $D/test.jsonl \
  --output "$OUT_ROOT/results/baseline/predictions.jsonl" --max-new-tokens 512
$PY code/evaluate_baseline.py --model "$MODEL_PATH" --adapter adapter --data $D/test.jsonl \
  --output "$OUT_ROOT/results/finetuned/predictions.jsonl" --max-new-tokens 512

# 打分
$PY code/rescore_baseline.py --data $D/test.jsonl --predictions <pred>.jsonl --output <dir>/rescored.jsonl
$PY code/score_sevenllm.py     --data $D/test.jsonl --predictions <pred>.jsonl --output <dir>/paper_metrics.jsonl
$PY code/score_coarse_embedding.py --data $D/test.jsonl --predictions <pred>.jsonl --output <dir>/coarse_embedding.jsonl
$PY code/score_structured_variants.py --data $D/test.jsonl --predictions <pred>.jsonl --output <dir>/variants.jsonl
```

---

## 5. 训练参数与随机种子（速查）

| 项 | 值 |
| --- | --- |
| 基座 | `Foundation-Sec-8B-Instruct`（Llama-3.1 架构，8B） |
| 微调方式 | QLoRA：NF4 4-bit + double quant，compute dtype `bfloat16` |
| LoRA | `r=8`、`lora_alpha=16`、`lora_dropout=0.05`、`bias=none`、`task_type=CAUSAL_LM` |
| LoRA 目标层 | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`（注意力 + MLP 全部投影） |
| **随机种子** | **`seed=42`**（TrainingArguments 的 `seed`，同时用于按类分层抽样的 `random.Random`） |
| 抽样种子细节 | 每类用 `random.Random(f"42:{category}")` 洗牌后取前 1,250 条；整体再用 `random.Random(42)` 洗牌 |
| 训练数据 | `data/SEVENLLM-24class-balanced-30k-v1/train.jsonl`，30,000 条，24 类 × 1,250 条 |
| epoch | 1 |
| per-device batch | 4 |
| 梯度累积 | 16（有效 batch = 64） |
| 训练步数 | 468 = ⌈30,000 / 64⌉ |
| 学习率 | 1e-5，`cosine` 调度，`warmup_ratio=0.03` |
| max_length | 1280（prompt + answer；超长 prompt 从中间截断保留首尾，答案超长直接报错） |
| 其它 | `bf16=True`、`tf32=True`、`paged_adamw_8bit`、`gradient_checkpointing=True`、`group_by_length=True`、`logging_steps=1`、`save_steps=250`、`save_total_limit=2` |
| 优化器细节 | `weight_decay=0.0`、`max_grad_norm=1.0`（均取 Trainer 默认值，未特意设置） |
| 硬件 / 耗时 | 单卡 RTX 4090 48 GB；468 steps，8,418.8 秒；最终 `train_loss=1.0930`（exit_code=0） |

> 上表每一项都已用实跑的 `results/logs/training_args.bin` 反查确认（`seed` 字段就是 42），
> 不是照抄脚本默认值。想自己核对：
> `python -c "import torch; a=torch.load('results/logs/training_args.bin', weights_only=False); print(a.seed, a.learning_rate)"`

推理（基线与微调后完全一致）：

| 项 | 值 |
| --- | --- |
| 提示词模板 | `### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n` |
| 解码 | 贪心 `do_sample=False`，`max_new_tokens=512`，`pad_token_id=eos` |
| 权重 | 基线直接加载基座；微调后用 `PeftModel.from_pretrained` 挂 adapter，不合并权重 |
| 数据 | `data/SEVENLLM-24class-balanced-30k-v1/test.jsonl` 全量 1,300 条（ex 600 / gen 700） |

机器可读版本见 `configs/experiment.json`；每次训练的实跑参数还会落到
输出目录的 `run_config.json`（含 torch / transformers / python 版本）。

---

## 6. 评测口径

| 指标 | 定义 | 代码 |
| --- | --- | --- |
| Coarse Semantic F1（主） | JSON 递归拍平只留叶子值、忽略字段名；NFKC+小写+去空白归一后先做精确/子串匹配，未命中用多语言句向量余弦≥τ，Hungarian 一对一匹配；逐样本 P/R/F1 后 Macro 平均 | `score_coarse_embedding.py` |
| Loose Leaf F1（辅） | 叶子值一对一贪心：相等或"归一化参考值是预测值子串"即 TP | `rescore_baseline.py` / `metrics_common.greedy_scores` |
| Exact Leaf F1（辅） | 同上但要求完全相等 | 同上 |
| JSON 合法率 | 预测能否解析成 JSON 对象/数组（含 ```json 代码块） | `score_sevenllm.py` |
| Rouge-L | 中文用 jieba 分词、其余按空白切分，token 级 LCS F1 | `score_sevenllm.py` |
| 语义相似度 | `paraphrase-multilingual-MiniLM-L12-v2` 句向量 L2 归一化后余弦 | `score_sevenllm.py` |
| 字段别名辅助口径 | 字段别名规范化 + embedding 的 strict / normalized-soft F1 | `score_structured_variants.py` |

题型判定（`metrics_common.task_type`）是全链路唯一来源：`answer_type == structured` → `ex`；
指令含选择题标记 → `mcq`；指令含 information extraction / JSON 标记或参考输出可解析为 JSON → `ex`；否则 `gen`。
**改动它会同时改变打分与报告统计，不要随手改。**

---

## 7. 期望结果

冻结结果（`results/*/**.metrics.json`），重跑后应逐项落在 1e-6 内
（同一硬件上一般完全一致；换卡会有 ~1e-3 量级的浮点抖动，不影响四位小数）。

> 实测：`rescore` / `coarse_embedding` / `variants` 三个口径在本机用 CPU 重打分时**逐字节一致**；
> `paper_metrics` 只有 `semantic_similarity` 一类指标出现 ~1e-7 的浮动（句向量在 CPU/GPU 上的浮点差异），
> 不影响四位小数。打分默认走 GPU（`SCORE_DEVICE=cuda`，可用 `SCORE_DEVICE=cpu` 覆盖）。

总表（同一 1,300 条官方测试集）：

| 指标 | 零样本基线 | QLoRA 微调后 |
| --- | ---: | ---: |
| Coarse Semantic F1（τ=0.80） | 0.4418 | 0.5071 |
| Coarse Semantic F1（τ=0.75） | 0.4686 | 0.5325 |
| Coarse Semantic F1（τ=0.70） | 0.4969 | 0.5546 |
| Coarse Semantic F1（τ=0.65） | 0.5247 | 0.5771 |
| Loose Leaf F1 | 0.3367 | 0.3995 |
| Exact Leaf F1 | 0.2247 | 0.2989 |
| JSON 合法率 | 96.50% | 97.83% |
| 生成 Rouge-L（jieba/whitespace） | 0.2971 | 0.3887 |
| 生成语义相似度 | 0.8514 | 0.8928 |

三大场景（告警研判 / 攻击链 / 日志漏洞）的构成与分场景指标：
`results/report/SCENARIO_TABLES.md`，可用 `bash scripts/05_report_tables.sh` 重算。

> 训练集构成表里"告警研判 ex"比报告正文多 1 条（2,464 vs 2,463）：
> 报告正文是当初临时脚本的产物，`report_scenarios.py` 用的是与打分链路一致的
> `task_type` 口径，测试集 6 个数字与报告完全一致。以脚本输出为准。

---

## 8. 已知坑

1. **`transformers` 版本**：5.x 与 `peft 0.13.2` 不兼容（`cannot import name 'Cache'`）。锁 4.46.3。
2. **`max_length` 必须 ≥ 1280**：训练集最长答案 1,255 token，用 1024 会直接抛
   `ValueError: Answer exceeds max_length=1024`（历史失败记录见 `results/logs/qlora-train.run.log`）。
3. **`group_by_length=True`**：训练顺序按长度分桶，不是随机顺序；步数固定 468。换 seed 会同时改变
   抽样集合与分桶顺序。
4. **测试集泄漏已审计**：`duplicate_audit.json` 记录了 17 条与测试集 `input` 完全重复的训练样本，
   选样时已剔除；96 条重复任务、6,040 条"仅训练集类别"也被排除。**不要绕过 `prepare_sevenllm_balanced.py` 直接取原始 train.jsonl 训练**。
5. **打分顺序**：`rescore → score_sevenllm → coarse_embedding → variants`，四者都按 `id` 对齐测试集。
6. **`adapter_config.json`** 里的 `base_model_name_or_path` 保留的是训练时的绝对路径，
   加载时以 `--model` 参数为准（`PeftModel.from_pretrained(model, adapter)`），换机不受影响。
7. **场景统计是"重叠候选集"，不是互斥切分**：风险评估、威胁分析等类别同时服务多个场景，
   三行不可相加（并集 1,228 / 1,300，其余 72 条只有单场景归属）。
8. **流量纪律**：换机后模型/数据一律在新服务器上下载，不要从本地 `scp` 上传；
   本机只有 NATFRP 隧道，大文件过隧道会吃掉全组额度。

---

## 9. 全库指纹

`FILE_HASHES.txt` 记录仓库内每个文件的 sha256（生成：`bash scripts/99_hash_repo.sh`）。
换机 clone 后可先跑一次，确认没有传输损坏。
