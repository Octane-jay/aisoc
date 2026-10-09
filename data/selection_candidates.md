# 场景相关数据筛选候选

本清单依据 SEVENLLM 的 `category` 字段和任务语义整理，属于候选映射，正式训练前需要结合样本内容抽样复核。

## 测试建议

建议保留 SEVENLLM 官方 `test.jsonl` 的全部 `1,300` 条作为分布内主测试集，不为凑场景而删除样本；再单独报告下面的场景候选子集。

| 场景 | 候选类别数 | 当前测试样本数 |
| --- | ---: | ---: |
| 告警研判 | 9 | 691 |
| 日志漏洞 | 8 | 298 |
| 攻击链 | 10 | 726 |
| 三场景并集（去重） | 21 | 1,228 |

### 告警研判候选

`Security Alert Generation`、`Cybersecurity Event Classification`、`Threat Analysis`、`Risk Assessment`、`Incident Response Planning`、`Protection Strategy Research`、`Correlation Analysis`、`Important Event Extraction`、`Impact Scope`

### 日志漏洞候选

`Vulnerability Information Extraction`、`Vulnerability Exploitation Analysis`、`Security Policy Audit`、`Key Entity Recognition`、`Main Relation Extraction`、`Summary Generation`、`Risk Assessment`、`Protection Strategy Research`

### 攻击链候选

`Attack Strategy Analysis`、`Attack Means Analysis`、`Attack Target Intelligence Gathering`、`Attack Tool Identification`、`Attacker Information Extraction`、`Malware Feature Extraction`、`Impact Scope`、`Threat Analysis`、`Main Relation Extraction`、`Correlation Analysis`

## 分布外测试

- `Primus-Instruct/alert_explanation.parquet`：100 条结构化告警研判。
- `Primus-Instruct/cmd_analysis.parquet`：100 条可疑命令分析。

## 训练使用原则

- 不能把 SEVENLLM `test.jsonl` 样本混入训练。
- 训练子集应依据确认后的类别清单从 `train.jsonl` 筛选，并保存类别分布和总数。
- 如果目标是复现报告，保留 18 类 / `62,210` 条作为单独实验；如果目标是探索效果，另建场景扩展实验，不能混写结果。

