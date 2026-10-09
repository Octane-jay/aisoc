# SEVENLLM 场景数据目录

数据源：`SEVENLLM-Dataset/train.jsonl`（约 9.14 万条）和 `test.jsonl`（1,300 条）。数据形态是 CTI 安全文本上的指令任务，不是原始主机日志、网络流量包或 SIEM 告警流。

## 推荐训练类别

### 日志分析与事件理解

`Cybersecurity Event Classification`、`Important Event Extraction`、`Key Entity Recognition`、`Main Relation Extraction`、`Time Element Acquisition`、`Summary Generation`、`Correlation Analysis`

### 漏洞检测与影响分析

`Vulnerability Information Extraction`、`Vulnerability Exploitation Analysis`、`Malware Feature Extraction`、`Impact Scope`、`Risk Assessment`

### 告警研判与响应

`Threat Analysis`、`Security Alert Generation`、`Incident Response Planning`、`Protection Strategy Research`

### 攻击链与攻击行为

`Attack Strategy Analysis`、`Attack Means Analysis`、`Attack Target Intelligence Gathering`、`Attack Tool Identification`、`Attacker Information Extraction`、`Attack Intent Analysis`、`Behavioral Pattern Analysis`、`Domain Intelligence Acquisition`、`Network Protocol Utilization`

## 可选扩展类别

`Encryption-Decryption Algorithm Identification`、`Security Policy Audit`、`Trend Prediction`。这些类别对安全能力有帮助，但与首轮日志、告警、漏洞、攻击链目标较间接，建议作为扩展实验。

## 推荐实验拆分

- 首轮训练：从 `train.jsonl` 按四个核心组筛选，保留类别分布，不使用 `test.jsonl`。
- 分布内测试：完整使用 `test.jsonl`，同时按四个核心组做分组统计。
- 消融实验：分别去掉日志、漏洞、告警或攻击链组，观察能力变化。
- 数据形态限制：真实日志分析还需另补充脱敏主机日志、告警 JSON 或网络流量数据；SEVENLLM 不能单独代表这些输入形态。

