"""重算评测报告里的“三大业务场景”统计表（训练/测试构成 + 分场景指标）。

场景由 category 语义映射得到（见 data/selection_candidates.md），
三个场景之间存在重叠类别，因此三行不做合计。

用法（默认路径相对仓库根目录，可全部用参数覆盖）：
    python code/report_scenarios.py
    python code/report_scenarios.py --data-dir data/SEVENLLM-24class-balanced-30k-v1 \
        --baseline-dir results/baseline --finetuned-dir results/finetuned \
        --output results/report/SCENARIO_TABLES.md
"""

import argparse
import json
from pathlib import Path

from metrics_common import task_type
from paths import repo_root


# 与 data/selection_candidates.md 一致；类别语义重叠是设计如此，不互斥。
SCENARIOS = {
    "告警研判": [
        "Security Alert Generation", "Cybersecurity Event Classification", "Threat Analysis",
        "Risk Assessment", "Incident Response Planning", "Protection Strategy Research",
        "Correlation Analysis", "Important Event Extraction", "Impact Scope",
    ],
    "攻击链": [
        "Attack Strategy Analysis", "Attack Means Analysis", "Attack Target Intelligence Gathering",
        "Attack Tool Identification", "Attacker Information Extraction", "Malware Feature Extraction",
        "Impact Scope", "Threat Analysis", "Main Relation Extraction", "Correlation Analysis",
    ],
    "日志漏洞": [
        "Vulnerability Information Extraction", "Vulnerability Exploitation Analysis", "Security Policy Audit",
        "Key Entity Recognition", "Main Relation Extraction", "Summary Generation",
        "Risk Assessment", "Protection Strategy Research",
    ],
}


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def fmt(value, digits=4, percent=False):
    if value is None:
        return "—"
    return f"{value * 100:.2f}%" if percent else f"{value:.{digits}f}"


def scenario_of(category):
    return [name for name, categories in SCENARIOS.items() if category in categories]


def composition(records, label):
    """按场景统计 ex / gen（gen 含 mcq）样本数。task_type 与评测链路完全一致。"""
    kinds = {record["id"]: task_type(record) for record in records}
    rows, covered = [], set()
    for name, categories in SCENARIOS.items():
        members = [record for record in records if record["category"] in categories]
        covered.update(record["id"] for record in members)
        ex = sum(kinds[record["id"]] == "ex" for record in members)
        rows.append((name, ex, len(members) - ex))
    summary = {
        "label": label,
        "total": len(records),
        "ex_total": sum(kind == "ex" for kind in kinds.values()),
        "gen_total": sum(kind != "ex" for kind in kinds.values()),
        "rows": rows,
        "scenario_union": len(covered),
    }
    return summary


def scenario_metrics(records, paper_rows, coarse_rows):
    by_id = {record["id"]: record for record in records}
    kind_of = {record["id"]: task_type(record) for record in records}
    paper = {row["id"]: row for row in paper_rows}
    coarse = {row["id"]: row for row in coarse_rows}
    result = {}
    for name, categories in SCENARIOS.items():
        ex_ids = [i for i, record in by_id.items() if record["category"] in categories and kind_of[i] == "ex"]
        gen_ids = [i for i, record in by_id.items() if record["category"] in categories and kind_of[i] == "gen"]
        result[name] = {
            "ex_samples": len(ex_ids),
            "gen_samples": len(gen_ids),
            "coarse_semantic_f1_tau0.80": mean([coarse[i]["scores"]["0.8"] for i in ex_ids if i in coarse]),
            "coarse_semantic_f1_tau0.75": mean([coarse[i]["scores"]["0.75"] for i in ex_ids if i in coarse]),
            "loose_leaf_f1": mean([paper[i].get("loose_f1") for i in ex_ids if i in paper]),
            "exact_leaf_f1": mean([paper[i].get("exact_f1") for i in ex_ids if i in paper]),
            "json_valid_rate": mean([paper[i].get("json_valid") for i in ex_ids if i in paper]),
            "rouge_l_jieba_whitespace": mean([paper[i].get("rouge_l_jieba_whitespace") for i in gen_ids if i in paper]),
            "semantic_similarity": mean([paper[i].get("semantic_similarity") for i in gen_ids if i in paper]),
        }
    return result


def markdown(train_summary, test_summary, baseline, finetuned):
    lines = ["# 三大业务场景统计（由 `code/report_scenarios.py` 重算）", ""]
    lines += ["场景由 `category` 语义映射得到，场景之间共享类别、彼此重叠，三行不做合计。", ""]
    for summary in (train_summary, test_summary):
        lines += [f"## {summary['label']}（共 {summary['total']} 条：ex {summary['ex_total']}、gen {summary['gen_total']}）", ""]
        lines += ["| 场景 | 信息抽取（ex） | 生成（gen） |", "| --- | ---: | ---: |"]
        lines += [f"| {name} | {ex:,} | {gen:,} |" for name, ex, gen in summary["rows"]]
        lines += ["", f"三场景并集覆盖 {summary['scenario_union']} 条（场景重叠，不合计）。", ""]

    lines += ["## 分场景指标：信息抽取（ex）", "", "| 场景 / 指标 | 零样本基线 | QLoRA 微调后 | 变化 |", "| --- | ---: | ---: | ---: |"]
    for metric, label, percent in [
        ("coarse_semantic_f1_tau0.80", "Coarse Semantic F1（τ=0.80）", False),
        ("coarse_semantic_f1_tau0.75", "Coarse Semantic F1（τ=0.75）", False),
        ("loose_leaf_f1", "Loose Leaf F1", False),
        ("exact_leaf_f1", "Exact Leaf F1", False),
        ("json_valid_rate", "JSON 合法率", True),
    ]:
        for name in SCENARIOS:
            old, new = baseline[name][metric], finetuned[name][metric]
            delta = "—" if old is None or new is None else (f"+{(new - old) * 100:.2f}pp" if percent else f"+{new - old:.4f}")
            lines.append(f"| {name}｜{label} | {fmt(old, percent=percent)} | {fmt(new, percent=percent)} | {delta} |")

    lines += ["", "## 分场景指标：开放式生成（gen）", "", "| 场景 / 指标 | 零样本基线 | QLoRA 微调后 | 变化 |", "| --- | ---: | ---: | ---: |"]
    for metric, label in [("rouge_l_jieba_whitespace", "Rouge-L（jieba/whitespace）"), ("semantic_similarity", "语义相似度")]:
        for name in SCENARIOS:
            old, new = baseline[name][metric], finetuned[name][metric]
            delta = "—" if old is None or new is None else f"+{new - old:.4f}"
            lines.append(f"| {name}｜{label} | {fmt(old)} | {fmt(new)} | {delta} |")
    lines.append("")
    return "\n".join(lines)


def main():
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=root / "data/SEVENLLM-24class-balanced-30k-v1")
    parser.add_argument("--baseline-dir", type=Path, default=root / "results/baseline")
    parser.add_argument("--finetuned-dir", type=Path, default=root / "results/finetuned")
    parser.add_argument("--output", type=Path, default=root / "results/report/SCENARIO_TABLES.md")
    parser.add_argument("--json-output", type=Path, default=root / "results/report/scenario_tables.json")
    args = parser.parse_args()

    train_records = read_jsonl(args.data_dir / "train.jsonl")
    test_records = read_jsonl(args.data_dir / "test.jsonl")
    baseline = scenario_metrics(test_records, read_jsonl(args.baseline_dir / "paper_metrics.jsonl"), read_jsonl(args.baseline_dir / "coarse_embedding.jsonl"))
    finetuned = scenario_metrics(test_records, read_jsonl(args.finetuned_dir / "paper_metrics.jsonl"), read_jsonl(args.finetuned_dir / "coarse_embedding.jsonl"))
    train_summary = composition(train_records, "训练集构成（SEVENLLM 24 类均衡 30,000 条）")
    test_summary = composition(test_records, "测试集构成（SEVENLLM 官方 test.jsonl 全量）")

    report = {
        "generated_by": "code/report_scenarios.py",
        "scenario_categories": SCENARIOS,
        "train_composition": train_summary,
        "test_composition": test_summary,
        "baseline_metrics": baseline,
        "finetuned_metrics": finetuned,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown(train_summary, test_summary, baseline, finetuned))
    args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"train": train_summary["rows"], "test": test_summary["rows"]}, ensure_ascii=False, indent=2))
    print(f"wrote {args.output}\nwrote {args.json_output}")


if __name__ == "__main__":
    main()
