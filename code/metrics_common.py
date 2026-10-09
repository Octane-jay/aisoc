"""评测纯函数：JSON 解析、叶子值匹配、Rouge-L、题型判定、提示词模板。

这里**只依赖标准库**，所以 `report_scenarios.py` 这类统计脚本不需要 torch /
peft / transformers 就能跑。`evaluate_baseline.py` 会把这些名字重新导出，
保证历史调用脚本的 import 路径不变。
"""

import ast
import json
import re


def normalize(value):
    value = str(value).lower().strip()
    value = re.sub(r"\s+", "", value)
    return value.translate(str.maketrans({
        "。": ".", "，": ",", "、": ",", "；": ";", "：": ":", "！": "!", "？": "?",
        "（": "(", "）": ")", "【": "[", "】": "]",
    }))


def leaves(value):
    if isinstance(value, dict):
        result = []
        for child in value.values():
            result.extend(leaves(child))
        return result
    if isinstance(value, list):
        result = []
        for child in value:
            result.extend(leaves(child))
        return result
    if value is None or value == "":
        return []
    return [str(value)]


def parse_json(text):
    candidates = re.findall(r"```(?:json)?\s*(.*?)```", text, re.S)
    candidates.append(text.strip())
    for candidate in candidates:
        candidate = candidate.strip()
        for parser in (json.loads, ast.literal_eval):
            try:
                value = parser(candidate)
                if isinstance(value, (dict, list)):
                    return value
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
    return None


def greedy_scores(reference, prediction):
    """叶子值一对一贪心匹配，返回 (loose_contains_f1, exact_leaf_f1)。"""
    reference = [normalize(value) for value in leaves(reference)]
    prediction = [normalize(value) for value in leaves(prediction)]
    used = set()
    loose = 0
    exact = 0
    for expected in reference:
        for index, actual in enumerate(prediction):
            if index in used:
                continue
            if expected == actual:
                used.add(index)
                loose += 1
                exact += 1
                break
            if expected and expected in actual:
                used.add(index)
                loose += 1
                break
    loose_precision = loose / len(prediction) if prediction else 0.0
    loose_recall = loose / len(reference) if reference else 0.0
    exact_precision = exact / len(prediction) if prediction else 0.0
    exact_recall = exact / len(reference) if reference else 0.0
    loose_f1 = 2 * loose_precision * loose_recall / (loose_precision + loose_recall) if loose_precision + loose_recall else 0.0
    exact_f1 = 2 * exact_precision * exact_recall / (exact_precision + exact_recall) if exact_precision + exact_recall else 0.0
    return loose_f1, exact_f1


def rouge_l(reference, prediction):
    """字符级 LCS F1（evaluate_baseline 的快速粗算）。正式报告用 jieba/whitespace 分词版。"""
    reference = list(reference.lower())
    prediction = list(prediction.lower())
    table = [[0] * (len(prediction) + 1) for _ in range(len(reference) + 1)]
    for row, expected in enumerate(reference, 1):
        for column, actual in enumerate(prediction, 1):
            table[row][column] = table[row - 1][column - 1] + 1 if expected == actual else max(table[row - 1][column], table[row][column - 1])
    length = table[-1][-1]
    precision = length / len(prediction) if prediction else 0.0
    recall = length / len(reference) if reference else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def task_type(item):
    """题型判定：ex（结构化抽取）/ mcq（选择题）/ gen（开放生成）。

    判定顺序：answer_type == structured → ex；指令里出现选择题标记 → mcq；
    指令里出现信息抽取/JSON 标记，或参考输出本身能解析成 JSON → ex；否则 gen。
    评测链路与报告统计共用这一个定义，改动会同时影响两侧，务必谨慎。
    """
    if item.get("answer_type") == "structured":
        return "ex"
    instruction = item.get("instruction", "").lower()
    if re.search(r"\b(?:multiple[- ]choice|options?)\b", instruction) or any(marker in instruction for marker in ("多项选择", "单项选择", "选择题")):
        return "mcq"
    if any(marker in instruction for marker in ("information extraction", "json format", "json格式", "json 格式", "信息抽取")):
        return "ex"
    if parse_json(item.get("output", "")) is not None:
        return "ex"
    return "gen"


def prompt_for(item):
    return f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item.get('input', '')}\n\n### Response:\n"
