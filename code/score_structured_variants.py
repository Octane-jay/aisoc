import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer

from evaluate_baseline import parse_json, task_type
from score_structured_soft import flatten, is_entity, score_pair
from paths import semantic_cache_dir


FIELD_ALIASES = {
    "malware": "malware_name",
    "malware_name": "malware_name",
    "恶意软件": "malware_name",
    "病毒名称": "malware_name",
    "漏洞编号": "cve",
    "cve编号": "cve",
    "cve_id": "cve",
    "cve": "cve",
    "漏洞名称": "vulnerability_name",
    "漏洞类型": "vulnerability_type",
    "受影响组件": "affected_component",
    "affected_systems": "affected_system",
    "受影响系统": "affected_system",
    "攻击目标平台": "target_platform",
    "target_platform": "target_platform",
    "c2地址": "c2_address",
    "c&c地址": "c2_address",
    "c2服务器地址": "c2_address",
    "c2_server": "c2_address",
    "攻击组织": "attack_group",
    "攻击团伙": "attack_group",
    "apt组织": "attack_group",
    "attackgroup": "attack_group",
    "攻击手段": "attack_method",
    "攻击方式": "attack_method",
    "恶意行为": "malicious_behavior",
    "行为特征": "malicious_behavior",
    "传播方式": "propagation",
    "文件类型": "file_type",
    "文件名": "file_name",
    "md5": "hash",
    "sha1": "hash",
    "sha256": "hash",
}


def canonical_field(path):
    field = unicodedata.normalize("NFKC", str(path[-1] if path else "")).casefold()
    compact = re.sub(r"[\s_.:：/\\-]", "", field)
    return FIELD_ALIASES.get(field, FIELD_ALIASES.get(compact, compact))


def normalize_value(value):
    value = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    value = re.sub(r"\s+", "", value)
    value = value.replace("cve-", "cve")
    return re.sub(r"[。。，、；：！？（）【】\[\]{}]", "", value)


def variant_score(reference, prediction, similarity, threshold):
    ref_items = flatten(reference)
    pred_items = flatten(prediction)
    if not ref_items or not pred_items:
        return 0.0
    matrix = np.zeros((len(ref_items), len(pred_items)), dtype=np.float32)
    for i, (ref_path, ref_value) in enumerate(ref_items):
        ref_norm = normalize_value(ref_value)
        for j, (pred_path, pred_value) in enumerate(pred_items):
            if canonical_field(ref_path) != canonical_field(pred_path):
                continue
            pred_norm = normalize_value(pred_value)
            if ref_norm == pred_norm or (ref_norm and ref_norm in pred_norm):
                matrix[i, j] = 1.0
            elif not is_entity(ref_path, ref_value) and similarity is not None:
                if similarity[i, j] >= threshold:
                    matrix[i, j] = float(similarity[i, j])
    rows, cols = linear_sum_assignment(-matrix)
    matched = float(matrix[rows, cols].sum())
    precision = matched / len(pred_items)
    recall = matched / len(ref_items)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate(args):
    source_by_id = {r["id"]: r for r in (json.loads(line) for line in Path(args.data).open())}
    rows, texts = [], []
    for line in Path(args.predictions).open():
        prediction = json.loads(line)
        source = source_by_id[prediction["id"]]
        kind = "ex" if source.get("answer_type") == "structured" else task_type({**source, "instruction": str(source.get("instruction", ""))})
        if kind != "ex":
            continue
        reference = parse_json(str(source.get("output", "")))
        answer = parse_json(str(prediction.get("prediction", "")))
        if reference is None or answer is None:
            rows.append({"strict_f1": 0.0, "normalized_soft_f1": 0.0, "json_valid": 0.0})
            continue
        ref_items, pred_items = flatten(reference), flatten(answer)
        texts.extend([v for _, v in ref_items] + [v for _, v in pred_items])
        rows.append({"reference": reference, "prediction": answer, "json_valid": 1.0})
    model = SentenceTransformer(args.semantic_model, cache_folder=str(args.cache_dir), device=args.device)
    embeddings = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True) if texts else []
    offset = 0
    for row in rows:
        if "reference" not in row:
            continue
        ref_items, pred_items = flatten(row["reference"]), flatten(row["prediction"])
        ref_n = len(ref_items)
        pred_n = len(pred_items)
        ref_emb = embeddings[offset:offset + ref_n]
        pred_emb = embeddings[offset + ref_n:offset + ref_n + pred_n]
        offset += ref_n + pred_n
        similarity = ref_emb @ pred_emb.T
        row["strict_f1"] = score_pair(row["reference"], row["prediction"], similarity)
        row["normalized_soft_f1"] = variant_score(row["reference"], row["prediction"], similarity, args.threshold)
        row.pop("reference")
        row.pop("prediction")
    count = len(rows)
    metrics = {
        "structured_samples": count,
        "strict_macro_f1": sum(r["strict_f1"] for r in rows) / count if count else 0.0,
        "normalized_soft_macro_f1": sum(r["normalized_soft_f1"] for r in rows) / count if count else 0.0,
        "json_valid_rate": sum(r["json_valid"] for r in rows) / count if count else 0.0,
        "semantic_threshold": args.threshold,
        "matching": "canonical field aliases; exact/entity or embedding cosine threshold",
    }
    output = Path(args.output)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--semantic-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--cache-dir", type=Path, default=semantic_cache_dir())
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.72)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
