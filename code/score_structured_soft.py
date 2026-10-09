import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer

from evaluate_baseline import parse_json, task_type
from paths import semantic_cache_dir


ENTITY_RE = re.compile(r"(?:^|[_ .-])(ip|ipv4|ipv6|hash|md5|sha1|sha256|domain|url|cve|port|date|time|timestamp)(?:$|[_ .-])", re.I)


def normalize(value):
    value = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    value = re.sub(r"\s+", "", value)
    punctuation = str.maketrans({
        "。": ".", "．": ".", "，": ",", "、": ",", "；": ";", "：": ":",
        "！": "!", "？": "?", "（": "(", "）": ")", "【": "[", "】": "]",
    })
    return value.translate(punctuation)


def flatten(value, path=()):
    if isinstance(value, dict):
        items = []
        for key, child in value.items():
            items.extend(flatten(child, path + (str(key),)))
        return items
    if isinstance(value, list):
        items = []
        for index, child in enumerate(value):
            items.extend(flatten(child, path + (str(index),)))
        return items
    if value is None or value == "":
        return []
    return [(path, str(value))]


def field_name(path):
    return path[-1] if path else ""


def is_entity(path, value):
    return bool(ENTITY_RE.search("_".join(path))) or bool(re.fullmatch(r"(?:[0-9a-f]{32}|[0-9a-f]{40}|[0-9a-f]{64}|(?:\d{1,3}\.){3}\d{1,3})", normalize(value)))


def score_pair(reference, prediction, similarity):
    ref_items = flatten(reference)
    pred_items = flatten(prediction)
    if not ref_items or not pred_items:
        return 0.0
    matrix = np.zeros((len(ref_items), len(pred_items)), dtype=np.float32)
    for i, (ref_path, ref_value) in enumerate(ref_items):
        ref_norm = normalize(ref_value)
        for j, (pred_path, pred_value) in enumerate(pred_items):
            pred_norm = normalize(pred_value)
            if field_name(ref_path) != field_name(pred_path) and field_name(ref_path) and field_name(pred_path):
                continue
            if ref_norm == pred_norm or (ref_norm and ref_norm in pred_norm):
                matrix[i, j] = 1.0
            elif not is_entity(ref_path, ref_value) and similarity is not None:
                matrix[i, j] = float(similarity[i, j]) if similarity[i, j] >= 0.80 else 0.0
    rows, cols = linear_sum_assignment(-matrix)
    matched = float(matrix[rows, cols].sum())
    precision = matched / len(pred_items)
    recall = matched / len(ref_items)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--semantic-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--cache-dir", type=Path, default=semantic_cache_dir())
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    source_by_id = {r["id"]: r for r in (json.loads(line) for line in Path(args.data).open())}
    rows = []
    texts = []
    for line in Path(args.predictions).open():
        prediction = json.loads(line)
        source = source_by_id[prediction["id"]]
        kind = "ex" if source.get("answer_type") == "structured" else task_type({**source, "instruction": str(source.get("instruction", ""))})
        if kind != "ex":
            continue
        reference = parse_json(str(source.get("output", "")))
        answer = parse_json(str(prediction.get("prediction", "")))
        if reference is None or answer is None:
            rows.append({"id": prediction["id"], "soft_f1": 0.0, "json_valid": 0.0})
            continue
        ref_items, pred_items = flatten(reference), flatten(answer)
        texts.extend([v for _, v in ref_items] + [v for _, v in pred_items])
        rows.append({"id": prediction["id"], "reference": reference, "prediction": answer, "json_valid": 1.0})
    model = SentenceTransformer(args.semantic_model, cache_folder=str(args.cache_dir), device=args.device)
    embeddings = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True) if texts else []
    offset = 0
    for row in rows:
        if "reference" not in row:
            continue
        ref_items, pred_items = flatten(row["reference"]), flatten(row["prediction"])
        ref_n, pred_n = len(ref_items), len(pred_items)
        ref_emb = embeddings[offset:offset + ref_n]
        pred_emb = embeddings[offset + ref_n:offset + ref_n + pred_n]
        offset += ref_n + pred_n
        row["soft_f1"] = score_pair(row["reference"], row["prediction"], ref_emb @ pred_emb.T)
        row.pop("reference")
        row.pop("prediction")
    mean_f1 = sum(r["soft_f1"] for r in rows) / len(rows) if rows else 0.0
    valid_rate = sum(r["json_valid"] for r in rows) / len(rows) if rows else 0.0
    output = Path(args.output)
    with output.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics = {"structured_samples": len(rows), "field_aware_soft_f1": mean_f1, "json_valid_rate": valid_rate, "matching": "path-aware Hungarian; exact/entity or embedding cosine >= 0.80"}
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
