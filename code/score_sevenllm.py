import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import jieba
import numpy as np
from sentence_transformers import SentenceTransformer

from evaluate_baseline import greedy_scores, leaves, parse_json, task_type
from paths import semantic_cache_dir


def tokenize_for_rouge(text):
    if re.search(r"[\u4e00-\u9fff]", text):
        return [token for token in jieba.lcut(text) if token.strip()]
    return text.split()


def rouge_l_tokens(reference, prediction):
    reference_tokens = tokenize_for_rouge(reference)
    prediction_tokens = tokenize_for_rouge(prediction)
    if not reference_tokens or not prediction_tokens:
        return 0.0
    previous = [0] * (len(prediction_tokens) + 1)
    for expected in reference_tokens:
        current = [0]
        for index, actual in enumerate(prediction_tokens, start=1):
            current.append(previous[index - 1] + 1 if expected == actual else max(previous[index], current[-1]))
        previous = current
    shared = previous[-1]
    precision = shared / len(prediction_tokens)
    recall = shared / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def semantic_text(answer, structured):
    if structured:
        parsed = parse_json(answer)
        if parsed is not None:
            return "\n".join(str(value) for value in leaves(parsed))
    return answer


def mean(values):
    return sum(values) / len(values) if values else None


def grouped_metrics(rows):
    metrics = {}
    for name, group in rows.items():
        structured = [row for row in group if row["task_type"] == "ex"]
        generated = [row for row in group if row["task_type"] == "gen"]
        metrics[name] = {
            "samples": len(group),
            "structured_samples": len(structured),
            "generation_samples": len(generated),
            "rouge_l_jieba_whitespace": mean([row["rouge_l_jieba_whitespace"] for row in group]),
            "semantic_similarity": mean([row["semantic_similarity"] for row in group]),
            "structured_loose_leaf_f1": mean([row["loose_f1"] for row in structured]),
            "structured_exact_leaf_f1": mean([row["exact_f1"] for row in structured]),
            "structured_json_valid_rate": mean([row["json_valid"] for row in structured]),
            "generation_rouge_l_jieba_whitespace": mean([row["rouge_l_jieba_whitespace"] for row in generated]),
            "generation_semantic_similarity": mean([row["semantic_similarity"] for row in generated]),
        }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--semantic-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--cache-dir", type=Path, default=semantic_cache_dir())
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    source_by_id = {record["id"]: record for record in (json.loads(line) for line in Path(args.data).open())}
    rows = []
    for line in Path(args.predictions).open():
        prediction = json.loads(line)
        source = source_by_id[prediction["id"]]
        kind = task_type(source)
        structured = kind == "ex"
        reference = source["output"]
        answer = prediction["prediction"]
        row = {
            "id": prediction["id"],
            "category": source["category"],
            "task_type": kind,
            "rouge_l_jieba_whitespace": rouge_l_tokens(reference, answer),
            "semantic_reference": semantic_text(reference, structured),
            "semantic_prediction": semantic_text(answer, structured),
        }
        if structured:
            parsed_reference = parse_json(reference)
            parsed_prediction = parse_json(answer)
            row["json_valid"] = float(parsed_prediction is not None)
            row["loose_f1"], row["exact_f1"] = (
                greedy_scores(parsed_reference, parsed_prediction) if parsed_reference is not None and parsed_prediction is not None else (0.0, 0.0)
            )
        rows.append(row)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(args.semantic_model, cache_folder=str(args.cache_dir), device=args.device)
    reference_embeddings = model.encode([row["semantic_reference"] for row in rows], batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    prediction_embeddings = model.encode([row["semantic_prediction"] for row in rows], batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    similarities = np.sum(reference_embeddings * prediction_embeddings, axis=1)
    for row, similarity in zip(rows, similarities):
        row["semantic_similarity"] = float(similarity)
        row.pop("semantic_reference")
        row.pop("semantic_prediction")

    groups = {"overall": rows}
    for category in sorted({row["category"] for row in rows}):
        groups[f"category:{category}"] = [row for row in rows if row["category"] == category]
    metrics = {
        "protocol": {
            "rouge_l": "English whitespace tokens; Chinese jieba tokens",
            "semantic_similarity_model": args.semantic_model,
            "semantic_similarity": "cosine similarity of normalized multilingual sentence embeddings",
            "structured_f1": "leaf-value greedy exact and reference-substring matching; auxiliary only",
        },
        "metrics": grouped_metrics(groups),
    }
    with Path(args.output).open("w") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics_path = Path(args.output).with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metrics["metrics"]["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
