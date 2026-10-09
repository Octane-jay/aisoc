import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer

from evaluate_baseline import leaves, parse_json, task_type
from paths import semantic_cache_dir


def normalize(value):
    value = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    return re.sub(r"\s+", "", value)


def score(reference, prediction, similarity, threshold):
    reference = [normalize(value) for value in leaves(reference)]
    prediction = [normalize(value) for value in leaves(prediction)]
    if not reference or not prediction:
        return 0.0
    matrix = np.zeros((len(reference), len(prediction)), dtype=np.float32)
    for row, expected in enumerate(reference):
        for column, actual in enumerate(prediction):
            if expected == actual or (expected and expected in actual):
                matrix[row, column] = 1.0
            elif similarity is not None and similarity[row, column] >= threshold:
                matrix[row, column] = 1.0
    rows, columns = linear_sum_assignment(-matrix)
    matched = float(matrix[rows, columns].sum())
    precision = matched / len(prediction)
    recall = matched / len(reference)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--thresholds", default="0.65,0.70,0.75,0.80")
    parser.add_argument("--semantic-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--cache-dir", type=Path, default=semantic_cache_dir())
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    thresholds = [float(value) for value in args.thresholds.split(",")]
    source_by_id = {record["id"]: record for record in (json.loads(line) for line in Path(args.data).open())}
    rows, texts = [], []
    for line in Path(args.predictions).open():
        prediction = json.loads(line)
        source = source_by_id[prediction["id"]]
        if task_type(source) != "ex":
            continue
        reference = parse_json(str(source.get("output", "")))
        answer = parse_json(str(prediction.get("prediction", "")))
        if reference is None or answer is None:
            rows.append({"id": prediction["id"], "json_valid": 0.0, "scores": {str(t): 0.0 for t in thresholds}})
            continue
        reference_leaves, prediction_leaves = leaves(reference), leaves(answer)
        texts.extend([str(value) for value in reference_leaves] + [str(value) for value in prediction_leaves])
        rows.append({"id": prediction["id"], "reference": reference_leaves, "prediction": prediction_leaves, "json_valid": 1.0})
    model = SentenceTransformer(args.semantic_model, cache_folder=str(args.cache_dir), device=args.device)
    embeddings = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True) if texts else []
    offset = 0
    for row in rows:
        if "reference" not in row:
            continue
        reference_leaves, prediction_leaves = row.pop("reference"), row.pop("prediction")
        ref_n, pred_n = len(reference_leaves), len(prediction_leaves)
        similarity = embeddings[offset:offset + ref_n] @ embeddings[offset + ref_n:offset + ref_n + pred_n].T
        offset += ref_n + pred_n
        row["scores"] = {str(t): score(reference_leaves, prediction_leaves, similarity, t) for t in thresholds}
    count = len(rows)
    metrics = {"structured_samples": count, "json_valid_rate": sum(row["json_valid"] for row in rows) / count if count else 0.0, "coarse_embedding_macro_f1": {key: sum(row["scores"][key] for row in rows) / count if count else 0.0 for key in map(str, thresholds)}, "matching": "field/path agnostic; exact/substring or cosine similarity threshold; one-to-one Hungarian matching"}
    output = Path(args.output)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
