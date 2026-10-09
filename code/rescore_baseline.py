import argparse
import json
import re
from pathlib import Path

from evaluate_baseline import greedy_scores, parse_json, rouge_l, task_type


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source_by_id = {record["id"]: record for record in (json.loads(line) for line in Path(args.data).open())}
    totals = {"ex": [], "mcq": [], "gen": []}
    valid_json = []
    with Path(args.output).open("w") as destination:
        for line in Path(args.predictions).open():
            result = json.loads(line)
            source = source_by_id[result["id"]]
            kind = task_type(source)
            result["task_type"] = kind
            if kind == "ex":
                reference = parse_json(source["output"])
                prediction = parse_json(result["prediction"])
                if reference is not None and prediction is not None:
                    result["loose_f1"], result["exact_f1"] = greedy_scores(reference, prediction)
                    valid_json.append(1)
                else:
                    result["loose_f1"], result["exact_f1"] = 0.0, 0.0
                    valid_json.append(0)
                totals[kind].append(result["loose_f1"])
            elif kind == "gen":
                result["rouge_l"] = rouge_l(source["output"], result["prediction"])
                totals[kind].append(result["rouge_l"])
            else:
                expected = re.search(r"\b([A-D])\b", source["output"].upper())
                actual = re.search(r"\b([A-D])\b", result["prediction"].upper())
                result["correct"] = bool(expected and actual and expected.group(1) == actual.group(1))
                totals[kind].append(float(result["correct"]))
            destination.write(json.dumps(result, ensure_ascii=False) + "\n")
    metrics = {
        "samples": sum(len(values) for values in totals.values()),
        "ex_samples": len(totals["ex"]),
        "mcq_samples": len(totals["mcq"]),
        "gen_samples": len(totals["gen"]),
        "loose_contains_f1": sum(totals["ex"]) / len(totals["ex"]) if totals["ex"] else 0.0,
        "rouge_l": sum(totals["gen"]) / len(totals["gen"]) if totals["gen"] else 0.0,
        "mcq_accuracy": sum(totals["mcq"]) / len(totals["mcq"]) if totals["mcq"] else None,
        "json_valid_rate": sum(valid_json) / len(valid_json) if valid_json else 0.0,
    }
    metrics_path = Path(args.output).with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
