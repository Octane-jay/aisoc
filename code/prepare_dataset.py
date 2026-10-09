import argparse
import json
from collections import Counter
from pathlib import Path


TEXT_FIELDS = ("instruction", "input", "thought", "output")


def as_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def normalize_split(source_path, destination_path):
    category_counts = Counter()
    record_count = 0
    with Path(source_path).open() as source, Path(destination_path).open("w") as destination:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {source_path}:{line_number}") from error
            category = record.get("category")
            if not isinstance(category, str) or not category:
                raise ValueError(f"Missing category at {source_path}:{line_number}")
            normalized = {"id": record.get("id"), "category": category}
            normalized.update({field: as_text(record.get(field)) for field in TEXT_FIELDS})
            destination.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            category_counts[category] += 1
            record_count += 1
    return category_counts, record_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ("train", "test"):
        counts, record_count = normalize_split(
            args.source_dir / f"{split}.jsonl", args.output_dir / f"{split}.jsonl"
        )
        summary[split] = {"records": record_count, "categories": dict(sorted(counts.items()))}
    train_categories = set(summary["train"]["categories"])
    test_categories = set(summary["test"]["categories"])
    summary["category_alignment"] = {
        "test_categories_missing_from_train": sorted(test_categories - train_categories),
        "train_categories_missing_from_test": sorted(train_categories - test_categories),
        "shared_categories": sorted(train_categories & test_categories),
    }
    with (args.output_dir / "dataset_summary.json").open("w") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)
        summary_file.write("\n")
    print(json.dumps(summary["category_alignment"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
