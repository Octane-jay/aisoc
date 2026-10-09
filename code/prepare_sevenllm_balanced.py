import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer


TEXT_FIELDS = ("instruction", "input", "output")


def as_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_text(value):
    value = unicodedata.normalize("NFKC", as_text(value)).casefold()
    return re.sub(r"\s+", " ", value).strip()


def content_hash(*values):
    content = "\x1f".join(canonical_text(value) for value in values)
    return hashlib.sha256(content.encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_records(path):
    with Path(path).open() as source:
        return [json.loads(line) for line in source]


def has_target(record):
    output = record.get("output")
    return output is not None and (not isinstance(output, str) or bool(output.strip()))


def normalize_record(record):
    output = record["output"]
    return {
        "id": record.get("id"),
        "category": record["category"],
        "instruction": as_text(record.get("instruction")),
        "input": as_text(record.get("input")),
        "output": as_text(output),
        "answer_type": "structured" if isinstance(output, (dict, list)) else "text",
    }


def prompt_for(record):
    return f"### Instruction:\n{record['instruction']}\n\n### Input:\n{record['input']}\n\n### Response:\n"


def percentile(values, fraction):
    index = min(len(values) - 1, int(len(values) * fraction))
    return values[index]


def length_report(records, tokenizer, max_length):
    prompt_lengths = []
    output_lengths = []
    combined_lengths = []
    for record in records:
        prompt_length = len(tokenizer(prompt_for(record), add_special_tokens=False)["input_ids"])
        output_length = len(tokenizer(record["output"], add_special_tokens=False)["input_ids"]) + 1
        prompt_lengths.append(prompt_length)
        output_lengths.append(output_length)
        combined_lengths.append(prompt_length + output_length)
    report = {}
    for name, values in (("prompt", prompt_lengths), ("output", output_lengths), ("combined", combined_lengths)):
        values.sort()
        report[name] = {
            "p50": percentile(values, 0.50),
            "p90": percentile(values, 0.90),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "max": values[-1],
        }
    report["combined_over_max_length"] = sum(length > max_length for length in combined_lengths)
    return report


def write_jsonl(path, records):
    with Path(path).open("w") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-category", type=int, default=1250)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    train_path = args.source_dir / "train.jsonl"
    test_path = args.source_dir / "test.jsonl"
    train_records = read_records(train_path)
    test_records = read_records(test_path)
    train_categories = {record["category"] for record in train_records}
    test_categories = {record["category"] for record in test_records}
    shared_categories = sorted(train_categories & test_categories)
    test_input_hashes = {content_hash(record.get("input")) for record in test_records}

    grouped = defaultdict(list)
    excluded = Counter()
    seen_tasks = set()
    leakage_ids = []
    for record in train_records:
        if record["category"] not in shared_categories:
            excluded["train_only_category"] += 1
            continue
        if not has_target(record):
            excluded["empty_output"] += 1
            continue
        if content_hash(record.get("input")) in test_input_hashes:
            excluded["test_input_overlap"] += 1
            leakage_ids.append(record.get("id"))
            continue
        task_hash = content_hash(record.get("category"), record.get("instruction"), record.get("input"), record.get("output"))
        if task_hash in seen_tasks:
            excluded["duplicate_train_task"] += 1
            continue
        seen_tasks.add(task_hash)
        grouped[record["category"]].append(normalize_record(record))

    selected = []
    selected_ids = {}
    available_counts = {}
    for category in shared_categories:
        candidates = grouped[category]
        available_counts[category] = len(candidates)
        if len(candidates) < args.per_category:
            raise ValueError(f"{category} has only {len(candidates)} eligible records; need {args.per_category}")
        randomizer = random.Random(f"{args.seed}:{category}")
        randomizer.shuffle(candidates)
        category_selection = candidates[:args.per_category]
        selected.extend(category_selection)
        selected_ids[category] = [record["id"] for record in category_selection]
    random.Random(args.seed).shuffle(selected)

    normalized_test = [normalize_record(record) for record in test_records]
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    args.output_dir.mkdir(parents=True)
    write_jsonl(args.output_dir / "train.jsonl", selected)
    write_jsonl(args.output_dir / "test.jsonl", normalized_test)
    length_reports = {
        "train": length_report(selected, tokenizer, args.max_length),
        "test": length_report(normalized_test, tokenizer, args.max_length),
    }
    manifest = {
        "version": "sevenllm-24class-balanced-30k-v1",
        "source": {
            "train": {"path": str(train_path), "sha256": file_hash(train_path), "records": len(train_records)},
            "test": {"path": str(test_path), "sha256": file_hash(test_path), "records": len(test_records)},
        },
        "selection": {
            "shared_categories": shared_categories,
            "per_category": args.per_category,
            "seed": args.seed,
            "selected_records": len(selected),
            "available_records_by_category": available_counts,
            "selected_ids_by_category": selected_ids,
            "excluded": dict(excluded),
        },
        "schema": {"fields": ["id", "category", "instruction", "input", "output", "answer_type"], "omitted_source_field": "thought"},
        "answer_type_counts": {
            "train": dict(Counter(record["answer_type"] for record in selected)),
            "test": dict(Counter(record["answer_type"] for record in normalized_test)),
        },
        "lengths": {"tokenizer": args.model, "max_length": args.max_length, **length_reports},
        "output": {
            "train": {"path": "train.jsonl", "sha256": file_hash(args.output_dir / "train.jsonl")},
            "test": {"path": "test.jsonl", "sha256": file_hash(args.output_dir / "test.jsonl")},
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    audit = {"test_input_overlap_train_ids": leakage_ids, "excluded": dict(excluded)}
    (args.output_dir / "duplicate_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "length_report.json").write_text(json.dumps(length_reports, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"selected_records": len(selected), "categories": len(shared_categories), "excluded": dict(excluded)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
