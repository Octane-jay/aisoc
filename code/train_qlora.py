import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq, Trainer, TrainingArguments


CORE_CATEGORIES = {
    "Cybersecurity Event Classification", "Important Event Extraction", "Key Entity Recognition", "Summary Generation", "Time Element Acquisition", "Correlation Analysis",
    "Vulnerability Information Extraction", "Malware Feature Extraction", "Impact Scope", "Risk Assessment",
    "Threat Analysis", "Security Alert Generation", "Incident Response Planning", "Protection Strategy Research", "Attack Strategy Analysis",
    "Attack Means Analysis", "Attack Target Intelligence Gathering", "Attack Tool Identification", "Attacker Information Extraction", "Attack Intent Analysis",
    "Domain Intelligence Acquisition", "Encryption-Decryption Algorithm Identification", "Security Policy Audit", "Trend Prediction",
}


def build_prompt(item):
    return f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item.get('input', '')}\n\n### Response:\n"


def as_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def choose_records(path, categories, limit, seed):
    records = []
    with Path(path).open() as stream:
        for line in stream:
            item = json.loads(line)
            if item.get("category") in categories:
                records.append({
                    "category": item["category"],
                    "instruction": as_text(item.get("instruction")),
                    "input": as_text(item.get("input")),
                    "output": as_text(item.get("output")),
                })
    if not limit or limit >= len(records):
        return records
    groups = {}
    for item in records:
        groups.setdefault(item["category"], []).append(item)
    randomizer = random.Random(seed)
    for group in groups.values():
        randomizer.shuffle(group)
    selected = []
    remaining = limit
    ordered_groups = sorted(groups.items())
    for index, (_, group) in enumerate(ordered_groups):
        quota = remaining // (len(ordered_groups) - index)
        selected.extend(group[:quota])
        remaining -= quota
    randomizer.shuffle(selected)
    return selected


def tokenize(example, tokenizer, max_length):
    prompt_tokens = tokenizer(build_prompt(example), add_special_tokens=False)
    answer_tokens = tokenizer(str(example.get("output", "")), add_special_tokens=False)
    answer_ids = answer_tokens["input_ids"] + [tokenizer.eos_token_id]
    if len(answer_ids) > max_length:
        raise ValueError(f"Answer exceeds max_length={max_length}")
    prompt_ids = prompt_tokens["input_ids"]
    available_prompt_tokens = max_length - len(answer_ids)
    if len(prompt_ids) > available_prompt_tokens:
        prefix_length = available_prompt_tokens // 2
        suffix_length = available_prompt_tokens - prefix_length
        prompt_ids = prompt_ids[:prefix_length] + prompt_ids[-suffix_length:]
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-train-samples", type=int, default=30000)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()
    records = choose_records(args.train, CORE_CATEGORIES, args.max_train_samples, args.seed)
    selected_path = Path(args.output) / "selected_train.jsonl"
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    # 记录本次运行的全部可复现参数（种子、长度、环境版本），换机后可直接对照。
    run_config = {
        "args": vars(args),
        "seed": args.seed,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": sys.version.split()[0],
        "lora": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM",
                 "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
        "quantization": {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "bfloat16", "bnb_4bit_use_double_quant": True},
        "training_arguments": {"num_train_epochs": 1, "learning_rate": 1e-5, "lr_scheduler_type": "cosine",
                               "warmup_ratio": 0.03, "bf16": True, "tf32": True, "optim": "paged_adamw_8bit",
                               "group_by_length": True, "save_steps": 250, "save_total_limit": 2, "logging_steps": 1},
        "selected_samples": len(records),
        "categories": dict(sorted(Counter(item["category"] for item in records).items())),
    }
    (Path(args.output) / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2) + "\n")
    with selected_path.open("w") as stream:
        for item in records:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps({"selected_samples": len(records), "categories": Counter(item["category"] for item in records)}, ensure_ascii=False, indent=2), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quantization, torch_dtype=torch.bfloat16, device_map={"": 0})
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    dataset = Dataset.from_list(records)
    dataset = dataset.map(lambda example: tokenize(example, tokenizer, args.max_length), remove_columns=dataset.column_names, desc="Tokenizing")
    training_args = TrainingArguments(output_dir=args.output, num_train_epochs=1, per_device_train_batch_size=args.per_device_train_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps, learning_rate=1e-5, logging_steps=1, save_steps=250, save_total_limit=2, bf16=True, tf32=True, gradient_checkpointing=args.gradient_checkpointing, group_by_length=True, optim="paged_adamw_8bit", lr_scheduler_type="cosine", warmup_ratio=0.03, seed=args.seed, report_to=[], disable_tqdm=False, remove_unused_columns=False)
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt"))
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"saved_model={args.output}", flush=True)


if __name__ == "__main__":
    main()
