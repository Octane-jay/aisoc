import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 共享纯函数集中在 metrics_common（只依赖标准库）；这里重新导出，
# 使 rescore / score_* 里历史写法 `from evaluate_baseline import ...` 继续可用。
from metrics_common import greedy_scores, leaves, normalize, parse_json, prompt_for, rouge_l, task_type  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter or args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    records = [json.loads(line) for line in Path(args.data).open()]
    if args.smoke:
        samples = {}
        for record in records:
            samples.setdefault(task_type(record), record)
        records = [samples[kind] for kind in ("ex", "mcq", "gen") if kind in samples]
    elif args.max_samples is not None:
        records = records[:args.max_samples]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    totals = {"ex": [], "mcq": [], "gen": []}
    valid_json = []
    with output_path.open("w") as stream:
        for index, item in enumerate(records, 1):
            encoded = tokenizer(prompt_for(item), return_tensors="pt").to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
            answer = tokenizer.decode(generated[0][encoded.input_ids.shape[1]:], skip_special_tokens=True).strip()
            kind = task_type(item)
            result = {"id": item.get("id", index), "category": item.get("category"), "task_type": kind, "prediction": answer, "reference": item.get("output", "")}
            if kind == "ex":
                reference = parse_json(item.get("output", ""))
                prediction = parse_json(answer)
                if reference is not None and prediction is not None:
                    result["loose_f1"], result["exact_f1"] = greedy_scores(reference, prediction)
                    valid_json.append(1)
                else:
                    result["loose_f1"], result["exact_f1"] = 0.0, 0.0
                    valid_json.append(0)
                totals[kind].append(result["loose_f1"])
            elif kind == "gen":
                result["rouge_l"] = rouge_l(item.get("output", ""), answer)
                totals[kind].append(result["rouge_l"])
            else:
                expected = re.search(r"\b([A-D])\b", item.get("output", "").upper())
                actual = re.search(r"\b([A-D])\b", answer.upper())
                result["correct"] = bool(expected and actual and expected.group(1) == actual.group(1))
                totals[kind].append(float(result["correct"]))
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            if index % 25 == 0:
                stream.flush()
                print(f"processed={index}/{len(records)}", flush=True)
    metrics = {
        "samples": len(records),
        "ex_samples": len(totals["ex"]),
        "mcq_samples": len(totals["mcq"]),
        "gen_samples": len(totals["gen"]),
        "loose_contains_f1": sum(totals["ex"]) / len(totals["ex"]) if totals["ex"] else 0.0,
        "rouge_l": sum(totals["gen"]) / len(totals["gen"]) if totals["gen"] else 0.0,
        "mcq_accuracy": sum(totals["mcq"]) / len(totals["mcq"]) if totals["mcq"] else 0.0,
        "json_valid_rate": sum(valid_json) / len(valid_json) if valid_json else 0.0,
    }
    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
