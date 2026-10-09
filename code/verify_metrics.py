"""一致性校验：重跑产物 vs 仓库内冻结结果。

1) 数据文件 sha256 与 manifest.json 记录比对（防止数据链路漂移）；
2) results/<tag>/*.metrics.json 的每个数值与重跑结果比对，输出差值。

用法：python code/verify_metrics.py --reference results --candidate ../poc-runs/results --data data/SEVENLLM-24class-balanced-30k-v1
"""

import argparse
import hashlib
import json
from pathlib import Path


TOLERANCE = 1e-6


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten(value, prefix=""):
    if isinstance(value, dict):
        items = {}
        for key, child in value.items():
            items.update(flatten(child, f"{prefix}{key}."))
        return items
    return {prefix.rstrip("."): value}


def compare(reference, candidate, tolerance=TOLERANCE):
    rows, ok = [], True
    for key, expected in sorted(reference.items()):
        actual = candidate.get(key, "<缺失>")
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            delta = abs(float(actual) - float(expected))
            status = "OK" if delta <= tolerance else ("接近" if delta <= 1e-3 else "不一致")
            if delta > 1e-3:
                ok = False
            rows.append((key, f"{expected:.6f}", f"{float(actual):.6f}", f"{delta:.2e}", status))
        else:
            status = "OK" if expected == actual else "不一致"
            ok &= expected == actual
            rows.append((key, str(expected), str(actual), "-", status))
    return rows, ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = parser.parse_args()
    all_ok = True

    if args.data:
        manifest_path = args.data / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            print("== 数据校验 ==")
            for name, record in manifest["output"].items():
                actual = sha256(args.data / record["path"])
                status = "OK" if actual == record["sha256"] else "不一致"
                all_ok &= actual == record["sha256"]
                print(f"{record['path']}: {status}（{actual[:16]} vs {record['sha256'][:16]}）")
            print()

    for tag in ("baseline", "finetuned"):
        for metrics_path in sorted((args.reference / tag).glob("*.metrics.json")):
            candidate_path = args.candidate / tag / metrics_path.name
            print(f"== {tag}/{metrics_path.name} ==")
            if not candidate_path.exists():
                print(f"  跳过：缺少重跑结果 {candidate_path}")
                all_ok = False
                continue
            reference = flatten(json.loads(metrics_path.read_text()))
            candidate = flatten(json.loads(candidate_path.read_text()))
            rows, ok = compare(reference, candidate, args.tolerance)
            for key, expected, actual, delta, status in rows:
                if status != "OK":
                    print(f"  {key}: 冻结 {expected} / 重跑 {actual} / Δ{delta} → {status}")
            print(f"  共 {len(rows)} 项，{'全部一致' if ok else '存在差异'}")
            all_ok &= ok
            print()

    print("校验通过：重跑结果与冻结结果一致。" if all_ok else "校验失败：请检查上面的差异项。")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
