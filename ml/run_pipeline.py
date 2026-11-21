"""一键跑完整流水线:生成模拟样本 → 建集 → 训练 → 评测 → 导出 ONNX。"""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    from ml.data.build_dataset import build_dataset
    from ml.data.generate_synthetic import generate
    from ml.eval import evaluate
    from ml.export_onnx import export
    from ml.train import train

    print("=== 1. 生成模拟样本(1700 条,慢) ===", flush=True)
    n = generate()
    print(f"生成 {n} 条", flush=True)

    print("=== 2. 建集 ===", flush=True)
    d = build_dataset()
    print(d, flush=True)

    print("=== 3. 训练 ===", flush=True)
    best = train()

    print("=== 4. 评测 ===", flush=True)
    report = evaluate(Path(best), Path("ml/data/test.jsonl"))
    print("macro:", report["macro"], flush=True)

    print("=== 5. 导出 ONNX ===", flush=True)
    export(best)

    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
