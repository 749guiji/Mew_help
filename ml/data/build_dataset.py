"""建集:真实 + 模拟 → 清洗 → 分层抽样 → 增强训练集 → 写 train/val/test.jsonl。"""

from __future__ import annotations

import json
from pathlib import Path

from ml.data.augment import augment
from ml.data.clean import clean
from ml.data.split import stratified_split

DATA = Path("ml/data")


def _load(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_dataset() -> dict:
    raw_synth = _load(DATA / "raw_synthetic.jsonl")
    hand = _load(DATA / "hand_written.jsonl")

    seen: set[str] = set()
    items: list[dict] = []
    for it in raw_synth + hand:
        text = clean(it["text"])
        if text and text not in seen:
            seen.add(text)
            items.append({"text": text, "labels": it["labels"]})

    train, val, test = stratified_split(items, seed=42)

    # 增强只扩训练集
    aug: list[dict] = []
    for it in train:
        aug.append(it)
        for v in augment(it["text"]):
            aug.append({"text": v, "labels": it["labels"]})

    for name, part in (("train", aug), ("val", val), ("test", test)):
        (DATA / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in part), encoding="utf-8"
        )
    return {"train": len(aug), "val": len(val), "test": len(test)}


if __name__ == "__main__":
    print(build_dataset())
