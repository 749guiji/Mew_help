"""大模型生成 17 类多标签模拟样本(飞轮池空,补足 1700)。

每个类目批量生成 ~100 条,一条一个 JSON 数组(call 少、快)。产 raw_synthetic.jsonl。
"""

from __future__ import annotations

import json
from pathlib import Path

from app.core.config import get_settings
from app.core.llm import get_llm
from ml.taxonomy import TOPICS, topic_names

OUT = Path("ml/data/raw_synthetic.jsonl")
PER_CLASS = 100
BATCH = 25  # 每 call 生成条数

_VALID = set(topic_names())


def _prompt(name: str, boundary: str, n: int) -> str:
    lines = "\n".join(f"- {t['name']}：{t['boundary']}" for t in TOPICS)
    return (
        f"你是电商售后客服语料生成器。生成 {n} 条真实用户提问,"
        f"主主题要覆盖「{name}」({boundary}),并自然穿插其他主题。\n"
        f"多标签规则:一句字面提到几个诉求就标几个标签,一个不多一个不少;"
        f"只从下面 17 类里取标签。\n\n17 类术语表:\n{lines}\n\n"
        f"只输出一个 JSON 数组,每项形如 "
        f'{{"text": "<问题>", "labels": ["<类名>", ...]}},不要任何其他文字、不要代码围栏。'
    )


def _parse(text: str) -> list[dict]:
    """从 LLM 输出里抠出 JSON 数组并校验标签。"""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except Exception:
        return []
    out = []
    for it in data:
        if not isinstance(it, dict):
            continue
        t, labels = it.get("text"), it.get("labels")
        if not isinstance(t, str) or not isinstance(labels, list):
            continue
        labels = [l for l in labels if l in _VALID]
        if t.strip() and labels:
            out.append({"text": t.strip(), "labels": labels})
    return out


def generate(per_class: int = PER_CLASS, batch: int = BATCH) -> int:
    llm = get_llm(get_settings())
    written = 0
    with OUT.open("w", encoding="utf-8") as fh:
        for t in TOPICS:
            for _ in range(per_class // batch):
                try:
                    resp = llm.invoke(_prompt(t["name"], t["boundary"], batch))
                    items = _parse(resp.content)
                except Exception as e:  # noqa: BLE001
                    print(f"[error] {t['name']}: {e}", flush=True)
                    continue
                for it in items:
                    fh.write(json.dumps(it, ensure_ascii=False) + "\n")
                    written += 1
                print(f"{t['name']}: +{len(items)}", flush=True)
    print(f"done: {written} 条 -> {OUT}")
    return written


if __name__ == "__main__":
    generate()
