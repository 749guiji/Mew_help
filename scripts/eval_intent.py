"""Eval intent classification against annotated samples (non-TDD prompt eval).

Usage: PYTHONPATH=. .venv/Scripts/python.exe scripts/eval_intent.py
"""

import json
from pathlib import Path

from app.core.config import get_settings
from app.core.llm import get_llm
from app.graph.nodes import classify_intent


def main() -> int:
    samples = json.loads(Path("tests/data/intent_samples.json").read_text(encoding="utf-8"))
    llm = get_llm(get_settings())
    correct = 0
    for s in samples:
        r = classify_intent(llm, s["query"])
        ok = r.intent == s["intent"]
        correct += int(ok)
        print(
            f"{'OK  ' if ok else 'MISS'} {s['query']} -> {r.intent} "
            f"(conf={r.confidence}, expect {s['intent']})"
        )
    print(f"accuracy: {correct}/{len(samples)}")

    # 多轮漂移:物流→退款→物流,当前句意图应随上下文(读历史消歧)
    drift = [
        ("那它现在到哪了", "用户:订单1001到哪了\n客服:已发货\n用户:那我想退了\n客服:好的帮您看下退款\n", "物流"),
        ("那它能退吗", "用户:订单1001到哪了\n客服:已发货\n", "退款退货"),
    ]
    for q, hist, expect in drift:
        r = classify_intent(llm, q, hist)
        ok = r.intent == expect
        print(f"{'OK  ' if ok else 'MISS'} [多轮] {q!r} -> {r.intent} (expect {expect})")

    return 0 if correct == len(samples) else 1


if __name__ == "__main__":
    raise SystemExit(main())
