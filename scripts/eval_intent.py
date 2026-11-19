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
        got = classify_intent(llm, s["query"])
        ok = got == s["intent"]
        correct += int(ok)
        print(f"{'OK  ' if ok else 'MISS'} {s['query']} -> {got} (expect {s['intent']})")
    print(f"accuracy: {correct}/{len(samples)}")
    return 0 if correct == len(samples) else 1


if __name__ == "__main__":
    raise SystemExit(main())
