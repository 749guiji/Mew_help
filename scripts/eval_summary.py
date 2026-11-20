"""Eval the ch07 summary prompt against annotated samples.

Usage: PYTHONPATH=. .venv/Scripts/python.exe scripts/eval_summary.py
"""

import json
from pathlib import Path

from app.core.config import get_settings
from app.core.llm import get_llm
from app.core.summarizer import _summarize


def main() -> int:
    samples = json.loads(
        Path("tests/data/summary_samples.json").read_text(encoding="utf-8")
    )
    llm = get_llm(get_settings())
    correct = 0
    for s in samples:
        turns = s["conversation"].split("\n")
        got = _summarize(llm, turns)
        ok = all(t in got for t in s["must_contain"]) and not any(
            t in got for t in s.get("must_not", [])
        )
        correct += int(ok)
        print(f"{'OK  ' if ok else 'MISS'} -> {got}")
    print(f"accuracy: {correct}/{len(samples)}")
    return 0 if correct == len(samples) else 1


if __name__ == "__main__":
    raise SystemExit(main())
