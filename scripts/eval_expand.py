"""Eval query expansion against annotated samples (must-cover key aspects).

Usage: PYTHONPATH=. .venv/Scripts/python.exe scripts/eval_expand.py
"""

import json
from pathlib import Path

from app.core.config import get_settings
from app.core.llm import get_llm
from app.graph.nodes import expand_queries


def main() -> int:
    samples = json.loads(
        Path("tests/data/expand_samples.json").read_text(encoding="utf-8")
    )
    llm = get_llm(get_settings())
    correct = 0
    for s in samples:
        queries = expand_queries(llm, s["query"])
        ok = len(queries) >= 2 and all(
            any(t in q for q in queries) for t in s.get("must_cover", [])
        )
        correct += int(ok)
        print(f"{'OK  ' if ok else 'MISS'} [{s['query']}] -> {queries}")
    print(f"accuracy: {correct}/{len(samples)}")
    return 0 if correct == len(samples) else 1


if __name__ == "__main__":
    raise SystemExit(main())
