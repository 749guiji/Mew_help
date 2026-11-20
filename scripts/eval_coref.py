"""Eval coreference resolution + rewrite against annotated multi-turn samples.

Usage: PYTHONPATH=. .venv/Scripts/python.exe scripts/eval_coref.py
"""

import json
from pathlib import Path

from app.core.config import get_settings
from app.core.llm import get_llm
from app.graph.nodes import resolve_coref


def main() -> int:
    samples = json.loads(
        Path("tests/data/coref_samples.json").read_text(encoding="utf-8")
    )
    llm = get_llm(get_settings())
    correct = 0
    for s in samples:
        got = resolve_coref(llm, s["query"], s.get("history", []))
        ok = (got == s["expected"]) or all(
            t in got for t in s.get("must_contain", [])
        )
        correct += int(ok)
        print(
            f"{'OK  ' if ok else 'MISS'} [{s['query']}] -> {got} "
            f"(expect {s['expected']})"
        )
    print(f"accuracy: {correct}/{len(samples)}")
    return 0 if correct == len(samples) else 1


if __name__ == "__main__":
    raise SystemExit(main())
