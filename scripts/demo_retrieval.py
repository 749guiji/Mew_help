"""Demo: query the knowledge base via vector retrieval (acceptance #1).

Usage::

    python scripts/demo_retrieval.py "邮费是多少"

Prints the best-matching chunk's answer, or "未查到" if nothing qualifies.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when run as a plain script (not -m).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.kb.retrieval import default_retrieve  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print("usage: python scripts/demo_retrieval.py <question>")
        return 2
    query = args[0]
    answer = default_retrieve(query)
    print(f"Q: {query}")
    print(f"A: {answer if answer else '未查到'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
