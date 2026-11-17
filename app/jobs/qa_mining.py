"""CLI: mine QA pairs from historical customer-service conversations.

Usage::

    python -m app.jobs.qa_mining [--source knowledge/conversations/sample.jsonl] [--batch-size 10]

Reads a jsonl file of conversation texts, runs the QA-extraction chain over it
in batches, deduplicates the extracted pairs, and ingests the kept ones into
``knowledge_chunks`` + Milvus.
"""

from __future__ import annotations

import argparse

from app.core.config import get_settings
from app.core.llm import get_llm
from app.db import SessionLocal
from app.kb.embedding import Embedder
from app.kb.milvus_store import MilvusStore
from app.kb.qa_extract import build_qa_extraction_chain, mine_conversations


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Mine QA pairs from historical conversations."
    )
    parser.add_argument(
        "--source",
        default="knowledge/conversations/sample.jsonl",
        help="Path to the jsonl conversation file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of conversations per extraction batch.",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    session = SessionLocal()
    store = MilvusStore(
        settings.milvus_uri,
        collection=settings.milvus_collection,
        dim=settings.embedding_dim,
    )
    store.ensure_collection()
    embedder = Embedder(dim=settings.embedding_dim)
    chain = build_qa_extraction_chain(get_llm(settings))

    try:
        n = mine_conversations(
            session, store, embedder, chain, args.source, batch_size=args.batch_size
        )
        print(f"qa mining: {n} kept chunks from {args.source}")
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
