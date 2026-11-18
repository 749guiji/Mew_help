"""CLI: build the knowledge base from ``knowledge/*.md`` + the ``faq`` table.

Usage::

    python -m app.jobs.ingest_docs             # full build
    python -m app.jobs.ingest_docs --reconcile # only backfill pending rows

The full build reads every ``knowledge/*.md`` file, derives its ``content_type``
from the file name (``faq`` / ``policy`` / ``manual``), ingests it, then migrates
the ``faq`` table into ``knowledge_chunks``. ``--reconcile`` skips all of that and
only runs ``vectorize_pending``, which is the idempotent rerun that backfills
rows left behind by an interrupted build (acceptance #2).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.core.config import get_settings
from app.db import SessionLocal
from app.db.models import KnowledgeChunk
from app.kb.embedding import Embedder
from app.kb.ingest import ingest_document, migrate_faq, rebuild, vectorize_pending
from app.kb.milvus_store import MilvusStore

_KNOWLEDGE_DIR = Path("knowledge")


def _content_type_from_path(path: Path) -> str:
    name = path.name.lower()
    if "faq" in name:
        return "faq"
    if "policy" in name or "退货" in name:
        return "policy"
    return "manual"


def _build_session_store_embedder():
    settings = get_settings()
    session = SessionLocal()
    store = MilvusStore(
        settings.milvus_uri,
        collection=settings.milvus_collection,
        dim=settings.embedding_dim,
    )
    embedder = Embedder(dim=settings.embedding_dim)
    return session, store, embedder, settings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the RAG knowledge base.")
    parser.add_argument(
        "--reconcile",
        action="store_true",
        help="Only vectorize pending chunks (idempotent rerun; no re-read/re-insert).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow a full build even when knowledge_chunks already has rows.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop + recreate the Milvus collection and re-vectorize every chunk.",
    )
    args = parser.parse_args(argv)

    session, store, embedder, settings = _build_session_store_embedder()
    try:
        if args.rebuild:
            n = rebuild(session, store, embedder)
            print(f"rebuild: re-vectorized {n} chunks into a fresh collection")
            return 0

        # Non-rebuild paths need the collection to already exist with the new
        # schema. If it's the old ch03 layout, ensure_collection raises the
        # clear "--rebuild" hint instead of failing later at search time.
        store.ensure_collection()

        if args.reconcile:
            n = vectorize_pending(session, store, embedder)
            print(f"reconcile: vectorized {n} pending chunks")
            return 0

        existing = session.query(KnowledgeChunk).count()
        if existing > 0 and not args.force:
            print(
                f"已有 {existing} 块,全量 build 会重复;"
                "用 --reconcile 增量,或 --force 强制重建"
            )
            return 1

        total = 0
        for path in sorted(_KNOWLEDGE_DIR.glob("*.md")):
            content_type = _content_type_from_path(path)
            ids = ingest_document(
                session,
                store,
                embedder,
                path.read_text(encoding="utf-8"),
                content_type,
                max_chars=settings.chunk_max_chars,
                overlap=settings.chunk_overlap_chars,
                source=str(path),
            )
            total += len(ids)
            print(f"ingested {path} ({content_type}): {len(ids)} chunks")

        migrated = migrate_faq(session, store, embedder)
        total += migrated
        print(f"migrated faq: {migrated} chunks")
        print(f"knowledge base built: {total} chunks")
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
