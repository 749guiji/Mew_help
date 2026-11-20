"""Production retrieval entry point: query understanding -> hybrid rerank.

This is the read half of the knowledge-base pipeline. A query goes through
query rewrite + synonym expansion (see :mod:`app.kb.hybrid`) and is answered by
the hybrid+rerank strategy, which searches Milvus and backfills the
source-of-truth ``knowledge_chunks`` row from MySQL.

Heavy components (Milvus client, embedder, reranker, rewrite LLM) are cached
process-wide so repeated queries do not re-open Milvus Lite or reload models.
"""

from __future__ import annotations

import functools

from app.core.config import get_settings
from app.core.llm import get_llm
from app.db import SessionLocal
from app.kb.embedding import Embedder
from app.kb.hybrid import ScoredChunk, hybrid_rerank_retrieve, retrieve_with_understanding
from app.kb.milvus_store import MilvusStore
from app.kb.query_understanding import expand_synonyms
from app.kb.reranker import Reranker


@functools.lru_cache(maxsize=1)
def _get_store() -> MilvusStore:
    """Build (and cache) the configured Milvus store for the process lifetime.

    ``MilvusClient`` holds an open handle per instance and Milvus Lite is
    single-process per data dir, so constructing a fresh client per query
    accumulates handles and can trip the lock; cache one shared instance.
    """
    settings = get_settings()
    store = MilvusStore(
        settings.milvus_uri,
        collection=settings.milvus_collection,
        dim=settings.embedding_dim,
    )
    store.ensure_collection()
    return store


@functools.lru_cache(maxsize=1)
def _get_embedder() -> Embedder:
    """Build (and cache) the configured embedder."""
    settings = get_settings()
    return Embedder(dim=settings.embedding_dim)


@functools.lru_cache(maxsize=1)
def _get_reranker() -> Reranker:
    """Build (and cache) the configured cross-encoder reranker."""
    settings = get_settings()
    return Reranker(
        model_name=settings.rerank_model,
        use_fp16=settings.rerank_use_fp16,
    )


@functools.lru_cache(maxsize=1)
def _get_rewrite_llm():
    """Build (and cache) the LLM used for query rewriting."""
    return get_llm()


def default_retrieve(query: str) -> list[ScoredChunk]:
    """Integration entry point: build cached components and run the pipeline.

    Used by the chat tool path (``query_faq``). The store is cached
    process-wide (see :func:`_get_store`); the collection is ensured at build
    time so a fresh data dir still degrades gracefully to an empty list instead
    of raising.
    """
    store = _get_store()
    embedder = _get_embedder()
    reranker = _get_reranker()
    rewrite_llm = _get_rewrite_llm()

    with SessionLocal() as session:
        return retrieve_with_understanding(
            query,
            session=session,
            store=store,
            embedder=embedder,
            reranker=reranker,
            rewrite_llm=rewrite_llm,
        )


def retrieve_multi(
    queries: list[str], *, content_type: str = "policy"
) -> list[ScoredChunk]:
    """对多条(已扩写的)查询并行检索、按 ``chunk_id`` 去重合并。

    用于退款/售后子流程:扩写产出的查询已是标准问法,直接走 ``hybrid_rerank_retrieve``
    (跳过逐条 ``rewrite_query``,省 LLM 调用),强制 ``content_type`` 过滤政策条款。
    ``score`` 取同一 chunk 在各查询里的最高分,返回按 score 降序。
    """
    store = _get_store()
    embedder = _get_embedder()
    reranker = _get_reranker()

    with SessionLocal() as session:
        merged: dict[int, ScoredChunk] = {}
        for q in queries:
            chunks = hybrid_rerank_retrieve(
                q,
                session=session,
                store=store,
                embedder=embedder,
                reranker=reranker,
                filter=f'content_type == "{content_type}"',
                sparse_text=expand_synonyms(q),
            )
            for c in chunks:
                if c.chunk_id not in merged or c.score > merged[c.chunk_id].score:
                    merged[c.chunk_id] = c
        return sorted(merged.values(), key=lambda c: c.score, reverse=True)
