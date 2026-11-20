"""Tests for the production retrieval entry point (ch04 Task 5)."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.kb.retrieval as retrieval_mod
from app.db import Base
from app.db.models import KnowledgeChunk
from app.kb.hybrid import ScoredChunk
from app.kb.milvus_store import Hit
from app.kb.query_understanding import QueryUnderstanding


class _FakeStore:
    def hybrid_search(self, dense_vector, sparse_text, top_k=1, filter=None):
        return [
            Hit(
                id=1,
                score=0.9,
                text="",
                category="退款",
                content_type="faq",
                section_path=None,
            )
        ]


class _FakeEmbedder:
    def embed_texts(self, texts):
        return [[0.0] * 1024 for _ in texts]


class _FakeReranker:
    def rerank(self, query, docs):
        return [0.7] * len(docs)


class _FakeStructuredRunnable:
    def __call__(self, inp):
        return QueryUnderstanding(standard_question="退货", category="退款")


class _FakeRewriteLLM:
    def with_structured_output(self, model, method=None):
        return _FakeStructuredRunnable()


def _make_session_local():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine)
    with session_local() as s:
        s.add(
            KnowledgeChunk(
                id=1,
                category="退款",
                questions="退货",
                answer="7天无理由",
                section_path=None,
                content_type="faq",
                is_key_clause=0,
                vectorize_status="done",
            )
        )
        s.commit()
    return session_local


def test_default_retrieve_returns_scored_chunks(monkeypatch):
    monkeypatch.setattr(retrieval_mod, "_get_store", lambda: _FakeStore())
    monkeypatch.setattr(retrieval_mod, "_get_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr(retrieval_mod, "_get_reranker", lambda: _FakeReranker())
    monkeypatch.setattr(retrieval_mod, "_get_rewrite_llm", lambda: _FakeRewriteLLM())
    monkeypatch.setattr(retrieval_mod, "SessionLocal", _make_session_local())

    out = retrieval_mod.default_retrieve("怎么退货")

    assert isinstance(out, list)
    assert len(out) == 1
    assert isinstance(out[0], ScoredChunk)
    assert out[0].chunk_id == 1
    assert out[0].answer == "7天无理由"
    assert out[0].score == 0.7


def test_retrieve_multi_dedupes_merges_and_filters(monkeypatch):
    captured_filters = []

    def fake_hybrid_rerank(
        query,
        *,
        session,
        store,
        embedder,
        reranker,
        top_k=10,
        dense_top_k=50,
        sparse_top_k=50,
        filter=None,
        sparse_text=None,
    ):
        captured_filters.append(filter)
        if "政策" in query:
            return [
                ScoredChunk(1, 0.9, None, "退款", "policy", "a", ""),
                ScoredChunk(2, 0.5, None, "退款", "policy", "b", ""),
            ]
        return [
            ScoredChunk(1, 0.4, None, "退款", "policy", "a", ""),
            ScoredChunk(3, 0.8, None, "退款", "policy", "c", ""),
        ]

    monkeypatch.setattr(retrieval_mod, "hybrid_rerank_retrieve", fake_hybrid_rerank)
    monkeypatch.setattr(retrieval_mod, "_get_store", lambda: _FakeStore())
    monkeypatch.setattr(retrieval_mod, "_get_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr(retrieval_mod, "_get_reranker", lambda: _FakeReranker())
    monkeypatch.setattr(retrieval_mod, "SessionLocal", _make_session_local())

    out = retrieval_mod.retrieve_multi(["退款政策", "退货条件"])

    assert [c.chunk_id for c in out] == [1, 3, 2]  # score 降序 + 去重(chunk 1 取 0.9)
    assert out[0].chunk_id == 1 and out[0].score == 0.9
    assert captured_filters == ['content_type == "policy"', 'content_type == "policy"']


def test_query_faq_degrades_to_refused(monkeypatch):  # Review Focus #5
    import json

    import app.tools.db_tools as db_tools
    from app.tools import TOOL_REGISTRY

    monkeypatch.setattr(
        db_tools,
        "default_retrieve",
        lambda q: (_ for _ in ()).throw(RuntimeError("milvus down")),
    )
    out = json.loads(TOOL_REGISTRY["query_faq"].invoke({"keyword": "邮费是多少"}))
    assert out == {"refused": True, "evidence": "", "citations": []}
