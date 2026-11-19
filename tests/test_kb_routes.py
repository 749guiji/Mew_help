"""Tests for the /kb admin page backend + /api/kb endpoints."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.api.routes.kb as kb
from app.db import Base
from app.db.models import KnowledgeChunk
from app.kb.hybrid import ScoredChunk
from app.main import create_app


class FakeStore:
    def __init__(self):
        self.upserted = []

    def upsert(self, rows):
        self.upserted.extend(rows)


class FakeEmbedder:
    dim = 1024

    def embed_texts(self, texts):
        return [[0.1] * self.dim for _ in texts]


@pytest.fixture
def kb_env(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'kb.db'}")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)

    store = FakeStore()
    monkeypatch.setattr(kb, "SessionLocal", SessionLocal)
    monkeypatch.setattr(kb, "_get_store", lambda: store)
    monkeypatch.setattr(kb, "Embedder", lambda *a, **k: FakeEmbedder())
    return store, SessionLocal


def _client():
    return TestClient(create_app())


def test_materials_lists_markdown(kb_env):
    r = _client().get("/api/kb/materials")
    assert r.status_code == 200
    names = {item["name"] for item in r.json()}
    assert "return-policy.md" in names
    assert all({"name", "content_type", "chars"} <= set(item) for item in r.json())


def test_preview_chunks_file(kb_env):
    r = _client().get("/api/kb/preview", params={"path": "return-policy.md"})
    assert r.status_code == 200
    chunks = r.json()
    assert chunks
    assert all(
        {"category", "questions", "answer", "section_path", "content_type", "is_key_clause"}
        <= set(c)
        for c in chunks
    )


def test_preview_rejects_path_traversal(kb_env):
    r = _client().get("/api/kb/preview", params={"path": "../.env"})
    assert r.status_code == 404


def test_add_chunk_inserts_pending_row(kb_env):
    store, SessionLocal = kb_env
    r = _client().post(
        "/api/kb/chunks",
        json={"content_type": "faq", "category": "售后", "questions": "怎么退", "answer": "七天无理由"},
    )
    assert r.status_code == 200
    chunk_id = r.json()["id"]
    with SessionLocal() as s:
        row = s.get(KnowledgeChunk, chunk_id)
        assert row is not None
        assert row.vectorize_status == "pending"
        assert row.answer == "七天无理由"


def test_add_chunk_rejects_blank_answer(kb_env):
    r = _client().post(
        "/api/kb/chunks",
        json={"content_type": "faq", "category": "售后", "questions": "怎么退", "answer": "   "},
    )
    assert r.status_code == 422


def test_vectorize_reuses_cached_store(kb_env):
    store, SessionLocal = kb_env
    # seed one pending chunk
    with SessionLocal() as s:
        s.add(KnowledgeChunk(
            category="售后", questions="怎么退", answer="七天无理由",
            content_type="faq", vectorize_status="pending",
        ))
        s.commit()
    r = _client().post("/api/kb/vectorize")
    assert r.status_code == 200
    assert r.json()["vectorized"] == 1
    assert len(store.upserted) == 1  # upsert went through the cached store


def test_search_returns_retrieved_chunks(kb_env, monkeypatch):
    fake = [
        ScoredChunk(chunk_id=7, score=0.9, section_path="退货政策", category="售后",
                    content_type="policy", answer="七天无理由", questions="怎么退"),
    ]
    monkeypatch.setattr(kb, "hybrid_rerank_retrieve", lambda query, **kw: fake)
    r = _client().post("/api/kb/search", json={"query": "怎么退"})
    assert r.status_code == 200
    body = r.json()
    assert body[0]["chunk_id"] == 7
    assert body[0]["section_path"] == "退货政策"
    assert body[0]["score"] == 0.9


def test_search_rejects_bad_strategy(kb_env):
    r = _client().post("/api/kb/search", json={"query": "怎么退", "strategy": "bogus"})
    assert r.status_code == 422


def test_chunks_dedup_returns_409(kb_env):
    payload = {"content_type": "faq", "category": "售后", "questions": "怎么退", "answer": "七天无理由"}
    assert _client().post("/api/kb/chunks", json=payload).status_code == 200
    second = _client().post("/api/kb/chunks", json=payload)
    assert second.status_code == 409
    assert second.json()["duplicate"] is True


def test_overview_consistent_true_when_counts_match(kb_env, monkeypatch):
    from types import SimpleNamespace

    _, SessionLocal = kb_env
    with SessionLocal() as s:
        for i in range(2):
            s.add(KnowledgeChunk(category="c", questions=f"q{i}", answer=f"a{i}",
                                 content_type="faq", vectorize_status="done"))
        s.add(KnowledgeChunk(category="c", questions="q3", answer="a3",
                             content_type="faq", vectorize_status="pending"))
        s.commit()

    monkeypatch.setattr(kb, "_get_store", lambda: SimpleNamespace(count=lambda: 2))

    r = _client().get("/api/kb/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["chunks"] == 3
    assert body["pending"] == 1
    assert body["done"] == 2
    assert body["milvus"] == 2
    assert body["consistent"] is True
