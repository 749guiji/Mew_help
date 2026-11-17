"""Tests for the online retrieval path (ch03 Task 6)."""

from app.kb.chunker import Chunk
from app.kb.ingest import insert_chunks, vectorize_pending
from app.kb.milvus_store import MilvusStore
from app.kb.retrieval import retrieve


class HitEmbedder:
    dim = 1024

    def embed_texts(self, texts):
        return [[0.1] * self.dim for _ in texts]  # 一切文本同向量 → 恒命中


class MissEmbedder:
    dim = 1024

    def embed_texts(self, texts):
        return [[-0.1] * self.dim for _ in texts]  # 与存储向量相反 → 恒未命中


def _chunk():
    return Chunk(
        category="售后",
        questions="运费说明",
        answer="满99包邮",
        section_path=None,
        content_type="policy",
        is_key_clause=False,
    )


def test_retrieve_hit_returns_answer(session, tmp_path):
    store = MilvusStore(str(tmp_path / "kv.db"))
    store.ensure_collection()
    insert_chunks(session, [_chunk()])
    vectorize_pending(session, store, HitEmbedder())
    assert retrieve(session, store, HitEmbedder(), "邮费是多少") == "满99包邮"


def test_retrieve_below_threshold_returns_none(session, tmp_path):
    store = MilvusStore(str(tmp_path / "kv.db"))
    store.ensure_collection()
    insert_chunks(session, [_chunk()])
    vectorize_pending(session, store, HitEmbedder())
    assert retrieve(session, store, MissEmbedder(), "邮费是多少") is None


def test_retrieve_empty_collection(session, tmp_path):  # Review Focus #4
    store = MilvusStore(str(tmp_path / "kv.db"))
    store.ensure_collection()
    assert retrieve(session, store, HitEmbedder(), "邮费是多少") is None


def test_query_faq_degrades_to_not_found(monkeypatch):  # Review Focus #5
    import app.tools.db_tools as db_tools
    from app.tools import TOOL_REGISTRY

    monkeypatch.setattr(
        db_tools,
        "default_retrieve",
        lambda q: (_ for _ in ()).throw(RuntimeError("milvus down")),
    )
    assert TOOL_REGISTRY["query_faq"].invoke({"keyword": "邮费是多少"}) == "未查到"
