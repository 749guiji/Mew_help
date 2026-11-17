"""Tests for the dual-write ingest path (ch03 Task 5)."""

from app.db.models import Faq, KnowledgeChunk
from app.kb.chunker import Chunk
from app.kb.ingest import insert_chunks, migrate_faq, vectorize_pending
from app.kb.milvus_store import MilvusStore


class FakeEmbedder:
    def __init__(self, dim=1024):
        self.dim = dim

    def embed_texts(self, texts):
        return [[0.1] * self.dim for _ in texts]


def _chunks():
    return [
        Chunk(
            category="售后",
            questions="退货政策是什么",
            answer="7天无理由",
            section_path="退货政策",
            content_type="faq",
            is_key_clause=False,
        ),
        Chunk(
            category="售后",
            questions="运费怎么收",
            answer="满99包邮",
            section_path="运费说明",
            content_type="faq",
            is_key_clause=False,
        ),
    ]


def test_insert_sets_pending_and_pointers(session):
    ids = insert_chunks(session, _chunks())
    assert len(ids) == 2
    a, b = session.get(KnowledgeChunk, ids[0]), session.get(KnowledgeChunk, ids[1])
    assert a.vectorize_status == "pending" and a.next_chunk_id == ids[1] and b.prev_chunk_id == ids[0]


def test_vectorize_pending_backfills(session, tmp_path):
    store = MilvusStore(str(tmp_path / "kv.db"))
    store.ensure_collection()
    ids = insert_chunks(session, _chunks())
    n = vectorize_pending(session, store, FakeEmbedder())
    assert n == 2
    assert all(session.get(KnowledgeChunk, i).vectorize_status == "done" for i in ids)


def test_interrupt_then_rerun_picks_up_pending(session, tmp_path):  # 验收 #2
    store = MilvusStore(str(tmp_path / "kv.db"))
    store.ensure_collection()
    ids = insert_chunks(session, _chunks())  # 模拟「只写 MySQL,没向量化就挂了」
    assert all(session.get(KnowledgeChunk, i).vectorize_status == "pending" for i in ids)
    assert vectorize_pending(session, store, FakeEmbedder()) == 2  # 重跑补齐
    assert vectorize_pending(session, store, FakeEmbedder()) == 0  # 再跑无事可做(幂等)
    assert all(session.get(KnowledgeChunk, i).vectorize_status == "done" for i in ids)
    hits = store.search([0.1] * 1024, top_k=10)
    assert len(hits) == 2


def test_migrate_faq(session, tmp_path):
    session.add(Faq(question="退货政策是什么？", answer="7天无理由", category="售后"))
    session.commit()
    store = MilvusStore(str(tmp_path / "kv.db"))
    store.ensure_collection()
    assert migrate_faq(session, store, FakeEmbedder()) == 1
    row = session.query(KnowledgeChunk).filter(KnowledgeChunk.content_type == "faq").one()
    assert row.questions == "退货政策是什么？" and row.vectorize_status == "done"
