"""Tests for the dual-write ingest path (ch03 Task 5)."""

from app.db.models import Faq, KnowledgeChunk
from app.kb.chunker import Chunk
from app.kb.ingest import (
    chunk_fingerprint,
    find_duplicate_chunk,
    insert_chunks,
    migrate_faq,
    rebuild,
    vectorize_pending,
)
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
    hits = store.search_dense([0.1] * 1024, top_k=10)
    assert len(hits) == 2


def test_migrate_faq(session, tmp_path):
    session.add(Faq(question="退货政策是什么？", answer="7天无理由", category="售后"))
    session.commit()
    store = MilvusStore(str(tmp_path / "kv.db"))
    store.ensure_collection()
    assert migrate_faq(session, store, FakeEmbedder()) == 1
    row = session.query(KnowledgeChunk).filter(KnowledgeChunk.content_type == "faq").one()
    assert row.questions == "退货政策是什么？" and row.vectorize_status == "done"


def test_vectorize_pending_batches_over_32(session, tmp_path):
    # 70 条 pending 会跨 3 个 batch(32/32/6),验证逐批 embed+upsert 全部落地。
    store = MilvusStore(str(tmp_path / "kv.db"))
    store.ensure_collection()
    chunks = [
        Chunk(
            category="售后",
            questions=f"问题{i}",
            answer=f"答案{i}",
            section_path=None,
            content_type="faq",
            is_key_clause=False,
        )
        for i in range(70)
    ]
    ids = insert_chunks(session, chunks)
    n = vectorize_pending(session, store, FakeEmbedder())
    assert n == 70
    assert all(session.get(KnowledgeChunk, i).vectorize_status == "done" for i in ids)
    hits = store.search_dense([0.1] * 1024, top_k=100)
    assert len(hits) == 70


def test_rebuild_drops_and_revectors_every_row(session, tmp_path):  # Review Focus #4
    store = MilvusStore(str(tmp_path / "kv.db"))
    store.ensure_collection()
    ids = insert_chunks(session, _chunks())
    assert vectorize_pending(session, store, FakeEmbedder()) == 2
    assert all(session.get(KnowledgeChunk, i).vectorize_status == "done" for i in ids)

    # --rebuild: drop + recreate the collection, reset status, re-vectorize.
    assert rebuild(session, store, FakeEmbedder()) == 2
    assert all(session.get(KnowledgeChunk, i).vectorize_status == "done" for i in ids)
    hits = store.search_dense([0.1] * 1024, top_k=10)
    assert len(hits) == 2


def test_build_session_store_embedder_does_not_ensure_collection(monkeypatch, tmp_path):
    """Regression: --rebuild drops the old collection before ensure_collection,
    so the CLI builder must NOT ensure the collection up front — otherwise the
    ch03->ch04 schema guard in ensure_collection raises before the drop."""
    from types import SimpleNamespace

    import app.jobs.ingest_docs as ingest_docs

    ensured = []

    class Store:
        def ensure_collection(self):
            ensured.append(True)

    settings = SimpleNamespace(
        milvus_uri=str(tmp_path / "x.db"),
        milvus_collection="knowledge",
        embedding_dim=1024,
    )
    monkeypatch.setattr(ingest_docs, "MilvusStore", lambda *a, **k: Store())
    monkeypatch.setattr(ingest_docs, "Embedder", lambda *a, **k: object())
    monkeypatch.setattr(ingest_docs, "SessionLocal", lambda: object())
    monkeypatch.setattr(ingest_docs, "get_settings", lambda: settings)

    ingest_docs._build_session_store_embedder()
    assert ensured == []


def test_chunk_fingerprint_same_qa_equal():
    a = chunk_fingerprint("怎么退", "七天无理由")
    b = chunk_fingerprint(" 怎么退？", "七天无理由")
    assert a == b


def test_chunk_fingerprint_diff_answer_diff():
    assert chunk_fingerprint("怎么退", "七天无理由") != chunk_fingerprint("怎么退", "7天包退")


def test_find_duplicate_chunk_returns_existing_id(session):
    from app.db.models import KnowledgeChunk

    session.add(KnowledgeChunk(category="售后", questions="怎么退", answer="七天无理由"))
    session.commit()
    fp = chunk_fingerprint("怎么退", "七天无理由")
    assert find_duplicate_chunk(session, fp) is not None
    assert find_duplicate_chunk(session, chunk_fingerprint("怎么退", "别的内容")) is None
