from app.kb.milvus_store import MilvusStore


def test_upsert_search_and_idempotency(tmp_path):
    store = MilvusStore(str(tmp_path / "kv.db"), collection="knowledge", dim=2)
    store.ensure_collection(); store.ensure_collection()   # 幂等不炸
    store.upsert([(1, [1.0, 0.0])])
    hits = store.search([1.0, 0.0], top_k=1)
    assert hits and hits[0][0] == 1 and hits[0][1] > 0.9
    store.upsert([(1, [0.0, 1.0])])   # 同 id 覆盖,不新增
    hits2 = store.search([0.0, 1.0], top_k=5)
    assert [h[0] for h in hits2] == [1]   # 仍只有一条


def test_search_empty_collection(tmp_path):
    store = MilvusStore(str(tmp_path / "empty.db"), dim=2)
    store.ensure_collection()
    assert store.search([1.0, 0.0], top_k=1) == []   # Review Focus #4
