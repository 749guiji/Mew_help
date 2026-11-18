"""Tests for the ch04 BM25/hybrid Milvus schema + search methods."""

import pytest

from app.kb.milvus_store import Hit, MilvusRecord, MilvusStore


def _records():
    return [
        MilvusRecord(
            id=1,
            text="售后\n退货政策是什么\n7天无理由",
            dense=[1.0, 0.0],
            category="退货政策",
            content_type="faq",
            section_path="退货政策",
            is_key_clause=False,
        ),
        MilvusRecord(
            id=2,
            text="物流\n运费怎么收\n满99包邮",
            dense=[0.0, 1.0],
            category="运费说明",
            content_type="faq",
            section_path="运费说明",
            is_key_clause=False,
        ),
    ]


def test_ensure_collection_schema_has_sparse_and_bm25_function(tmp_path):
    from pymilvus import DataType

    store = MilvusStore(str(tmp_path / "kv.db"), dim=2)
    store.ensure_collection()
    store.ensure_collection()  # 幂等不炸

    desc = store.client.describe_collection("knowledge")
    fields = {f["name"]: f["type"] for f in desc["fields"]}
    assert fields["sparse"] == DataType.SPARSE_FLOAT_VECTOR
    assert fields["dense"] == DataType.FLOAT_VECTOR
    assert fields["text"] == DataType.VARCHAR
    assert fields["category"] == DataType.VARCHAR
    assert fields["section_path"] == DataType.VARCHAR
    assert fields["is_key_clause"] == DataType.BOOL

    functions = {f["name"] for f in desc.get("functions", [])}
    assert "text_bm25" in functions


def test_ensure_collection_rejects_old_ch03_schema(tmp_path):
    """A persisted ch03 collection (id + vector, no sparse/text) must fail fast."""
    from pymilvus import DataType

    store = MilvusStore(str(tmp_path / "old.db"), dim=2)
    schema = store.client.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=2)
    index_params = store.client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
    store.client.create_collection("knowledge", schema=schema, index_params=index_params)

    with pytest.raises(RuntimeError, match="ingest_docs --rebuild"):
        store.ensure_collection()


def test_chinese_analyzer_raises(tmp_path):
    from pymilvus import DataType, Function, FunctionType, MilvusClient
    from pymilvus.exceptions import MilvusException

    client = MilvusClient(str(tmp_path / "bad.db"))
    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field(
        "text",
        DataType.VARCHAR,
        max_length=2048,
        enable_analyzer=True,
        analyzer_params={"type": "chinese"},
    )
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=2)
    schema.add_function(
        Function(
            name="text_bm25",
            input_field_names=["text"],
            output_field_names=["sparse"],
            function_type=FunctionType.BM25,
        )
    )
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="dense", index_type="FLAT", metric_type="COSINE")
    index_params.add_index(
        field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25"
    )

    with pytest.raises(MilvusException):
        client.create_collection("bad", schema=schema, index_params=index_params)


def test_search_sparse_dense_and_hybrid(tmp_path):
    store = MilvusStore(str(tmp_path / "kv.db"), dim=2)
    store.ensure_collection()
    store.upsert(_records())

    sparse = store.search_sparse("退货", top_k=5)
    assert sparse and sparse[0].id == 1
    assert isinstance(sparse[0], Hit)

    dense = store.search_dense([1.0, 0.0], top_k=5)
    assert dense and dense[0].id == 1

    hybrid = store.hybrid_search([1.0, 0.0], "退货", top_k=5)
    assert hybrid and hybrid[0].id == 1
    assert {h.id for h in hybrid} >= {1, 2}


def test_filter_returns_only_matching_category(tmp_path):
    store = MilvusStore(str(tmp_path / "kv.db"), dim=2)
    store.ensure_collection()
    store.upsert(_records())

    # [0,1] favours record 2 (运费说明), but the filter keeps only record 1.
    hits = store.search_dense([0.0, 1.0], top_k=5, filter='category == "退货政策"')
    assert hits and all(h.category == "退货政策" for h in hits)
    assert all(h.id != 2 for h in hits)


def test_section_path_none_roundtrips_to_none(tmp_path):
    store = MilvusStore(str(tmp_path / "kv.db"), dim=2)
    store.ensure_collection()
    store.upsert(
        [
            MilvusRecord(
                id=9,
                text="测试\n无章节",
                dense=[1.0, 0.0],
                category="其他",
                content_type="faq",
                section_path=None,
                is_key_clause=False,
            )
        ]
    )
    hits = store.search_dense([1.0, 0.0], top_k=1)
    assert hits[0].section_path is None


def test_upsert_is_idempotent_by_id(tmp_path):
    store = MilvusStore(str(tmp_path / "kv.db"), dim=2)
    store.ensure_collection()
    records = _records()
    store.upsert(records[:1])
    store.upsert([records[0]])  # 同 id 覆盖,不新增
    hits = store.search_dense([1.0, 0.0], top_k=5)
    assert [h.id for h in hits] == [1]


def test_search_empty_collection(tmp_path):
    store = MilvusStore(str(tmp_path / "kv.db"), dim=2)
    store.ensure_collection()
    assert store.search_dense([1.0, 0.0], top_k=1) == []
    assert store.search_sparse("退货", top_k=1) == []
    assert store.hybrid_search([1.0, 0.0], "退货", top_k=1) == []


def test_init_creates_parent_dir(tmp_path):
    store = MilvusStore(str(tmp_path / "nested" / "sub" / "kv.db"), dim=2)
    store.ensure_collection()
    store.upsert(_records()[:1])
    assert store.search_dense([1.0, 0.0], top_k=1)[0].id == 1
