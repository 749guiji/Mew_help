"""Tests for ch04 hybrid retrieval orchestration (four strategies + understanding)."""

from app.db.models import KnowledgeChunk
from app.kb.hybrid import (
    dense_retrieve,
    hybrid_rerank_retrieve,
    hybrid_retrieve,
    retrieve_with_understanding,
    sparse_retrieve,
)
from app.kb.milvus_store import Hit
from app.kb.query_understanding import QueryUnderstanding, expand_synonyms


class FakeStore:
    """Canned ``Hit`` lists per method, recording every call for assertions."""

    def __init__(self, dense=None, sparse=None, hybrid=None):
        self.dense_hits = dense or []
        self.sparse_hits = sparse or []
        self.hybrid_hits = hybrid or []
        self.calls = []

    def search_dense(self, vector, top_k=1, filter=None):
        self.calls.append(("search_dense", vector, top_k, filter))
        return self.dense_hits[:top_k]

    def search_sparse(self, text, top_k=1, filter=None):
        self.calls.append(("search_sparse", text, top_k, filter))
        return self.sparse_hits[:top_k]

    def hybrid_search(self, dense_vector, sparse_text, top_k=1, filter=None):
        self.calls.append(("hybrid_search", dense_vector, sparse_text, top_k, filter))
        return self.hybrid_hits[:top_k]


class FakeEmbedder:
    def __init__(self, vectors=None):
        self.vectors = vectors or {}
        self.calls = []

    def embed_texts(self, texts):
        self.calls.extend(texts)
        return [self.vectors.get(t, [0.0]) for t in texts]


class FakeReranker:
    def __init__(self, scores):
        self.scores = list(scores)
        self.last_query = None
        self.last_docs = None

    def rerank(self, query, docs):
        self.last_query = query
        self.last_docs = list(docs)
        return self.scores


class FakeStructuredRunnable:
    def __init__(self, result):
        self._result = result

    def __call__(self, inp):
        return self._result


class FakeRewriteLLM:
    def __init__(self, result):
        self._result = result

    def with_structured_output(self, model, method=None):
        return FakeStructuredRunnable(self._result)


def _hit(chunk_id, score):
    return Hit(
        id=chunk_id,
        score=score,
        text="",
        category="",
        content_type="",
        section_path=None,
    )


def _seed(session):
    session.add_all(
        [
            KnowledgeChunk(
                id=1,
                category="退款",
                questions="怎么退款",
                answer="联系客服退款",
                section_path="退款政策",
                content_type="faq",
                is_key_clause=0,
                vectorize_status="done",
            ),
            KnowledgeChunk(
                id=2,
                category="运费说明",
                questions="运费怎么收",
                answer="满99包邮",
                section_path="运费",
                content_type="faq",
                is_key_clause=0,
                vectorize_status="done",
            ),
            KnowledgeChunk(
                id=3,
                category="退款",
                questions="如何换货",
                answer="支持换货",
                section_path=None,
                content_type="policy",
                is_key_clause=1,
                vectorize_status="done",
            ),
        ]
    )
    session.commit()


def test_dense_retrieve_returns_hits_in_canned_order(session):
    _seed(session)
    store = FakeStore(dense=[_hit(2, 0.9), _hit(1, 0.8)])
    embedder = FakeEmbedder({"退款": [1.0, 0.0]})

    out = dense_retrieve("退款", session=session, store=store, embedder=embedder)

    assert [c.chunk_id for c in out] == [2, 1]
    assert out[0].answer == "满99包邮"
    assert out[0].category == "运费说明"
    assert store.calls[0][0] == "search_dense"
    assert store.calls[0][1] == [1.0, 0.0]
    assert store.calls[0][3] is None


def test_sparse_retrieve_passes_query_and_returns_order(session):
    _seed(session)
    store = FakeStore(sparse=[_hit(1, 0.7), _hit(3, 0.5)])

    out = sparse_retrieve("退款", session=session, store=store)

    assert [c.chunk_id for c in out] == [1, 3]
    assert store.calls[0][0] == "search_sparse"
    assert store.calls[0][1] == "退款"


def test_hybrid_retrieve_uses_rrf_store_call(session):
    _seed(session)
    store = FakeStore(hybrid=[_hit(3, 0.95), _hit(2, 0.6)])
    embedder = FakeEmbedder({"退款": [2.0]})

    out = hybrid_retrieve("退款", session=session, store=store, embedder=embedder)

    assert [c.chunk_id for c in out] == [3, 2]
    assert out[0].section_path is None
    assert store.calls[0][0] == "hybrid_search"
    assert store.calls[0][1] == [2.0]
    assert store.calls[0][2] == "退款"
    assert store.calls[0][3] == 10


def test_hybrid_rerank_retrieve_reorders_and_truncates(session):
    _seed(session)
    store = FakeStore(hybrid=[_hit(1, 0.5), _hit(2, 0.4), _hit(3, 0.3)])
    embedder = FakeEmbedder({"q": [3.0]})
    reranker = FakeReranker([0.2, 0.9, 0.5])

    out = hybrid_rerank_retrieve(
        "q", session=session, store=store, embedder=embedder, reranker=reranker
    )

    assert [c.chunk_id for c in out] == [2, 3, 1]
    assert reranker.last_query == "q"
    assert reranker.last_docs == ["联系客服退款", "满99包邮", "支持换货"]
    assert store.calls[0][3] == 50  # dense/sparse top-k widen the candidate pool

    top1 = hybrid_rerank_retrieve(
        "q",
        session=session,
        store=store,
        embedder=embedder,
        reranker=reranker,
        top_k=1,
    )
    assert [c.chunk_id for c in top1] == [2]


def test_retrieve_with_understanding_uses_expanded_sparse_and_standard_dense(session):
    _seed(session)
    store = FakeStore(hybrid=[_hit(1, 0.9)])
    embedder = FakeEmbedder({"退款怎么处理": [9.0]})
    reranker = FakeReranker([0.8])
    llm = FakeRewriteLLM(
        QueryUnderstanding(standard_question="退款怎么处理", category="退款")
    )

    out = retrieve_with_understanding(
        "咋退款",
        session=session,
        store=store,
        embedder=embedder,
        reranker=reranker,
        rewrite_llm=llm,
    )

    assert [c.chunk_id for c in out] == [1]
    call = store.calls[0]
    assert call[0] == "hybrid_search"
    assert call[1] == [9.0]  # dense vector from standard_question
    assert call[2] == expand_synonyms("退款怎么处理")  # BM25 uses expanded text
    assert "退钱" in call[2]  # synonym actually appended
    assert call[4] == 'category == "退款"'
    assert embedder.calls == ["退款怎么处理"]
    assert reranker.last_query == "退款怎么处理"


def test_filter_passes_through_to_store(session):
    _seed(session)
    store = FakeStore(dense=[_hit(1, 0.9)])
    embedder = FakeEmbedder({"q": [1.0]})

    dense_retrieve(
        "q", session=session, store=store, embedder=embedder, filter='category == "x"'
    )
    sparse_retrieve("q", session=session, store=store, filter='category == "x"')
    hybrid_retrieve(
        "q", session=session, store=store, embedder=embedder, filter='category == "x"'
    )

    assert store.calls[0][3] == 'category == "x"'  # search_dense
    assert store.calls[1][3] == 'category == "x"'  # search_sparse
    assert store.calls[2][4] == 'category == "x"'  # hybrid_search (filter is index 4)


def test_retrieve_with_understanding_filter_precedence(session):
    _seed(session)
    store = FakeStore(hybrid=[_hit(1, 0.9)])
    embedder = FakeEmbedder({"退款": [1.0]})
    reranker = FakeReranker([0.8])

    # A caller-supplied filter overrides the inferred category filter.
    llm = FakeRewriteLLM(QueryUnderstanding(standard_question="退款", category="退款"))
    retrieve_with_understanding(
        "q",
        session=session,
        store=store,
        embedder=embedder,
        reranker=reranker,
        rewrite_llm=llm,
        filter='category == "运费说明"',
    )
    assert store.calls[0][4] == 'category == "运费说明"'

    # No category -> no filter.
    llm2 = FakeRewriteLLM(QueryUnderstanding(standard_question="退款", category=None))
    retrieve_with_understanding(
        "q",
        session=session,
        store=store,
        embedder=embedder,
        reranker=reranker,
        rewrite_llm=llm2,
    )
    assert store.calls[1][4] is None
