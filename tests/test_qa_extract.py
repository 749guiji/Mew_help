"""Tests for the historical-conversation QA mining pipeline (ch03 Task 7)."""

from app.db.models import KnowledgeChunk, QaExtractionStaging
from app.kb.milvus_store import MilvusStore
from app.kb.qa_extract import QA, dedupe, extract_qa, mine_conversations


class FakeChain:
    def __init__(self, pairs):
        self._pairs = pairs

    def invoke(self, inp):
        return type("R", (), {"pairs": self._pairs})()


class FakeEmbedder:
    def __init__(self, dim=1024):
        self.dim = dim

    def embed_texts(self, texts):
        return [[0.1] * self.dim for _ in texts]


def test_extract_qa_returns_pairs():
    chain = FakeChain([type("P", (), {"question": "q1", "answer": "a1"})()])
    out = extract_qa(chain, "任意文本")
    assert out == [QA("q1", "a1")]


def test_dedupe_by_normalized_question():
    qas = [QA("邮费多少？", "a"), QA(" 邮费多少 ", "a"), QA("退货？", "b")]
    assert dedupe(qas) == [QA("邮费多少？", "a"), QA("退货？", "b")]


def test_dedupe_removes_internal_punctuation():
    from app.kb.qa_extract import normalize

    assert normalize("请问，运费怎么收") == normalize("请问运费怎么收")
    assert dedupe([QA("运费，怎么收", "a"), QA("运费怎么收", "b")]) == [
        QA("运费，怎么收", "a")
    ]


def test_mine_conversations_end_to_end(session, tmp_path):
    src = tmp_path / "conv.jsonl"
    src.write_text('{"text": "用户:运费怎么收 客服:满99包邮"}\n', encoding="utf-8")
    store = MilvusStore(str(tmp_path / "kv.db"))
    store.ensure_collection()
    chain = FakeChain(
        [type("P", (), {"question": "运费怎么收", "answer": "满99包邮"})()]
    )
    n = mine_conversations(session, store, FakeEmbedder(), chain, str(src))
    assert n == 1
    assert (
        session.query(QaExtractionStaging)
        .filter(QaExtractionStaging.status == "kept")
        .count()
        == 1
    )
    row = session.query(KnowledgeChunk).filter(KnowledgeChunk.content_type == "mined").one()
    assert row.vectorize_status == "done" and "满99包邮" in row.answer
