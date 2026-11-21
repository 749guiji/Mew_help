"""Tests for ch09 self_check node (post-hoc log, no answer gating)."""

import app.db as db
import app.services.generation_quality as gq
from langchain_core.messages import AIMessage, HumanMessage

from app.graph.nodes import self_check_node
from app.kb.hybrid import ScoredChunk
from app.services.generation_quality import SelfCheckResult


def _chunk(cid, score):
    return ScoredChunk(
        chunk_id=cid,
        score=score,
        section_path="s",
        category="c",
        content_type="faq",
        answer=f"answer{cid}",
        questions="",
    )


class _FakeCtx:
    def __enter__(self):
        return object()

    def __exit__(self, *a):
        return False


def test_insufficient_logs_to_pool(monkeypatch):
    calls = []
    monkeypatch.setattr(
        gq, "self_check", lambda llm, q, e: SelfCheckResult(sufficient=False, reason="证据不足")
    )
    monkeypatch.setattr(gq, "log_low_confidence", lambda session, **kw: calls.append(kw))
    monkeypatch.setattr(db, "SessionLocal", lambda: _FakeCtx())

    state = {
        "messages": [HumanMessage(content="运费怎么算"), AIMessage(content="不太确定")],
        "resolved_query": "运费怎么算",
        "retrieval_chunks": [_chunk(1, 0.9)],
        "conversation_id": 1,
    }
    result = self_check_node(state)

    assert result["visited"] == ["self_check"]
    assert len(calls) == 1 and calls[0]["source"] == "self_check"
    assert "messages" not in result  # 不拦答案、不追加消息


def test_sufficient_does_not_log(monkeypatch):
    calls = []
    monkeypatch.setattr(
        gq, "self_check", lambda llm, q, e: SelfCheckResult(sufficient=True, reason="")
    )
    monkeypatch.setattr(gq, "log_low_confidence", lambda session, **kw: calls.append(kw))
    monkeypatch.setattr(db, "SessionLocal", lambda: _FakeCtx())

    state = {
        "messages": [HumanMessage(content="x"), AIMessage(content="答案")],
        "resolved_query": "x",
        "retrieval_chunks": [_chunk(1, 0.9)],
        "conversation_id": 1,
    }
    result = self_check_node(state)

    assert result["visited"] == ["self_check"]
    assert calls == []
