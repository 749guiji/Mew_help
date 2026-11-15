from langchain_core.messages import HumanMessage
from app.services.history import count_tokens, trim_history

def test_trim_history_keeps_last_n():
    msgs = [HumanMessage(f"m{i}") for i in range(5)]
    out = trim_history(msgs, max_tokens=10_000, keep_last=3)
    assert [m.content for m in out] == ["m2", "m3", "m4"]

def test_trim_history_token_budget():
    msgs = [HumanMessage("x" * 1000), HumanMessage("x" * 1000), HumanMessage("ok")]
    out = trim_history(msgs, max_tokens=10, keep_last=10)
    assert [m.content for m in out] == ["ok"]

def test_trim_history_keeps_recent_within_budget():
    msgs = [HumanMessage("x" * 1000), HumanMessage("keep me")]
    out = trim_history(msgs, max_tokens=10, keep_last=10)
    assert [m.content for m in out] == ["keep me"]

def test_count_tokens_positive_int():
    n = count_tokens([HumanMessage("你好,世界,这个订单什么时候发货")])
    assert isinstance(n, int) and n > 0

def test_trim_history_keep_last_zero_returns_empty():
    msgs = [HumanMessage("a"), HumanMessage("b")]
    out = trim_history(msgs, max_tokens=10_000, keep_last=0)
    assert out == []
