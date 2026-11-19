"""Tests for the eval harness wiring (failure handling in the segments)."""

from evaluation import run_eval


def test_run_answer_coverage_skips_hybrid_rerank_on_runtime_failure(monkeypatch):
    """A runtime reranker failure must skip hybrid_rerank, not crash the run."""
    monkeypatch.setattr("app.core.llm.get_llm", lambda settings: object())

    def fake_run_strategy(name, items, **kwargs):
        if name == "hybrid_rerank":
            raise RuntimeError("cross-encoder load failed")
        return [[] for _ in items]

    monkeypatch.setattr(run_eval, "run_strategy", fake_run_strategy)

    items = [
        {
            "id": "A01",
            "bucket": "A_policy",
            "query": "q",
            "expect_points": ["p1"],
            "should_refuse": False,
        }
    ]

    rc = run_eval.run_answer_coverage(
        items,
        settings=object(),
        session=None,
        store=None,
        embedder=None,
        reranker=None,
        strategy_names=["hybrid_rerank"],
        top_k=10,
        hybrid_rerank_ready=True,
    )

    assert rc == 0
