"""Tests for ch09 eval pipeline (metrics + eval_runs)."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.jobs.eval_run as evr
from app.db import Base
from app.db.models import EvalRun
from app.kb.hybrid import ScoredChunk


def _chunk(cid, answer):
    return ScoredChunk(
        chunk_id=cid, score=0.9, section_path="s", category="c",
        content_type="faq", answer=answer, questions="",
    )


def test_compute_retrieval_metrics():
    items = [
        {
            "query": "q1",
            "ground_truth_chunk_ids": [1, 2],
            "expect_points": ["满99包邮"],
        }
    ]
    def retrieve_fn(q):
        return [_chunk(1, "满99包邮"), _chunk(3, "无关")]

    m = evr.compute_retrieval_metrics(items, retrieve_fn)
    assert m["recall_at_k"] == pytest.approx(0.5, abs=1e-3)  # 2 个 gt 命中 1 个
    assert m["mrr"] == pytest.approx(1.0, abs=1e-3)
    assert m["evidence_coverage"] == pytest.approx(1.0, abs=1e-3)


@pytest.fixture
def sqlite_session_local(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SL = sessionmaker(bind=engine)
    monkeypatch.setattr(evr, "SessionLocal", SL)
    return SL


def test_run_records_eval_run(sqlite_session_local, monkeypatch):
    monkeypatch.setattr(evr, "load_eval_set", lambda: [
        {"query": "q1", "ground_truth_chunk_ids": [1], "expect_points": []},
    ])
    monkeypatch.setattr(evr, "compute_retrieval_metrics", lambda items, rf: {"recall_at_k": 0.8, "mrr": 0.9, "evidence_coverage": 0.7})

    m = evr.run_eval_and_record(triggered_by="手动", retrieve_fn=lambda q: [])
    assert m["recall_at_k"] == 0.8

    with sqlite_session_local() as s:
        er = s.query(EvalRun).one()
        assert er.triggered_by == "手动"
        assert er.dataset_size == 1
        assert er.metrics["mrr"] == 0.9
