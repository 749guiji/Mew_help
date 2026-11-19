"""Tests for the in-process job runner machinery (app.core.jobs)."""

import threading
import time

import pytest

from app.core import jobs


def _make_blocking_job(event):
    def run():
        event.wait()
        return {"ok": True}

    return jobs.Job(name="fake", run_fn=run, is_heavy=False)


def test_start_job_unknown_name_raises():
    with pytest.raises(KeyError):
        jobs.start_job("nonexistent")


def test_reentrancy_rejected(monkeypatch, tmp_path):
    event = threading.Event()
    monkeypatch.setattr(jobs, "_LOG_DIR", tmp_path)
    monkeypatch.setitem(jobs.JOB_REGISTRY, "fake", _make_blocking_job(event))
    jobs._STATE.pop("fake", None)

    first = jobs.start_job("fake")
    assert first is not None
    assert jobs.start_job("fake") is None  # still running -> rejected

    event.set()  # release the blocking job
    for _ in range(200):
        if not jobs.job_status("fake")["running"]:
            break
        time.sleep(0.02)

    assert jobs.job_status("fake")["done"] is True
    assert jobs.start_job("fake") is not None  # finished -> can start again
    for _ in range(200):  # let the second run finish so no thread outlives the test
        if not jobs.job_status("fake")["running"]:
            break
        time.sleep(0.02)


def test_job_log_tail_returns_tail(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "_LOG_DIR", tmp_path)
    (tmp_path / "some.log").write_text("l1\nl2\nl3\nl4\nl5\n", encoding="utf-8")

    assert jobs.job_log_tail("some", n_lines=3) == "l3\nl4\nl5"


def test_jobs_run_serially(monkeypatch, tmp_path):
    event = threading.Event()
    order = []

    def job_a():
        order.append("a-start")
        event.wait()
        order.append("a-end")
        return {}

    def job_b():
        order.append("b-start")
        return {}

    monkeypatch.setattr(jobs, "_LOG_DIR", tmp_path)
    monkeypatch.setattr(jobs, "JOB_REGISTRY", {
        "a": jobs.Job("a", job_a, False),
        "b": jobs.Job("b", job_b, False),
    })
    jobs._STATE.clear()

    jobs.start_job("a")
    time.sleep(0.1)  # let job a start and block on the event
    jobs.start_job("b")

    assert "b-start" not in order  # b is serialized behind a

    event.set()
    for _ in range(200):
        if not jobs.job_status("a")["running"] and not jobs.job_status("b")["running"]:
            break
        time.sleep(0.02)

    assert order == ["a-start", "a-end", "b-start"]


def test_run_kb_build_refuses_when_existing(monkeypatch):
    import app.kb.ingest as ingest_mod

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def query(self, model):
            class _Q:
                def count(self):
                    return 5

            return _Q()

    called = []
    monkeypatch.setattr("app.db.SessionLocal", lambda: FakeSession())
    monkeypatch.setattr("app.kb.retrieval._get_store", lambda: object())
    monkeypatch.setattr("app.kb.embedding.Embedder", lambda *a, **k: object())
    monkeypatch.setattr(ingest_mod, "ingest_document", lambda *a, **k: called.append("ingest") or [1])
    monkeypatch.setattr(ingest_mod, "migrate_faq", lambda *a, **k: called.append("migrate") or 0)

    result = jobs._run_kb_build()

    assert result["skipped"] is True
    assert called == []
