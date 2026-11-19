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


def test_job_log_tail_returns_tail(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "_LOG_DIR", tmp_path)
    (tmp_path / "some.log").write_text("l1\nl2\nl3\nl4\nl5\n", encoding="utf-8")

    assert jobs.job_log_tail("some", n_lines=3) == "l3\nl4\nl5"
