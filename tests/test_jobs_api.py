"""Tests for the /api/jobs endpoints."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.jobs import router


def _client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_start_unknown_job_404(monkeypatch):
    def fake_start(name):
        raise KeyError(name)

    monkeypatch.setattr("app.core.jobs.start_job", fake_start)

    r = _client().post("/api/jobs/nope/start")
    assert r.status_code == 404


def test_start_known_job_returns_job_id(monkeypatch):
    monkeypatch.setattr("app.core.jobs.start_job", lambda name: "j1")

    r = _client().post("/api/jobs/kb-build/start")
    assert r.status_code == 200
    assert r.json() == {"job_id": "j1"}


def test_start_running_job_409(monkeypatch):
    monkeypatch.setattr("app.core.jobs.start_job", lambda name: None)

    r = _client().post("/api/jobs/kb-build/start")
    assert r.status_code == 409


def test_status_and_log_endpoints(monkeypatch):
    monkeypatch.setattr(
        "app.core.jobs.job_status",
        lambda name: {"running": False, "done": True, "error": None, "job_id": "j1"},
    )
    monkeypatch.setattr(
        "app.core.jobs.job_log_tail", lambda name, n_lines=200: "log tail"
    )

    c = _client()
    assert c.get("/api/jobs/kb-build/status").json()["done"] is True
    assert c.get("/api/jobs/kb-build/log").text == "log tail"


def test_status_unknown_job_404():
    r = _client().get("/api/jobs/nope/status")
    assert r.status_code == 404


def test_log_unknown_job_404():
    r = _client().get("/api/jobs/nope/log")
    assert r.status_code == 404
