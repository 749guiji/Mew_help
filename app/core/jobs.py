"""In-process job runner for the kb admin page.

The spec §10.3 wants a job runner behind the ``/kb`` page: a whitelist of named
jobs, no shell, log capture, heavy-job confirm, and no re-entrancy. Because
Milvus Lite is single-process per data dir (and the uvicorn process already
holds the Milvus store), jobs run **in-process** on a background thread instead
of as make subprocesses; the make targets in the ``Makefile`` call the same
underlying CLI commands for terminal use.
"""

from __future__ import annotations

import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.core.config import get_settings

_LOG_DIR = Path("log/jobs")


@dataclass
class Job:
    name: str
    run_fn: Callable[[], dict]
    is_heavy: bool = False


# Per-job state: running / done / error / job_id / thread. Keyed by job name.
_STATE: dict[str, dict] = {}


def _state(name: str) -> dict:
    if name not in _STATE:
        _STATE[name] = {
            "running": False,
            "done": False,
            "error": None,
            "job_id": None,
            "thread": None,
        }
    return _STATE[name]


def _log_file(name: str) -> Path:
    return _LOG_DIR / f"{name}.log"


# ``_START_LOCK`` makes the check-and-set in ``start_job`` atomic (TOCTOU); ``_RUN_LOCK``
# serializes job execution so the process-global stdout/stderr redirection in ``_wrap``
# is never clobbered by a second concurrent job (and matches Milvus Lite's single-writer model).
_START_LOCK = threading.Lock()
_RUN_LOCK = threading.Lock()


def start_job(name: str) -> str | None:
    """Start a registered job. Returns a job id, or ``None`` if already running.

    Raises ``KeyError`` for an unknown job name (the whitelist).
    """
    JOB_REGISTRY[name]  # raises KeyError for unknown names
    with _START_LOCK:
        st = _state(name)
        if st["running"]:
            return None
        job_id = uuid.uuid4().hex
        st.update(running=True, done=False, error=None, job_id=job_id)
        thread = threading.Thread(target=_wrap, args=(name,), daemon=True)
        st["thread"] = thread
        thread.start()
    return job_id


def _wrap(name: str) -> None:
    """Run a job with stdout/stderr captured to its log file (serialized)."""
    with _RUN_LOCK:
        job = JOB_REGISTRY[name]
        st = _state(name)
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(_log_file(name), "a", encoding="utf-8") as fh:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = fh
            try:
                job.run_fn()
                st["done"] = True
            except Exception as exc:  # noqa: BLE001 - surface in status, keep running
                st["error"] = str(exc)
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
                st["running"] = False


def job_status(name: str) -> dict:
    """Return the current status of a job (raises ``KeyError`` for unknown)."""
    JOB_REGISTRY[name]  # raises KeyError for unknown names
    st = _state(name)
    return {
        "running": st["running"],
        "done": st["done"],
        "error": st["error"],
        "job_id": st["job_id"],
    }


def job_log_tail(name: str, n_lines: int = 200) -> str:
    """Return the last ``n_lines`` of the job's log (``""`` when absent)."""
    path = _log_file(name)
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[-n_lines:])


# --- Job run functions (in-process, reuse the cached store/embedder) ---


def _run_kb_preview() -> dict:
    """Chunk every knowledge material without writing anything (dry run)."""
    from app.kb.sources import preview_kb

    result = preview_kb()
    print(
        f"kb-preview: {result['files']} files, {result['chunks']} chunks, "
        f"{result['key_clauses']} key clauses"
    )
    return result


def _run_kb_build() -> dict:
    """Ingest every material (chunk -> MySQL pending -> vectorize) + migrate faq.

    Mirrors ``ingest_docs`` (without ``--force``): refuses when chunks already
    exist, so a double-click cannot silently duplicate the corpus.
    """
    from app.db import SessionLocal
    from app.db.models import KnowledgeChunk
    from app.kb.embedding import Embedder
    from app.kb.ingest import ingest_document, migrate_faq
    from app.kb.retrieval import _get_store
    from app.kb.sources import list_kb_materials

    settings = get_settings()
    store = _get_store()
    embedder = Embedder(dim=settings.embedding_dim)
    total = 0
    with SessionLocal() as session:
        existing = session.query(KnowledgeChunk).count()
        if existing > 0:
            print(f"已有 {existing} 块，kb-build 跳过（用 kb-reset 重建）")
            return {"chunks": 0, "skipped": True, "existing": existing}
        for material in list_kb_materials():
            ids = ingest_document(
                session,
                store,
                embedder,
                material.path.read_text(encoding="utf-8"),
                material.content_type,
                max_chars=settings.chunk_max_chars,
                overlap=settings.chunk_overlap_chars,
                source=str(material.path),
            )
            total += len(ids)
            print(f"ingested {material.path} ({material.content_type}): {len(ids)} chunks")
        migrated = migrate_faq(session, store, embedder)
        total += migrated
        print(f"migrated faq: {migrated} chunks")
    print(f"kb-build: {total} chunks")
    return {"chunks": total}


def _run_kb_mine() -> dict:
    """Mine QA pairs from historical conversations into the KB."""
    from app.core.llm import get_llm
    from app.db import SessionLocal
    from app.kb.embedding import Embedder
    from app.kb.qa_extract import build_qa_extraction_chain, mine_conversations
    from app.kb.retrieval import _get_store

    settings = get_settings()
    store = _get_store()
    embedder = Embedder(dim=settings.embedding_dim)
    chain = build_qa_extraction_chain(get_llm(settings))
    with SessionLocal() as session:
        n = mine_conversations(
            session,
            store,
            embedder,
            chain,
            "knowledge/conversations/sample.jsonl",
        )
    print(f"kb-mine: {n} kept chunks")
    return {"chunks": n}


def _run_kb_vectorize() -> dict:
    """Vectorize any pending chunks (idempotent)."""
    from app.db import SessionLocal
    from app.kb.embedding import Embedder
    from app.kb.ingest import vectorize_pending
    from app.kb.retrieval import _get_store

    settings = get_settings()
    store = _get_store()
    embedder = Embedder(dim=settings.embedding_dim)
    with SessionLocal() as session:
        n = vectorize_pending(session, store, embedder)
    print(f"kb-vectorize: {n} pending chunks")
    return {"vectorized": n}


def _run_kb_reset() -> dict:
    """Drop the Milvus collection and re-vectorize every MySQL chunk."""
    from app.db import SessionLocal
    from app.kb.embedding import Embedder
    from app.kb.ingest import rebuild
    from app.kb.retrieval import _get_store

    settings = get_settings()
    store = _get_store()
    embedder = Embedder(dim=settings.embedding_dim)
    with SessionLocal() as session:
        n = rebuild(session, store, embedder)
    print(f"kb-reset: re-vectorized {n} chunks")
    return {"chunks": n}


def _run_flywheel() -> dict:
    """飞轮流水线:标准化 + 查重 + 入待审队列。"""
    from app.jobs.flywheel import run_flywheel

    result = run_flywheel()
    print(
        f"flywheel: processed={result['processed']} new={result['new']} merged={result['merged']}"
    )
    return result


def _run_eval() -> dict:
    """评估流水线:跑检索段指标并落 eval_runs。"""
    from app.jobs.eval_run import run_eval_and_record

    metrics = run_eval_and_record(triggered_by="手动")
    print(f"eval-run: {metrics}")
    return metrics


def _run_classify_topics() -> dict:
    """主题批量归类:调独立推理服务写 topic_classifications。"""
    from app.jobs.classify_topics import classify_topics

    result = classify_topics()
    print(f"classify-topics: {result}")
    return result


JOB_REGISTRY: dict[str, Job] = {
    "kb-preview": Job("kb-preview", _run_kb_preview, is_heavy=False),
    "kb-build": Job("kb-build", _run_kb_build, is_heavy=False),
    "kb-mine": Job("kb-mine", _run_kb_mine, is_heavy=True),
    "kb-vectorize": Job("kb-vectorize", _run_kb_vectorize, is_heavy=False),
    "kb-reset": Job("kb-reset", _run_kb_reset, is_heavy=True),
    "flywheel": Job("flywheel", _run_flywheel, is_heavy=True),
    "eval-run": Job("eval-run", _run_eval, is_heavy=True),
    "classify-topics": Job("classify-topics", _run_classify_topics, is_heavy=False),
}
