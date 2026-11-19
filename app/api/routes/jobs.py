"""Job runner API: start / status / log tail for whitelisted kb jobs.

The job name is a whitelist entry in :data:`app.core.jobs.JOB_REGISTRY`; argv is
fixed inside the job's ``run_fn``, so the frontend only ever passes a name.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from app.core import jobs

router = APIRouter()


@router.post("/api/jobs/{name}/start")
def start(name: str) -> dict:
    try:
        job_id = jobs.start_job(name)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown job")
    if job_id is None:
        raise HTTPException(status_code=409, detail="job already running")
    return {"job_id": job_id}


@router.get("/api/jobs/{name}/status")
def status(name: str) -> dict:
    try:
        return jobs.job_status(name)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown job")


@router.get("/api/jobs/{name}/log")
def log(name: str) -> PlainTextResponse:
    if name not in jobs.JOB_REGISTRY:
        raise HTTPException(status_code=404, detail="unknown job")
    return PlainTextResponse(jobs.job_log_tail(name))
