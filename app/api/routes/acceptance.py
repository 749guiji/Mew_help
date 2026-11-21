"""验收页面数据源(ch10):读评测报告 / 数据产物 / 错例(静态文件)。"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()
STATIC_DIR = Path(__file__).resolve().parents[3] / "static"
ROOT = Path(__file__).resolve().parents[3]

ARTIFACTS = ROOT / "ml" / "artifacts"
DATA = ROOT / "ml" / "data"


def _read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


@router.get("/acceptance")
def acceptance_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "acceptance.html")


@router.get("/acceptance/eval")
def eval_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "acceptance_eval.html")


@router.get("/acceptance/data")
def data_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "acceptance_data.html")


@router.get("/acceptance/errors")
def errors_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "acceptance_errors.html")


@router.get("/api/acceptance/overview")
def overview() -> dict:
    return {"eval_report": _read_json(ARTIFACTS / "eval_report.json")}


@router.get("/api/acceptance/eval-report")
def eval_report() -> dict:
    return _read_json(ARTIFACTS / "eval_report.json")


@router.get("/api/acceptance/data")
def data_summary() -> dict:
    counts = {}
    for name in ("train", "val", "test", "raw_synthetic", "hand_written"):
        p = DATA / f"{name}.jsonl"
        counts[name] = sum(1 for _ in p.open(encoding="utf-8")) if p.is_file() else 0
    return {"files": counts}


@router.get("/api/acceptance/errors")
def errors() -> dict:
    p = ARTIFACTS / "errors.jsonl"
    items = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.is_file() else []
    return {"items": items}
