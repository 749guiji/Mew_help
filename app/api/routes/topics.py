"""主题分布接口(ch10):读 topic_classifications 聚合各类目计数 + 类目问题列表。"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.db import SessionLocal
from app.db.models import LowConfidenceQuestion, TopicClassification

router = APIRouter()
STATIC_DIR = Path(__file__).resolve().parents[3] / "static"


@router.get("/topics")
def topics_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "topics.html")


@router.get("/api/topics/distribution")
def distribution() -> dict:
    with SessionLocal() as s:
        rows = s.query(TopicClassification).all()
    counts: dict[str, int] = {}
    for r in rows:
        for l in r.labels or []:
            counts[l] = counts.get(l, 0) + 1
    items = [{"label": k, "count": v} for k, v in sorted(counts.items(), key=lambda x: -x[1])]
    return {"items": items}


@router.get("/api/topics/questions")
def questions(label: str) -> dict:
    with SessionLocal() as s:
        rows = s.query(TopicClassification).all()
        matched = [r for r in rows if label in (r.labels or [])]
        qids = [r.low_conf_id for r in matched]
        qs = {q.id: q.raw_question for q in s.query(LowConfidenceQuestion).filter(LowConfidenceQuestion.id.in_(qids)).all()}
    return {"items": [{"text": qs.get(r.low_conf_id, ""), "labels": r.labels} for r in matched]}


@router.get("/api/topics/questions-data")
def questions_data(label: str) -> dict:
    """类目问题列表页数据源(纯 JSON,前端渲染)。"""
    return questions(label)
