"""待审队列后台接口(ch09):列表 / 详情 / 通过(写回知识库)/ 驳回。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from app.db import SessionLocal
from app.db.models import KnowledgeChunk, LowConfidenceQuestion, ReviewQueue

router = APIRouter()


class ApproveRequest(BaseModel):
    approved_answer: str

    @field_validator("approved_answer")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


def _vectorize_new_chunk(session) -> None:
    """向量化 pending chunk(最佳努力;失败由调用方兜住、chunk 留 pending)。"""
    from app.kb.ingest import vectorize_pending
    from app.kb.retrieval import _get_embedder, _get_store

    vectorize_pending(session, _get_store(), _get_embedder())


@router.get("/api/review")
def list_review(status: str | None = None) -> dict:
    """列待审队列(出现次数高的在前);可按 status 过滤。"""
    with SessionLocal() as s:
        q = s.query(ReviewQueue)
        if status:
            q = q.filter(ReviewQueue.review_status == status)
        rows = q.order_by(ReviewQueue.occurrence_count.desc(), ReviewQueue.id.desc()).all()
        return {
            "items": [
                {
                    "id": r.id,
                    "normalized_question": r.normalized_question,
                    "ai_suggested_answer": r.ai_suggested_answer,
                    "occurrence_count": r.occurrence_count,
                    "review_status": r.review_status,
                    "approved_answer": r.approved_answer,
                }
                for r in rows
            ]
        }


@router.get("/api/review/{review_id}")
def review_detail(review_id: int) -> dict:
    """详情:缺口本身 + 归并进来的用户原话 + 当时的召回片段快照。"""
    with SessionLocal() as s:
        r = s.get(ReviewQueue, review_id)
        if r is None:
            raise HTTPException(status_code=404, detail="不存在")
        merged = (
            s.query(LowConfidenceQuestion)
            .filter(LowConfidenceQuestion.matched_review_id == review_id)
            .order_by(LowConfidenceQuestion.id)
            .all()
        )
        return {
            "item": {
                "id": r.id,
                "normalized_question": r.normalized_question,
                "ai_suggested_answer": r.ai_suggested_answer,
                "occurrence_count": r.occurrence_count,
                "review_status": r.review_status,
                "approved_answer": r.approved_answer,
            },
            "originals": [
                {
                    "id": q.id,
                    "raw_question": q.raw_question,
                    "source": q.source,
                    "retrieved_chunks": q.retrieved_chunks,
                }
                for q in merged
            ],
        }


@router.post("/api/review/{review_id}/approve")
def approve(review_id: int, payload: ApproveRequest) -> dict:
    """通过:标通过 + 存核准答案 + 写 knowledge_chunks(content_type=faq)+ 向量化。"""
    with SessionLocal() as s:
        r = s.get(ReviewQueue, review_id)
        if r is None:
            raise HTTPException(status_code=404, detail="不存在")
        r.review_status = "通过"
        r.approved_answer = payload.approved_answer

        chunk = KnowledgeChunk(
            category="FAQ",
            questions=r.normalized_question,
            answer=payload.approved_answer,
            section_path="待审队列",
            content_type="faq",
            is_key_clause=0,
            vectorize_status="pending",
        )
        s.add(chunk)
        s.commit()

        try:
            _vectorize_new_chunk(s)
        except Exception:  # noqa: BLE001 - 向量化失败留 pending,下次 vectorize 可重试
            pass
        return {"ok": True, "chunk_id": chunk.id}


@router.post("/api/review/{review_id}/reject")
def reject(review_id: int) -> dict:
    with SessionLocal() as s:
        r = s.get(ReviewQueue, review_id)
        if r is None:
            raise HTTPException(status_code=404, detail="不存在")
        r.review_status = "驳回"
        s.commit()
        return {"ok": True}
