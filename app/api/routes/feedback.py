"""👎 反馈落池端点(ch09)。

用户对某条回答点 👎 → 把当轮用户问题落进 low_confidence_questions(标 source=
user_feedback),并尽力回捞召回片段快照(对该问题文本重跑一次检索)。
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, field_validator

from app.db import SessionLocal
from app.services.generation_quality import log_low_confidence

router = APIRouter()


class FeedbackRequest(BaseModel):
    conversation_id: int | None = None
    message: str

    @field_validator("message")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


@router.post("/api/feedback")
def submit_feedback(payload: FeedbackRequest) -> dict:
    # 尽力回捞:重跑检索拿 Top 快照;异常/无果则 None
    snapshot = None
    try:
        from app.kb.retrieval import default_retrieve

        chunks = default_retrieve(payload.message)
        if chunks:
            snapshot = [
                {
                    "score": c.score,
                    "section_path": c.section_path or c.category,
                    "answer": c.answer[:200],
                }
                for c in chunks[:5]
            ]
    except Exception:  # noqa: BLE001 - 检索失败只影响快照
        snapshot = None

    try:
        with SessionLocal() as s:
            log_low_confidence(
                s,
                conversation_id=payload.conversation_id,
                raw_question=payload.message,
                source="user_feedback",
                reason="用户点 👎",
                retrieved_chunks=snapshot,
            )
    except Exception:  # noqa: BLE001 - 落池失败不拦反馈
        pass
    return {"ok": True}
