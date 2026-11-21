"""旁路批量主题归类(ch10):攒未归类的低置信度问题 → 调推理服务 → 写 topic_classifications。"""

from __future__ import annotations

import httpx
from sqlalchemy import select

from app.db import SessionLocal
from app.db.models import LowConfidenceQuestion, TopicClassification

INFERENCE_URL = "http://127.0.0.1:8001/classify"


def classify_topics(batch_size: int = 32) -> dict:
    with SessionLocal() as s:
        sub = select(TopicClassification.low_conf_id)
        pending = (
            s.query(LowConfidenceQuestion)
            .filter(~LowConfidenceQuestion.id.in_(sub))
            .limit(batch_size)
            .all()
        )
        if not pending:
            return {"processed": 0}
        texts = [q.raw_question for q in pending]
        try:
            resp = httpx.post(INFERENCE_URL, json={"texts": texts}, timeout=60)
            resp.raise_for_status()
            labels_list = resp.json()["labels"]
        except Exception as e:  # noqa: BLE001 - 服务不可用留痕跳过,不崩
            print(f"[classify-topics] 推理服务不可用: {e}")
            return {"processed": 0}

        for q, labels in zip(pending, labels_list):
            s.add(TopicClassification(low_conf_id=q.id, labels=labels))
        s.commit()
        return {"processed": len(pending)}
