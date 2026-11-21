"""飞轮流水线(ch09):标准化 + 查重 + 入待审队列。

处理 ``low_confidence_questions`` 里尚未归并(``matched_review_id IS NULL``)的行:
1. 标准化:LLM 把口语原话改成标准 FAQ 式问题 + 一条示例答案。
2. 查重:LLM 判断与 ``review_queue`` 现有「待审/通过」问题是否同义。
3. 命中 → ``occurrence_count += 1`` + 回写 ``matched_review_id``;否则新建 review_queue 行。
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from app.db import SessionLocal
from app.db.models import LowConfidenceQuestion, ReviewQueue
from app.graph.json_utils import parse_json_object
from app.prompts.system import FLYWHEEL_DEDUPE_SYSTEM_PROMPT, FLYWHEEL_NORMALIZE_SYSTEM_PROMPT


def normalize_question(llm, raw: str) -> tuple[str, str | None]:
    """标准化:口语原话 → (标准 FAQ 问法, 示例答案)。解析失败回退原话。"""
    chain = ChatPromptTemplate.from_messages(
        [("system", FLYWHEEL_NORMALIZE_SYSTEM_PROMPT), ("human", "{input}")]
    ) | llm
    result = chain.invoke({"input": raw})
    text = getattr(result, "content", str(result)).strip()
    data = parse_json_object(text)
    if not data or not isinstance(data.get("question"), str) or not data["question"].strip():
        return raw, None
    answer = data.get("answer") if isinstance(data.get("answer"), str) else None
    return data["question"].strip(), answer


def dedupe_question(llm, normalized: str, candidates: list[tuple[int, str]]) -> int | None:
    """查重:返回与 normalized 同义的候选 review id,没有返回 None。"""
    if not candidates:
        return None
    lines = "\n".join(f"- id={cid}: {text}" for cid, text in candidates)
    chain = ChatPromptTemplate.from_messages(
        [
            ("system", FLYWHEEL_DEDUPE_SYSTEM_PROMPT),
            ("human", "新问题：{input}\n\n候选：\n{lines}"),
        ]
    ) | llm
    result = chain.invoke({"input": normalized, "lines": lines})
    text = getattr(result, "content", str(result)).strip()
    data = parse_json_object(text)
    match_id = data.get("match_id") if data else None
    if isinstance(match_id, int) and match_id in {cid for cid, _ in candidates}:
        return match_id
    return None


def run_flywheel() -> dict:
    """跑一轮飞轮,返回 {"processed": n, "new": x, "merged": y}。"""
    from app.core.config import get_settings
    from app.core.llm import get_llm

    llm = get_llm(get_settings())
    processed = new = merged = 0
    with SessionLocal() as s:
        pending = (
            s.query(LowConfidenceQuestion)
            .filter(LowConfidenceQuestion.matched_review_id.is_(None))
            .order_by(LowConfidenceQuestion.id)
            .all()
        )
        candidates = [
            (r.id, r.normalized_question)
            for r in s.query(ReviewQueue)
            .filter(ReviewQueue.review_status.in_(["待审", "通过"]))
            .all()
        ]
        for q in pending:
            norm, answer = normalize_question(llm, q.raw_question)
            dup_id = dedupe_question(llm, norm, candidates)
            if dup_id is not None:
                r = s.get(ReviewQueue, dup_id)
                if r is not None:
                    r.occurrence_count += 1
                    q.matched_review_id = dup_id
                    merged += 1
                    processed += 1
                    continue
            r = ReviewQueue(
                normalized_question=norm, ai_suggested_answer=answer, occurrence_count=1
            )
            s.add(r)
            s.flush()
            q.matched_review_id = r.id
            candidates.append((r.id, norm))
            new += 1
            processed += 1
        s.commit()
    return {"processed": processed, "new": new, "merged": merged}
