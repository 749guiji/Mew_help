"""Graph nodes (ch05).

Deterministic + LLM nodes: coreference passthrough, intent recognition,
knowledge retrieval, confidence gate, fixed replies, and logging. The main
Agent node lives in :mod:`app.graph.agent`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate

from app.graph.json_utils import parse_json_object
from app.prompts.system import (
    COREF_REWRITE_SYSTEM_PROMPT,
    EXPAND_SYSTEM_PROMPT,
    INTENT_SYSTEM_PROMPT,
)

INTENTS = ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "建工单")

# 合法意图 = 七类 + 「其他」兜底;越界的类别名一律归「其他」,防止 route 崩溃。
VALID_INTENTS = INTENTS + ("其他",)


@dataclass
class IntentResult:
    """意图识别结果:intent 为七类之一或「其他」,confidence ∈ [0, 1]."""

    intent: str
    confidence: float

_COREF_HISTORY_MAX_CHARS = 2000


def _build_coref_context(query: str, history: list[str]) -> str:
    """把历史 + 当前问题拼成给 LLM 的上下文,超出字符预算时丢最旧消息。"""
    lines: list[str] = []
    total = 0
    for msg in reversed(history):
        total += len(msg)
        if total > _COREF_HISTORY_MAX_CHARS:
            break
        lines.append(msg)
    lines.reverse()
    history_text = "\n".join(f"- {m}" for m in lines) if lines else "(无)"
    return f"对话历史：\n{history_text}\n\n当前问题：{query}"


def _clean_single_line(text: str) -> str:
    """去掉代码围栏、配对引号与首尾空白,只留单句。"""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'“”":
        text = text[1:-1].strip()
    return text


def resolve_coref(llm, query: str, history: list[str]) -> str:
    """指代消解 + 改写:结合历史把 query 补全成独立完整问句(单句,非 JSON)。

    ``llm`` 是 Runnable;``history`` 为已提取的最近若干轮用户/助手消息文本
    (旧→新)。空输出/异常一律回退 ``query``。
    """
    chain = ChatPromptTemplate.from_messages(
        [("system", COREF_REWRITE_SYSTEM_PROMPT), ("human", "{context}")]
    ) | llm
    result = chain.invoke({"context": _build_coref_context(query, history)})
    text = getattr(result, "content", str(result)).strip()
    return _clean_single_line(text) or query


def classify_intent(llm, query: str, history: str = "") -> IntentResult:
    """Classify ``query`` into one of seven intents (+「其他」) via JSON prompt.

    结合最近几轮历史判当前意图(应对物流→退款→物流的漂移)。强制 JSON + 稳健解析:
    解析失败或字段缺失 → ``IntentResult("其他", 0.0)``。
    """
    context = f"最近对话(可空):\n{history or '(无)'}\n\n当前问题:{query}"
    chain = ChatPromptTemplate.from_messages(
        [("system", INTENT_SYSTEM_PROMPT), ("human", "{input}")]
    ) | llm
    result = chain.invoke({"input": context})
    text = getattr(result, "content", str(result)).strip()
    data = parse_json_object(text)
    if not data:
        return IntentResult("其他", 0.0)
    intent = data.get("intent")
    confidence = data.get("confidence")
    if not isinstance(intent, str) or not isinstance(confidence, (int, float)):
        return IntentResult("其他", 0.0)
    if intent not in VALID_INTENTS:
        return IntentResult("其他", 0.0)
    return IntentResult(intent, float(confidence))


def expand_queries(llm, query: str) -> list[str]:
    """把一句完整问法扩成多条侧重点不同的检索查询(强制 JSON ``queries[]``)。

    解析失败/空/queries 非字符串列表 → 回退 ``[query]``。
    """
    chain = ChatPromptTemplate.from_messages(
        [("system", EXPAND_SYSTEM_PROMPT), ("human", "{input}")]
    ) | llm
    result = chain.invoke({"input": query})
    text = getattr(result, "content", str(result)).strip()
    data = parse_json_object(text)
    queries = data.get("queries") if data else None
    if not isinstance(queries, list) or not queries:
        return [query]
    cleaned = [q for q in queries if isinstance(q, str) and q.strip()]
    return cleaned[:3] or [query]


def gate_pass(score: float | None, threshold: float) -> bool:
    """Confidence gate: pass only when a retrieval score meets the threshold."""
    return score is not None and score >= threshold


def retrieve_node(state) -> dict:
    """知识检索:调 ch04 的 default_retrieve 取 Top-K,填 retrieval_chunks/score."""
    from app.kb.retrieval import default_retrieve

    chunks = default_retrieve(state["resolved_query"])
    score = max((c.score for c in chunks), default=None) if chunks else None
    return {
        "retrieval_chunks": chunks,
        "retrieval_score": score,
        "visited": ["retrieve"],
    }


def gate_node(state) -> dict:
    """知识类置信度闸(ch09):三信号 evidence_confidence 判是否进 Agent。"""
    from app.graph.confidence import CONFIDENCE_THRESHOLD, compute_evidence_confidence

    r = compute_evidence_confidence(state.get("retrieval_chunks") or [])
    passed = r["confidence"] >= CONFIDENCE_THRESHOLD
    return {"confidence_pass": passed, "evidence_confidence": r, "visited": ["gate"]}


def fallback_node(state) -> dict:
    """知识类证据不足:回兜底话术 + 落池(带召回快照 + 置信度信号)。"""
    from langchain_core.messages import AIMessage

    from app.graph.confidence import compute_evidence_confidence
    from app.db import SessionLocal
    from app.services.generation_quality import log_low_confidence

    chunks = state.get("retrieval_chunks") or []
    conf = state.get("evidence_confidence") or compute_evidence_confidence(chunks)
    snapshot = [
        {
            "score": c.score,
            "section_path": c.section_path or c.category,
            "answer": c.answer[:200],
        }
        for c in chunks[:5]
    ]
    try:
        with SessionLocal() as s:
            log_low_confidence(
                s,
                conversation_id=state.get("conversation_id"),
                raw_question=state.get("resolved_query", ""),
                source="retrieval_low_conf",
                reason=f"evidence_confidence={conf['confidence']} signals={conf['signals']}",
                retrieved_chunks=snapshot or None,
            )
    except Exception:  # noqa: BLE001 - 记录失败不阻塞主流程
        pass
    return {
        "messages": [AIMessage(content="抱歉，我目前的知识库中没有足够信息回答您的问题，已为您记录并转人工处理。")],
        "visited": ["fallback"],
    }


def complaint_node(state) -> dict:
    """投诉:安抚话术 + 转人工/建工单两个建议(不自动执行)。"""
    from langchain_core.messages import AIMessage

    return {
        "messages": [AIMessage(content="非常抱歉给您带来了不好的体验。您可以点击下方按钮转接人工客服，或创建工单由我们跟进处理。")],
        "handoff_suggestions": ["转人工", "建工单"],
        "visited": ["complaint"],
    }


def log_node(state) -> dict:
    """日志记录:把本次路径 + 意图 + 分流写进日志(供验收#1)。"""
    import logging

    logging.getLogger("graph").info(
        "visited=%s intent=%s route=%s",
        state.get("visited"),
        state.get("intent"),
        state.get("route"),
    )
    return {"visited": ["log"]}


_ORDER_RE = re.compile(r"(?:订单|单号|尾号)\s*(\d{4,})")


def _extract_order_id(text: str) -> str | None:
    """从文本抽订单号;抽不到返回 None。

    要求「订单/单号/尾号」上下文前缀,避免把「我1200买的」这种金额当成订单号
    (缺单不猜不编——抽不到就走 interrupt 让用户点选,不 fabricate 订单)。
    """
    m = _ORDER_RE.search(text or "")
    return m.group(1) if m else None


def fetch_order(state) -> dict:
    """退款子流程第一步:抽订单号;缺则 ``interrupt`` 弹订单选择器等前端点选;拿到后取订单数据。

    ``interrupt`` 之前只做只读(resume 时本节点从头重跑,不能有副作用)。
    """
    from langgraph.types import interrupt

    from app.orders import list_user_orders, order_snapshot

    oid = state.get("order_id") or _extract_order_id(
        state.get("resolved_query") or state.get("raw_query", "")
    )
    if not oid:
        orders = list_user_orders(state.get("user_id") or "guest")
        oid = interrupt({"type": "select_order", "orders": orders})
    return {"order_id": oid, "order_data": order_snapshot(oid), "visited": ["fetch_order"]}


def retrieve_policy(state) -> dict:
    """退款子流程强制检索政策:Query 扩写 3 条 → 多 query 检索 → 去重合并。"""
    from app.core.config import get_settings
    from app.core.llm import get_llm
    from app.kb.retrieval import retrieve_multi

    q = state.get("resolved_query") or state.get("raw_query") or ""
    queries = expand_queries(get_llm(get_settings()), q)
    chunks = retrieve_multi(queries)
    score = max((c.score for c in chunks), default=None) if chunks else None
    return {"retrieval_chunks": chunks, "retrieval_score": score, "visited": ["retrieve_policy"]}


def script_reply(state) -> dict:
    """闲聊/其他 兜底话术(零模型):按 intent 分文案——闲聊引回产品,其他请说具体些。

    分流后立即命中、不进 Agent;与知识路证据弱的 ``fallback_node`` 是两码事。
    """
    from langchain_core.messages import AIMessage

    if state.get("intent") == "其他":
        text = "抱歉，我不太确定您的意思。您可以把问题说得更具体些吗？比如您想咨询的商品、某个订单，或退款/售后问题。"
    else:
        text = "我暂时还不会回答这个，请告诉我您对我们产品的任何咨询~商品、订单、物流、售后都可以问我哦。"
    return {"messages": [AIMessage(content=text)], "visited": ["script_reply"]}


async def confirm_ticket(state) -> dict:
    """建工单确认流:interrupt 推工单预览;resume 放行落 tickets 或按权限拒绝落审计。

    无副作用:resume 时本节点从头重跑,interrupt 之前只读 state。
    """
    from langgraph.types import interrupt
    from langchain_core.messages import AIMessage, ToolMessage

    from app.tools import registry
    from app.tools.audit import log_tool_audit
    from app.tools.executor import ToolExecutor

    pending = state.get("pending_ticket") or {}
    args = pending.get("args") or {}
    tool_call_id = pending.get("tool_call_id")
    conversation_id = state.get("conversation_id")

    decision = interrupt(
        {
            "type": "ticket_preview",
            "draft": {
                "ticket_type": args.get("ticket_type", ""),
                "description": args.get("description", ""),
            },
        }
    )

    confirmed = isinstance(decision, dict) and decision.get("confirm") is True
    if confirmed:
        tool = registry.get("create_ticket")
        try:
            ticket_no = await ToolExecutor().execute_confirmed(
                tool, args, tool_call_id=tool_call_id, conversation_id=conversation_id
            )
        except Exception:
            return {
                "messages": [
                    ToolMessage(content="工单创建失败", tool_call_id=tool_call_id, name="create_ticket"),
                    AIMessage(content="工单创建失败，请稍后重试或联系人工客服。"),
                ],
                "pending_ticket": None,
                "visited": ["confirm_ticket"],
            }
        return {
            "messages": [
                ToolMessage(content=ticket_no, tool_call_id=tool_call_id, name="create_ticket"),
                AIMessage(content=f"工单已创建，工单号 {ticket_no}，我们会尽快跟进处理。"),
            ],
            "pending_ticket": None,
            "visited": ["confirm_ticket"],
        }

    # 取消 / 载荷形状不对 → 不执行,按权限拒绝落审计
    log_tool_audit(
        conversation_id=conversation_id,
        tool_call_id=tool_call_id,
        tool_name="create_ticket",
        tool_source="builtin",
        mcp_server=None,
        arguments=args,
        status="权限拒绝",
        error_message="用户取消建工单",
        retry_count=0,
    )
    return {
        "messages": [
            ToolMessage(content="已取消建工单", tool_call_id=tool_call_id, name="create_ticket"),
            AIMessage(content="已取消建工单，如需帮助请随时告诉我。"),
        ],
        "pending_ticket": None,
        "visited": ["confirm_ticket"],
    }


def self_check_node(state) -> dict:
    """生成阶段自评(ch09):agent 出答案后判证据是否够支撑;不足→落池,不拦答案。

    放在 agent 之后、log 之前。答案已经在 agent 节点里流式发给了用户,
    这里只做「事后判断 + 落池」,不改/不追加 messages。
    """
    from langchain_core.messages import AIMessage

    from app.core.config import get_settings
    from app.core.llm import get_llm
    from app.db import SessionLocal
    from app.services.generation_quality import (
        format_evidence,
        log_low_confidence,
        self_check,
    )

    answer = ""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage) and m.content:
            answer = m.content
            break
    chunks = state.get("retrieval_chunks") or []
    if not answer or not chunks:
        return {"visited": ["self_check"]}

    try:
        result = self_check(
            get_llm(get_settings()), state.get("resolved_query", ""), format_evidence(chunks)
        )
    except Exception:  # noqa: BLE001 - 自评失败不拦主流程
        return {"visited": ["self_check"]}

    if not result.sufficient:
        snapshot = [
            {
                "score": c.score,
                "section_path": c.section_path or c.category,
                "answer": c.answer[:200],
            }
            for c in chunks[:5]
        ]
        try:
            with SessionLocal() as s:
                log_low_confidence(
                    s,
                    conversation_id=state.get("conversation_id"),
                    raw_question=state.get("resolved_query", ""),
                    source="self_check",
                    reason=result.reason,
                    retrieved_chunks=snapshot or None,
                )
        except Exception:  # noqa: BLE001 - 落池失败不拦主流程
            pass
    return {"visited": ["self_check"]}
