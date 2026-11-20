"""ch07 上下文拼装:调模型前现场把完整历史裁成「三层 + 摘要」的精简版。

三层按 user/assistant 消息的序号切(工具消息归它前面的助手轮次):
- 层 3(已摘要):不渲染原文,只看 ``summary_text``。
- 层 2(中间):截短 —— 用户原话不动、客服答复留开头、工具结果一行标识。
- 层 1(最近):原文。

拼装顺序(system → 层2 → 层1 → 用户当前句 → 摘要+证据合成一条 HumanMessage)。
梗概不单独占 system(上游会合并重排 system,工具定义被挤到可变内容后、前缀缓存作废)。
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from app.prompts.system import SYSTEM_PROMPT

logger = logging.getLogger("context")

LAYER2_ASSISTANT_PREFIX = 40  # 层2 客服答复留开头 40 字


def _truncate_layer2(msg: BaseMessage) -> str:
    """层2 截短规则:用户原话不动;客服答复留开头;工具结果一行标识。"""
    if isinstance(msg, ToolMessage):
        return f"[工具:{msg.name or 'tool'}]"
    text = msg.content or ""
    if isinstance(msg, AIMessage):
        if len(text) > LAYER2_ASSISTANT_PREFIX:
            return text[:LAYER2_ASSISTANT_PREFIX] + "…"
        return text
    return text  # user 原样


def split_layers(
    messages: list[BaseMessage],
    *,
    summary_upto_pos: int = 0,
    layer1_from_pos: int | None = None,
) -> tuple[list[BaseMessage], list[BaseMessage], list[BaseMessage]]:
    """按 user/assistant 序号把 ``messages`` 切成 (层2, 层1, 当前轮)。

    ``layer1_from_pos``:层1 起点(0-based 的 user/assistant 序号,None=全在层1)。
    ``summary_upto_pos``:摘要覆盖到的序号(≤ 此序号的归层3,不返回原文)。
    返回的「当前轮」= 最后一条 user 消息及其后(不含前面的历史)。
    """
    layer2: list[BaseMessage] = []
    layer1: list[BaseMessage] = []
    current: list[BaseMessage] = []

    # 找到最后一条 user 消息的位置,把它及之后的算「当前轮」
    last_user = -1
    for i, m in enumerate(messages):
        if isinstance(m, HumanMessage):
            last_user = i
    if last_user < 0:
        return layer2, layer1, list(messages)

    ua = 0
    for i, m in enumerate(messages):
        if i < last_user:
            if isinstance(m, ToolMessage):
                pos = ua - 1
            else:
                pos = ua
                ua += 1
            if pos >= summary_upto_pos and (layer1_from_pos is None or pos >= layer1_from_pos):
                layer1.append(m)
            elif pos >= summary_upto_pos:
                layer2.append(m)
            # pos < summary_upto_pos → 层3,不渲染原文
        else:
            current.append(m)
    return layer2, layer1, current


def _render_layer2(msg: BaseMessage) -> BaseMessage:
    """层2 截短渲染,保留消息类型。"""
    text = _truncate_layer2(msg)
    if isinstance(msg, ToolMessage):
        return ToolMessage(
            content=text,
            tool_call_id=getattr(msg, "tool_call_id", None),
            name=getattr(msg, "name", None),
        )
    if isinstance(msg, AIMessage):
        return AIMessage(content=text)
    return HumanMessage(content=text)


def assemble_model_context(
    messages: list[BaseMessage],
    *,
    summary_text: str = "",
    evidence: str = "",
    summary_upto_pos: int = 0,
    layer1_from_pos: int | None = None,
) -> list[BaseMessage]:
    """主力 Agent 的精简版上下文。

    顺序:system → 层2截短 → 层1原文 → 用户当前句 → (摘要+证据)合成一条挂用户句之后。
    """
    layer2, layer1, current = split_layers(
        messages, summary_upto_pos=summary_upto_pos, layer1_from_pos=layer1_from_pos
    )
    out: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    out.extend(_render_layer2(m) for m in layer2)
    out.extend(layer1)
    out.extend(current)

    # 摘要 + 证据合成一条挂在用户句之后(不单独占 system)
    tail_parts = []
    if summary_text:
        tail_parts.append(f"【历史摘要】{summary_text}")
    if evidence:
        tail_parts.append(f"【检索证据】\n{evidence}")
    if tail_parts:
        out.append(HumanMessage(content="\n\n".join(tail_parts)))

    from app.core.budget import estimate_tokens

    logger.info(
        "model_ctx summary=%r window_msgs=%s tokens=%d",
        summary_text,
        [m.content for m in out[1:]],
        estimate_tokens("".join(str(m.content) for m in out)),
    )
    return out


def assemble_history_context(
    messages: list[BaseMessage],
    *,
    summary_text: str = "",
    max_turns: int = 6,
) -> str:
    """coref/意图共用的历史(字符串):摘要行 + 最近 N 轮原文滑窗。"""
    lines: list[str] = []
    if summary_text:
        lines.append(f"摘要:{summary_text}")
    recent = [m for m in messages if isinstance(m, (HumanMessage, AIMessage))][-max_turns * 2:]
    for m in recent:
        role = "用户" if isinstance(m, HumanMessage) else "客服"
        text = m.content if isinstance(m.content, str) else ""
        if text:
            lines.append(f"{role}:{text}")
    result = "\n".join(lines)
    logger.info("history_ctx %s", result)
    return result
