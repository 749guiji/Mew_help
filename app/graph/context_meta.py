"""加载会话的上下文元数据(summary 投影 + 两锚点换算成 user/assistant 序号)。"""

from __future__ import annotations

from app.db import SessionLocal
from app.db.models import Conversation, Message


def load_context_meta(conversation_id: int | None) -> tuple[str, int, int | None]:
    """返回 ``(summary_text, summary_upto_pos, layer1_from_pos)``。

    - ``summary_upto_pos`` = 已摘要的 user/assistant 消息数(COUNT(id ≤ summary_upto_msg_id))。
    - ``layer1_from_pos`` = 层1 起点序号(COUNT(id ≤ layer1_from_msg_id));无锚点返回 None。
    """
    if not conversation_id:
        return "", 0, None
    with SessionLocal() as s:
        conv = s.get(Conversation, conversation_id)
        if conv is None:
            return "", 0, None

        def count_upto(upto_id: int | None) -> int:
            if not upto_id:
                return 0
            return (
                s.query(Message)
                .filter(Message.conversation_id == conversation_id, Message.id <= upto_id)
                .count()
            )

        summary_upto_pos = count_upto(conv.summary_upto_msg_id)
        layer1_from_pos = (
            count_upto(conv.layer1_from_msg_id) if conv.layer1_from_msg_id else None
        )
        return conv.summary or "", summary_upto_pos, layer1_from_pos
