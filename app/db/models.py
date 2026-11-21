"""SQLAlchemy ORM models mapping to db/schema.sql."""

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects import mysql

from app.db import Base

# MySQL: BIGINT UNSIGNED; SQLite (tests): INTEGER so the PK autoincrements.
BIGINT_UNSIGNED = BigInteger().with_variant(
    mysql.BIGINT(unsigned=True), "mysql"
).with_variant(Integer, "sqlite")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(BIGINT_UNSIGNED, primary_key=True, autoincrement=True)
    user_id = Column(String(64), nullable=False, index=True)
    status = Column(
        Enum("进行中", "已转人工", "已结束", native_enum=False),
        nullable=False,
        server_default="进行中",
    )
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )
    # ch07 三层上下文:投影摘要 + 两锚点
    summary = Column(Text, nullable=True)
    summary_upto_msg_id = Column(BIGINT_UNSIGNED, nullable=True)
    layer1_from_msg_id = Column(BIGINT_UNSIGNED, nullable=True)
    title = Column(String(128), nullable=True)  # ch08 UX:用户自定义标题
    pinned = Column(Boolean, nullable=False, server_default="0")  # ch08 UX:置顶


class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"

    id = Column(BIGINT_UNSIGNED, primary_key=True, autoincrement=True)
    conversation_id = Column(
        BIGINT_UNSIGNED, ForeignKey("conversations.id"), nullable=False, index=True
    )
    seq = Column(Integer, nullable=False)
    from_msg_id = Column(BIGINT_UNSIGNED, nullable=False)
    upto_msg_id = Column(BIGINT_UNSIGNED, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("conversation_id", "seq", name="uk_conv_seq"),
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(BIGINT_UNSIGNED, primary_key=True, autoincrement=True)
    conversation_id = Column(
        BIGINT_UNSIGNED, ForeignKey("conversations.id"), nullable=False, index=True
    )
    role = Column(Enum("user", "assistant", "tool", native_enum=False), nullable=False)
    content = Column(Text, nullable=True)
    tool_calls = Column(JSON, nullable=True)
    tool_call_id = Column(String(64), nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())


class Faq(Base):
    __tablename__ = "faq"

    id = Column(BIGINT_UNSIGNED, primary_key=True, autoincrement=True)
    question = Column(String(512), nullable=False)
    answer = Column(Text, nullable=False)
    category = Column(String(64), nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )


class Ticket(Base):
    __tablename__ = "tickets"

    ticket_no = Column(String(32), primary_key=True)
    conversation_id = Column(
        BIGINT_UNSIGNED, ForeignKey("conversations.id"), nullable=False, index=True
    )
    description = Column(Text, nullable=False)
    ticket_type = Column(
        Enum("售后", "投诉", "咨询", "退款", native_enum=False), nullable=False
    )
    status = Column(
        Enum("待处理", "已处理", native_enum=False),
        nullable=False,
        server_default="待处理",
    )
    created_at = Column(DateTime, nullable=False, server_default=func.now())


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id = Column(BIGINT_UNSIGNED, primary_key=True, autoincrement=True)
    category = Column(String(255), nullable=False, index=True)
    questions = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    section_path = Column(String(512), nullable=True)
    content_type = Column(String(32), nullable=True)
    is_key_clause = Column(Integer, nullable=False, server_default="0")
    prev_chunk_id = Column(
        BIGINT_UNSIGNED,
        ForeignKey("knowledge_chunks.id", ondelete="SET NULL"),
        nullable=True,
    )
    next_chunk_id = Column(
        BIGINT_UNSIGNED,
        ForeignKey("knowledge_chunks.id", ondelete="SET NULL"),
        nullable=True,
    )
    vector_id = Column(String(64), nullable=True)
    vectorize_status = Column(
        Enum("pending", "done", native_enum=False),
        nullable=False,
        server_default="pending",
        index=True,
    )
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )


class QaExtractionStaging(Base):
    __tablename__ = "qa_extraction_staging"

    id = Column(BIGINT_UNSIGNED, primary_key=True, autoincrement=True)
    batch_no = Column(String(64), nullable=False, index=True)
    source_ref = Column(String(255), nullable=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    status = Column(
        Enum("extracted", "kept", "discarded", native_enum=False),
        nullable=False,
        server_default="extracted",
        index=True,
    )
    created_at = Column(DateTime, nullable=False, server_default=func.now())


class LowConfidenceQuestion(Base):
    __tablename__ = "low_confidence_questions"

    id = Column(BIGINT_UNSIGNED, primary_key=True, autoincrement=True)
    conversation_id = Column(
        BIGINT_UNSIGNED, ForeignKey("conversations.id"), nullable=True
    )
    raw_question = Column(Text, nullable=False)
    source = Column(
        Enum("retrieval_low_conf", "self_check", "user_feedback", native_enum=False),
        nullable=False,
        index=True,
    )
    reason = Column(Text, nullable=True)
    # ch09:落池时的召回片段快照(Top 几条原文+得分)与查重归并落点
    retrieved_chunks = Column(JSON, nullable=True)
    matched_review_id = Column(
        BIGINT_UNSIGNED,
        ForeignKey("review_queue.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    created_at = Column(DateTime, nullable=False, server_default=func.now(), index=True)


class FaithCase(Base):
    __tablename__ = "faith_cases"

    id = Column(BIGINT_UNSIGNED, primary_key=True, autoincrement=True)
    eval_id = Column(String(16), nullable=False, unique=True)
    bucket = Column(String(24), nullable=False)
    query = Column(String(512), nullable=False)
    strategy = Column(String(24), nullable=False, server_default="hybrid_rerank")
    answer = Column(Text, nullable=False)
    reason = Column(Text, nullable=False)
    citations = Column(JSON, nullable=True)
    judge_model = Column(String(64), nullable=True)
    status = Column(
        Enum("未解决", "已解决", "无需解决", native_enum=False),
        nullable=False,
        server_default="未解决",
        index=True,
    )
    seen_count = Column(
        Integer().with_variant(mysql.INTEGER(unsigned=True), "mysql"),
        nullable=False,
        server_default="1",
    )
    first_seen_at = Column(DateTime, nullable=False, server_default=func.now())
    last_seen_at = Column(
        DateTime, nullable=False, server_default=func.now(), index=True
    )
    resolution = Column(String(300), nullable=True)
    resolved_at = Column(DateTime, nullable=True)


class ToolAuditLog(Base):
    """工具调用审计留痕(ch08):内置和 MCP 工具都记,被权限拒/被校验拦也落一条。

    不挂外键:审计写入不能被引用约束拦住,conversation_id 只建普通索引。
    """

    __tablename__ = "tool_audit_logs"

    id = Column(BIGINT_UNSIGNED, primary_key=True, autoincrement=True)
    conversation_id = Column(BIGINT_UNSIGNED, nullable=True, index=True)
    tool_call_id = Column(String(64), nullable=True)
    tool_name = Column(String(128), nullable=False)
    tool_source = Column(Enum("builtin", "mcp", native_enum=False), nullable=False)
    mcp_server = Column(String(64), nullable=True)
    arguments = Column(JSON, nullable=True)
    result_summary = Column(Text, nullable=True)
    status = Column(
        Enum("成功", "失败", "超时", "校验拦下", "权限拒绝", native_enum=False),
        nullable=False,
    )
    error_message = Column(String(512), nullable=True)
    retry_count = Column(Integer, nullable=False, server_default="0")
    duration_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())


class ReviewQueue(Base):
    """飞轮待审队列(ch09):一行 = 一个去重后的知识缺口。"""

    __tablename__ = "review_queue"

    id = Column(BIGINT_UNSIGNED, primary_key=True, autoincrement=True)
    normalized_question = Column(String(512), nullable=False)
    ai_suggested_answer = Column(Text, nullable=True)
    occurrence_count = Column(Integer, nullable=False, server_default="1")
    review_status = Column(
        Enum("待审", "通过", "驳回", native_enum=False),
        nullable=False,
        server_default="待审",
    )
    approved_answer = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )


class EvalRun(Base):
    """自动化评估流水线轮次结果(ch09):一行一轮,按时间连成趋势。"""

    __tablename__ = "eval_runs"

    id = Column(BIGINT_UNSIGNED, primary_key=True, autoincrement=True)
    triggered_by = Column(
        Enum("定时", "手动", native_enum=False), nullable=False, server_default="定时"
    )
    dataset_size = Column(Integer, nullable=False)
    metrics = Column(JSON, nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now(), index=True)
