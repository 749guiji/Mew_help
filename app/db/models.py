"""SQLAlchemy ORM models mapping to db/schema.sql."""

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
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
        Enum("售后", "投诉", "咨询", native_enum=False), nullable=False
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
