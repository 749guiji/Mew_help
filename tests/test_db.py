import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base, SessionLocal, engine
from app.db.models import Conversation, ConversationSummary, Message, Faq, Ticket
from app.db.seed import seed_faq


def test_engine_and_session_exist():
    assert engine is not None
    assert SessionLocal is not None
    assert Base.metadata is not None


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


def test_seed_faq_has_return_policy(session):
    n = seed_faq(session)
    assert n >= 1
    q = session.query(Faq).filter(Faq.question.like("%退货%")).first()
    assert q is not None and q.answer


def test_ticket_uses_ticket_no_pk(session):
    t = Ticket(ticket_no="T20261116001", conversation_id=1, description="x", ticket_type="售后", status="待处理")
    session.add(t); session.commit()
    assert session.get(Ticket, "T20261116001") is t


def test_conversation_has_summary_columns(session):
    c = Conversation(
        user_id="u1",
        status="进行中",
        summary="订单1001想退货",
        summary_upto_msg_id=5,
        layer1_from_msg_id=10,
    )
    session.add(c)
    session.commit()
    assert c.summary == "订单1001想退货"
    assert c.summary_upto_msg_id == 5
    assert c.layer1_from_msg_id == 10


def test_conversation_summary_append_only(session):
    s = ConversationSummary(
        conversation_id=1, seq=1, from_msg_id=1, upto_msg_id=5, content="第1段"
    )
    session.add(s)
    session.commit()
    assert s.id is not None
    assert session.get(ConversationSummary, s.id).content == "第1段"
