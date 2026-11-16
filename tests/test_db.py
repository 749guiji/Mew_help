import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base, SessionLocal, engine
from app.db.models import Conversation, Message, Faq, Ticket
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
