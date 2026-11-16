from app.db import Base, SessionLocal, engine


def test_engine_and_session_exist():
    assert engine is not None
    assert SessionLocal is not None
    assert Base.metadata is not None
