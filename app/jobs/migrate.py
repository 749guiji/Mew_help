"""CLI: idempotent MySQL schema migration for an already-initialised database.

``docker-compose`` only runs ``db/schema.sql`` on the container's first init,
so an existing database never receives tables added in later chapters (ch04's
``low_confidence_questions`` + ``faith_cases``). Run::

    python -m app.jobs.migrate

This creates any missing tables declared by :data:`app.db.Base` via
``Base.metadata.create_all`` (``CREATE TABLE IF NOT EXISTS`` semantics) and
prints which tables it ensured. It only runs on demand -- never at import time
or inside ``create_app`` -- so the test suite's SQLite ``SessionLocal``
monkeypatch never touches the real MySQL engine.
"""

from __future__ import annotations

from sqlalchemy import inspect
from sqlalchemy.engine import Engine

from app.db import Base, engine
# Import the ORM models so they register on ``Base.metadata`` before
# ``create_all`` inspects it. ``app.db`` itself never imports models, so a
# standalone ``python -m app.jobs.migrate`` would otherwise see an empty
# metadata and create nothing.
from app.db import models  # noqa: F401


def ensure_tables(target_engine: Engine = engine) -> list[str]:
    """Create missing tables on ``target_engine`` and return the names ensured."""
    existing = set(inspect(target_engine).get_table_names())
    Base.metadata.create_all(target_engine)
    return sorted(name for name in Base.metadata.tables if name not in existing)


def main(argv=None) -> int:
    ensured = ensure_tables()
    if ensured:
        print("ensured tables:", ", ".join(ensured))
    else:
        print("schema up to date: no tables missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
