from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import get_settings

engine = create_engine(
    get_settings().database_url,
    connect_args={"charset": "utf8mb4"},
)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()
