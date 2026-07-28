"""Database engine / session for Neon PostgreSQL."""

from __future__ import annotations

import logging
import os
from collections.abc import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

load_dotenv()

logger = logging.getLogger(__name__)


def _database_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Add it to .env (local) or Render env vars."
        )
    # SQLAlchemy prefers postgresql+psycopg2://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif url.startswith("postgresql://") and "+psycopg2" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    # channel_binding can break some drivers
    url = url.replace("&channel_binding=require", "").replace("?channel_binding=require&", "?")
    return url


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""


_engine = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine():
    """Lazy-create the SQLAlchemy engine."""
    global _engine, _SessionLocal
    if _engine is None:
        _engine = create_engine(
            _database_url(),
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=5,
            connect_args={"connect_timeout": 30, "sslmode": "require"},
        )
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
        logger.info("Database engine created")
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return session factory (initializes engine if needed)."""
    get_engine()
    assert _SessionLocal is not None
    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yield a DB session."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create tables if they do not exist."""
    # Import models so metadata is populated
    from api import models  # noqa: F401

    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    # Soft connectivity check
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("Database tables ready")
