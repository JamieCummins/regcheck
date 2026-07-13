"""Database layer: async SQLAlchemy engine, ORM models, and session helpers.

Accounts (users, OAuth identities, API keys) and report ownership/visibility
metadata live in a relational database (Postgres in production via DATABASE_URL,
SQLite for local development and tests). Report *content* stays in Redis.
"""
from __future__ import annotations

from .base import Base
from . import models  # noqa: F401 - register ORM tables on Base.metadata
from .session import (
    create_engine_from_url,
    get_db,
    init_models,
    resolve_database_url,
)

__all__ = [
    "Base",
    "models",
    "create_engine_from_url",
    "get_db",
    "init_models",
    "resolve_database_url",
]
