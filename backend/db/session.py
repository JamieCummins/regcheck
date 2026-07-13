from __future__ import annotations

import os
import ssl
from pathlib import Path
from typing import AsyncIterator
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from fastapi import Request
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .base import Base

# Query params understood by libpq/psycopg but not by asyncpg via SQLAlchemy;
# stripped from the URL and translated into connect_args where relevant.
_LIBPQ_ONLY_PARAMS = {"sslmode", "channel_binding", "gssencmode"}


def _strip_libpq_params(url: str) -> tuple[str, dict[str, str]]:
    parts = urlsplit(url)
    if not parts.query:
        return url, {}
    kept: list[tuple[str, str]] = []
    removed: dict[str, str] = {}
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key.lower() in _LIBPQ_ONLY_PARAMS:
            removed[key.lower()] = value
        else:
            kept.append((key, value))
    rebuilt = urlunsplit(parts._replace(query=urlencode(kept)))
    return rebuilt, removed


def resolve_database_url() -> str:
    """Resolve and normalize the database URL for the async SQLAlchemy engine.

    - Honors DATABASE_URL (Heroku Postgres) and common alternatives.
    - Rewrites the driver to async (postgresql+asyncpg / sqlite+aiosqlite).
    - Defaults to a repo-local SQLite file for local development.
    """
    raw = (
        os.environ.get("DATABASE_URL")
        or os.environ.get("HEROKU_POSTGRESQL_URL")
        or ""
    ).strip()

    if not raw:
        base_dir = Path(__file__).resolve().parents[2]
        return f"sqlite+aiosqlite:///{base_dir / 'regcheck_dev.db'}"

    if raw.startswith("postgres://"):
        raw = "postgresql+asyncpg://" + raw[len("postgres://"):]
    elif raw.startswith("postgresql://"):
        raw = "postgresql+asyncpg://" + raw[len("postgresql://"):]
    elif raw.startswith("sqlite://") and "+aiosqlite" not in raw:
        raw = raw.replace("sqlite://", "sqlite+aiosqlite://", 1)

    cleaned, _removed = _strip_libpq_params(raw)
    return cleaned


def _is_remote_postgres(url: str) -> bool:
    if not url.startswith("postgresql"):
        return False
    host = (urlsplit(url).hostname or "").lower()
    return host not in {"", "localhost", "127.0.0.1", "::1"}


def _connect_args(url: str) -> dict:
    """Connection args, primarily TLS for managed Postgres (e.g. Heroku).

    DATABASE_SSL controls behavior: "disable" turns TLS off; "relaxed"/"require"
    give an encrypted but UNVERIFIED context (needed for managed providers with
    self-signed per-instance certs — an explicit, documented deployment choice);
    the DEFAULT for remote Postgres is full certificate verification.
    """
    mode = (os.environ.get("DATABASE_SSL") or "").strip().lower()
    if not url.startswith("postgresql+asyncpg"):
        return {}
    if mode == "disable":
        return {}
    if mode in {"require", "relaxed"}:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return {"ssl": ctx}
    if mode == "verify" or _is_remote_postgres(url):
        return {"ssl": ssl.create_default_context()}
    return {}


def _int_env(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int((os.environ.get(name) or "").strip()))
    except (TypeError, ValueError):
        return default


def _pool_kwargs(url: str) -> dict:
    """Connection-pool sizing. Only applied to managed Postgres — SQLite uses its
    own pool and ignores these. Defaults are modest so multiple web workers stay
    under a small Heroku Postgres connection cap; tune via env per plan.

    Each web process holds up to ``pool_size + max_overflow`` connections, so
    total ≈ that × WEB_CONCURRENCY. Defaults: (5 + 5) × 2 = 20.
    """
    if not url.startswith("postgresql"):
        return {}
    return {
        "pool_size": _int_env("DB_POOL_SIZE", 5, minimum=1),
        "max_overflow": _int_env("DB_MAX_OVERFLOW", 5, minimum=0),
        "pool_timeout": _int_env("DB_POOL_TIMEOUT", 30, minimum=1),
        "pool_recycle": _int_env("DB_POOL_RECYCLE", 1800, minimum=60),
    }


def create_engine_from_url(url: str | None = None) -> AsyncEngine:
    resolved = url or resolve_database_url()
    engine = create_async_engine(
        resolved,
        pool_pre_ping=True,
        connect_args=_connect_args(resolved),
        future=True,
        **_pool_kwargs(resolved),
    )
    if resolved.startswith("sqlite"):
        # Enforce foreign-key constraints (incl. ON DELETE CASCADE) on SQLite so
        # dev/test behavior matches Postgres.
        @event.listens_for(engine.sync_engine, "connect")
        def _enable_sqlite_fk(dbapi_connection, _record):  # pragma: no cover - trivial
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def create_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def init_models(engine: AsyncEngine) -> None:
    """Create tables from ORM metadata (used for SQLite/dev/tests; production
    schema is managed by Alembic migrations)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db(request: Request) -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a request-scoped AsyncSession."""
    sessionmaker = getattr(request.app.state, "db_sessionmaker", None)
    if sessionmaker is None:
        raise RuntimeError("Database is not configured on app.state")
    async with sessionmaker() as session:
        yield session
