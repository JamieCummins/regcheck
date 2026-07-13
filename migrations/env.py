from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import AsyncEngine

# Make the application package importable and pull in metadata + helpers.
from backend.db.base import Base
from backend.db import models  # noqa: F401 - ensures models register on Base.metadata
from backend.db.session import create_engine_from_url, resolve_database_url

# Fail the RELEASE phase with a clear message when a production dyno has no
# database configured — otherwise `alembic upgrade head` would silently run
# against the ephemeral SQLite fallback and the web dyno would refuse to boot
# afterwards. Failing here makes Heroku roll the deploy back instead.
if (os.environ.get("DYNO") or "").strip() and not (
    (os.environ.get("DATABASE_URL") or "").strip()
    or (os.environ.get("HEROKU_POSTGRESQL_URL") or "").strip()
):
    raise RuntimeError(
        "DATABASE_URL is not set on this production app; refusing to run "
        "migrations against the ephemeral SQLite fallback. Attach Heroku "
        "Postgres (heroku addons:create heroku-postgresql) or set "
        "DATABASE_URL, then redeploy."
    )

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    return resolve_database_url()


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL without a DB connection)."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_run_migrations(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        # Batch mode keeps ALTER operations working on SQLite (dev/test).
        render_as_batch=connection.dialect.name == "sqlite",
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations in 'online' mode using an async engine."""
    engine: AsyncEngine = create_engine_from_url(_database_url())
    try:
        async with engine.connect() as connection:
            await connection.run_sync(_do_run_migrations)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
