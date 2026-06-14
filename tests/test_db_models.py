from __future__ import annotations

import pytest
from sqlalchemy import select

from backend.core import security
from backend.db import models
from backend.db.session import (
    create_engine_from_url,
    create_sessionmaker,
    init_models,
    resolve_database_url,
)


def test_resolve_database_url_normalizes_heroku_scheme(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://u:p@host:5432/db")
    assert resolve_database_url() == "postgresql+asyncpg://u:p@host:5432/db"
    # libpq-only params (e.g. sslmode) are stripped so asyncpg can parse the URL.
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@host:5432/db?sslmode=require")
    assert resolve_database_url() == "postgresql+asyncpg://u:p@host:5432/db"


def test_resolve_database_url_defaults_to_sqlite(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("HEROKU_POSTGRESQL_URL", raising=False)
    url = resolve_database_url()
    assert url.startswith("sqlite+aiosqlite:///")


def test_api_key_helpers_roundtrip():
    key = security.generate_api_key()
    assert key.startswith("rc_live_")
    assert security.looks_like_api_key(key)
    assert not security.looks_like_api_key("nope")
    digest = security.hash_api_key(key)
    assert len(digest) == 64 and digest == security.hash_api_key(key)
    assert security.hash_api_key(key) != security.hash_api_key(security.generate_api_key())
    assert security.api_key_display_prefix(key).startswith("rc_live_")


@pytest.mark.asyncio
async def test_models_crud_and_cascade_delete(tmp_path):
    engine = create_engine_from_url(f"sqlite+aiosqlite:///{tmp_path / 'accounts.db'}")
    try:
        await init_models(engine)
        Session = create_sessionmaker(engine)

        async with Session() as s:
            user = models.User(email="a@uni.edu", display_name="Dr A", handle="dr-a",
                               research_field="Psychology")
            s.add(user)
            await s.flush()
            uid = user.id
            s.add(models.OAuthIdentity(user_id=uid, provider="orcid", subject="0000-1", email="a@uni.edu"))
            raw_key = security.generate_api_key()
            s.add(models.ApiKey(
                user_id=uid, name="laptop",
                prefix=security.api_key_display_prefix(raw_key),
                key_hash=security.hash_api_key(raw_key),
            ))
            s.add(models.Report(task_id="task-1", owner_id=uid, visibility="public", source="ui"))
            await s.commit()

        async with Session() as s:
            user = (await s.execute(select(models.User).where(models.User.id == uid))).scalar_one()
            assert user.research_field == "Psychology"
            assert user.is_public_profile is True
            key = (await s.execute(select(models.ApiKey).where(models.ApiKey.user_id == uid))).scalar_one()
            assert key.is_active and key.request_count == 0
            rep = (await s.execute(select(models.Report).where(models.Report.owner_id == uid))).scalar_one()
            assert rep.visibility == "public"

        # Deleting the user cascades (FK ON DELETE CASCADE, enforced on SQLite too).
        async with Session() as s:
            user = (await s.execute(select(models.User).where(models.User.id == uid))).scalar_one()
            await s.delete(user)
            await s.commit()

        async with Session() as s:
            assert (await s.execute(select(models.OAuthIdentity))).scalars().all() == []
            assert (await s.execute(select(models.ApiKey))).scalars().all() == []
            assert (await s.execute(select(models.Report))).scalars().all() == []
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_oauth_identity_uniqueness(tmp_path):
    from sqlalchemy.exc import IntegrityError

    engine = create_engine_from_url(f"sqlite+aiosqlite:///{tmp_path / 'uniq.db'}")
    try:
        await init_models(engine)
        Session = create_sessionmaker(engine)
        async with Session() as s:
            u1 = models.User(email="x@uni.edu")
            u2 = models.User(email="y@uni.edu")
            s.add_all([u1, u2])
            await s.flush()
            s.add(models.OAuthIdentity(user_id=u1.id, provider="google", subject="dup"))
            s.add(models.OAuthIdentity(user_id=u2.id, provider="google", subject="dup"))
            with pytest.raises(IntegrityError):
                await s.commit()
    finally:
        await engine.dispose()
