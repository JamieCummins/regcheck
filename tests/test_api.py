from __future__ import annotations

import json
import os
import tempfile
from types import SimpleNamespace

import pytest

from backend.core import security
from backend.db import models
from backend.db.session import create_engine_from_url, create_sessionmaker, init_models
from backend.services import users as users_service
from backend.services import reports as reports_service


class FakeRedis:
    def __init__(self):
        self.h = {}
        self.v = {}

    async def ping(self):
        return True

    async def hgetall(self, k):
        return self.h.get(k, {})

    async def hset(self, k, mapping=None):
        self.h.setdefault(k, {}).update(mapping or {})

    async def get(self, k):
        return self.v.get(k)

    async def delete(self, *ks):
        for k in ks:
            self.h.pop(k, None)
            self.v.pop(k, None)

    async def srem(self, k, *m):
        pass

    async def exists(self, k):
        return 0


# ── service layer ────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_api_key_service_create_list_revoke(tmp_path):
    engine = create_engine_from_url(f"sqlite+aiosqlite:///{tmp_path / 'k.db'}")
    try:
        await init_models(engine)
        Session = create_sessionmaker(engine)
        async with Session() as s:
            user = models.User(email="k@uni.edu")
            s.add(user)
            await s.flush()
            key, raw = await users_service.create_api_key(s, user, "laptop")
            await s.commit()
            assert raw.startswith("rc_live_")
            assert key.key_hash == security.hash_api_key(raw)
            assert key.prefix == security.api_key_display_prefix(raw)

            keys = await users_service.list_api_keys(s, user.id)
            assert len(keys) == 1 and keys[0].is_active

            assert await users_service.revoke_api_key(s, user.id, key.id) is True
            await s.commit()
            assert (await users_service.list_api_keys(s, user.id))[0].is_active is False
            # Revoking again / a foreign key id is a no-op.
            assert await users_service.revoke_api_key(s, user.id, key.id) is False
            assert await users_service.revoke_api_key(s, "someone-else", key.id) is False
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_require_api_key_valid_invalid_revoked(tmp_path):
    from backend.core.api_auth import require_api_key
    from fastapi import HTTPException

    engine = create_engine_from_url(f"sqlite+aiosqlite:///{tmp_path / 'a.db'}")
    try:
        await init_models(engine)
        Session = create_sessionmaker(engine)
        async with Session() as s:
            user = models.User(email="a@uni.edu")
            s.add(user)
            await s.flush()
            key, raw = await users_service.create_api_key(s, user, "k")
            await s.commit()
            key_id = key.id

        def req(token):
            return SimpleNamespace(
                headers={"Authorization": f"Bearer {token}"} if token else {},
                state=SimpleNamespace(),
            )

        # Valid key → returns user + increments usage.
        async with Session() as s:
            resolved = await require_api_key(req(raw), s)
            assert resolved.id == user.id
        async with Session() as s:
            k = await s.get(models.ApiKey, key_id)
            assert k.request_count == 1 and k.last_used_at is not None

        # Missing / malformed.
        async with Session() as s:
            with pytest.raises(HTTPException) as e1:
                await require_api_key(req(None), s)
            assert e1.value.status_code == 401
            with pytest.raises(HTTPException):
                await require_api_key(req("not-a-key"), s)

        # Revoked key → 401.
        async with Session() as s:
            await users_service.revoke_api_key(s, user.id, key_id)
            await s.commit()
        async with Session() as s:
            with pytest.raises(HTTPException) as e2:
                await require_api_key(req(raw), s)
            assert e2.value.status_code == 401
    finally:
        await engine.dispose()


# ── route layer ──────────────────────────────────────────────────────────────
def _build_app(tmp):
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{tmp}/api.db"
    os.environ["SESSION_SECRET"] = "t"
    for k in ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "ORCID_CLIENT_ID", "ORCID_CLIENT_SECRET"):
        os.environ.pop(k, None)
    import backend.core.config as cfg
    cfg.get_settings.cache_clear()
    from backend import create_app
    app = create_app()
    app.state.redis = FakeRedis()
    return app


def test_api_v1_owner_scoping_and_auth():
    import asyncio
    from starlette.testclient import TestClient

    tmp = tempfile.mkdtemp()
    app = _build_app(tmp)
    with TestClient(app) as client:
        async def seed():
            Session = app.state.db_sessionmaker
            async with Session() as s:
                a = models.User(email="a@uni.edu", display_name="A")
                b = models.User(email="b@uni.edu", display_name="B")
                s.add_all([a, b])
                await s.flush()
                _ka, raw_a = await users_service.create_api_key(s, a, "a-key")
                await reports_service.create_report_row(
                    s, task_id="ra", owner_id=a.id, visibility="private", title="A report", comparison_type=None
                )
                await reports_service.create_report_row(
                    s, task_id="rb", owner_id=b.id, visibility="private", title="B report", comparison_type=None
                )
                await s.commit()
                return raw_a
        raw_a = asyncio.get_event_loop().run_until_complete(seed())
        app.state.redis.h["ra"] = {"state": "SUCCESS", "result_json": json.dumps({"items": [{"dimension": "X"}]}), "title": "A report"}

        # No key → 401.
        assert client.get("/api/v1/reports").status_code == 401

        h = {"Authorization": f"Bearer {raw_a}"}
        # List returns only A's report.
        listed = client.get("/api/v1/reports", headers=h).json()["reports"]
        assert [r["task_id"] for r in listed] == ["ra"]

        # Owned report → result; other's report → 404.
        got = client.get("/api/v1/reports/ra", headers=h)
        assert got.status_code == 200 and got.json()["result"]["items"][0]["dimension"] == "X"
        assert client.get("/api/v1/reports/rb", headers=h).status_code == 404

        # Status owned.
        assert client.get("/api/v1/status/ra", headers=h).json()["state"] == "SUCCESS"

        # Delete owned → gone.
        assert client.delete("/api/v1/reports/ra", headers=h).status_code == 200
        assert client.get("/api/v1/reports/ra", headers=h).status_code == 404


def test_api_resolve_dimensions_set_or_data():
    """The API accepts a built-in dimension_set preset OR a custom dimensions_data
    JSON (exactly one)."""
    from fastapi import HTTPException

    from backend.routes import api

    # Preset → JSON of the backend's dimensions for that discipline.
    out = api._resolve_api_dimensions("psychology", None)
    dims = json.loads(out)
    assert len(dims) == 9 and dims[0]["dimension"] == "Hypotheses"

    # Custom JSON passes through unchanged.
    raw = '[{"dimension":"X","definition":""}]'
    assert api._resolve_api_dimensions(None, raw) == raw
    assert api._resolve_api_dimensions("", raw) == raw  # blank set ignored

    # Both / neither / unknown → 400.
    for bad_args in [("psychology", raw), (None, None), ("bogus", None)]:
        with pytest.raises(HTTPException):
            api._resolve_api_dimensions(*bad_args)
