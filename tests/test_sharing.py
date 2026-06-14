from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile

import itsdangerous
import pytest

from backend.db import models
from backend.db.session import create_engine_from_url, create_sessionmaker, init_models
from backend.services import reports as reports_service
from backend.services import sharing as sharing_service


class FakeRedis:
    def __init__(self):
        self.h: dict[str, dict] = {}

    async def ping(self):
        return True

    async def hgetall(self, k):
        return self.h.get(k, {})

    async def hset(self, k, mapping=None):
        self.h.setdefault(k, {}).update(mapping or {})

    async def exists(self, k):
        return 1 if k in self.h else 0

    async def get(self, k):
        return None

    async def delete(self, *ks):
        for k in ks:
            self.h.pop(k, None)

    async def srem(self, k, *m):
        pass


# ── parse_grantee ──────────────────────────────────────────────────────────────
def test_parse_grantee():
    assert sharing_service.parse_grantee("Jane@Uni.EDU") == ("jane@uni.edu", None)
    assert sharing_service.parse_grantee("0000-0002-1825-0097") == (None, "0000-0002-1825-0097")
    assert sharing_service.parse_grantee("0000-0002-1825-009x") == (None, "0000-0002-1825-009X")
    assert sharing_service.parse_grantee("https://orcid.org/0000-0002-1825-009X") == (None, "0000-0002-1825-009X")
    for bad in ("", "   ", "not-an-email", "1234"):
        with pytest.raises(sharing_service.ShareError):
            sharing_service.parse_grantee(bad)


# ── service: add / list / remove ────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_share_service_add_list_remove(tmp_path):
    engine = create_engine_from_url(f"sqlite+aiosqlite:///{tmp_path / 's.db'}")
    try:
        await init_models(engine)
        Session = create_sessionmaker(engine)
        async with Session() as s:
            owner = models.User(email="o@uni.edu")
            s.add(owner)
            await s.flush()
            await reports_service.create_report_row(
                s, task_id="rr", owner_id=owner.id, visibility="restricted", title="R", comparison_type=None
            )
            await s.commit()

        async with Session() as s:
            a = await sharing_service.add_share(s, "rr", "g@uni.edu")
            b = await sharing_service.add_share(s, "rr", "0000-0002-1825-0097")
            await s.commit()
            assert a.grantee_email == "g@uni.edu" and b.grantee_orcid == "0000-0002-1825-0097"
            # Idempotent: re-adding the same email returns the same row.
            again = await sharing_service.add_share(s, "rr", "G@UNI.EDU")
            assert again.id == a.id

            shares = await sharing_service.list_shares(s, "rr")
            assert {sharing_service.share_label(x) for x in shares} == {"g@uni.edu", "0000-0002-1825-0097"}

            assert await sharing_service.remove_share(s, "rr", a.id) is True
            await s.commit()
            assert await sharing_service.remove_share(s, "rr", a.id) is False  # already gone
            assert len(await sharing_service.list_shares(s, "rr")) == 1
    finally:
        await engine.dispose()


# ── service: user_can_view matrix ───────────────────────────────────────────────
@pytest.mark.asyncio
async def test_user_can_view_matrix(tmp_path):
    engine = create_engine_from_url(f"sqlite+aiosqlite:///{tmp_path / 'v.db'}")
    try:
        await init_models(engine)
        Session = create_sessionmaker(engine)
        async with Session() as s:
            owner = models.User(email="owner@uni.edu")
            by_email = models.User(email="grant@uni.edu")
            by_orcid = models.User(email=None)
            stranger = models.User(email="nope@uni.edu")
            s.add_all([owner, by_email, by_orcid, stranger])
            await s.flush()
            s.add(models.OAuthIdentity(user_id=by_orcid.id, provider="orcid", subject="0000-0002-1825-0097"))
            await reports_service.create_report_row(
                s, task_id="rr", owner_id=owner.id, visibility="restricted", title="R", comparison_type=None
            )
            await sharing_service.add_share(s, "rr", "grant@uni.edu")
            await sharing_service.add_share(s, "rr", "0000-0002-1825-0097")
            await s.commit()
            ids = (owner.id, by_email.id, by_orcid.id, stranger.id)

        async with Session() as s:
            report = await reports_service.get_report_row(s, "rr")
            owner, by_email, by_orcid, stranger = [await s.get(models.User, i) for i in ids]
            assert await sharing_service.user_can_view(s, report, owner) is True
            assert await sharing_service.user_can_view(s, report, by_email) is True
            assert await sharing_service.user_can_view(s, report, by_orcid) is True
            assert await sharing_service.user_can_view(s, report, stranger) is False
            assert await sharing_service.user_can_view(s, report, None) is False
    finally:
        await engine.dispose()


# ── integration: end-to-end access enforcement ──────────────────────────────────
def _build_app(tmp):
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{tmp}/share.db"
    os.environ["SESSION_SECRET"] = "share-secret"
    for k in ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "ORCID_CLIENT_ID", "ORCID_CLIENT_SECRET"):
        os.environ.pop(k, None)
    import backend.core.config as cfg
    cfg.get_settings.cache_clear()
    from backend import create_app
    app = create_app()
    app.state.redis = FakeRedis()
    return app


def _cookie_for(uid: str) -> str:
    signer = itsdangerous.TimestampSigner("share-secret")
    return signer.sign(base64.b64encode(json.dumps({"user_id": uid}).encode())).decode()


def test_restricted_access_enforcement():
    from starlette.testclient import TestClient

    tmp = tempfile.mkdtemp()
    app = _build_app(tmp)
    with TestClient(app) as client:
        async def seed():
            Session = app.state.db_sessionmaker
            async with Session() as s:
                owner = models.User(email="owner@uni.edu", display_name="Owner")
                grantee = models.User(email="grant@uni.edu", display_name="Grantee")
                stranger = models.User(email="nope@uni.edu", display_name="Stranger")
                s.add_all([owner, grantee, stranger])
                await s.flush()
                for tid, vis in (("rr", "restricted"), ("pp", "public"), ("uu", "unlisted")):
                    await reports_service.create_report_row(
                        s, task_id=tid, owner_id=owner.id, visibility=vis, title=tid.upper(), comparison_type=None
                    )
                await sharing_service.add_share(s, "rr", "grant@uni.edu")
                await s.commit()
                return owner.id, grantee.id, stranger.id

        owner_id, grantee_id, stranger_id = asyncio.get_event_loop().run_until_complete(seed())
        success = {"state": "SUCCESS", "result_json": json.dumps({"items": []}), "title": "RR"}
        for tid, vis in (("rr", "restricted"), ("pp", "public"), ("uu", "unlisted")):
            app.state.redis.h[tid] = {**success, "visibility": vis, "owner_id": owner_id}

        def as_user(uid):
            client.cookies.clear()
            client.cookies.set("session", _cookie_for(uid))

        def as_anon():
            client.cookies.clear()

        # Anonymous: public + unlisted open by link; restricted redirects to login.
        as_anon()
        assert client.get("/result/pp").status_code == 200
        assert client.get("/result/uu").status_code == 200
        r = client.get("/result/rr", follow_redirects=False)
        assert r.status_code == 302 and "/login" in r.headers["location"]
        assert client.get("/task_status/rr").json()["state"] == "FORBIDDEN"

        # Signed-in non-grantee: forbidden everywhere for the restricted report.
        as_user(stranger_id)
        assert client.get("/result/rr").status_code == 403
        assert client.get("/task_status/rr").status_code == 403
        assert client.get("/report/rr/manifest").status_code == 403

        # Granted viewer: full access.
        as_user(grantee_id)
        assert client.get("/result/rr").status_code == 200
        assert client.get("/task_status/rr").json()["state"] == "SUCCESS"

        # Owner can add a grantee, which immediately unlocks access; and remove it.
        as_user(owner_id)
        add = client.post("/reports/rr/shares", data={"grantee": "nope@uni.edu"})
        assert add.status_code == 200 and add.json()["ok"] is True
        listed = client.get("/reports/rr/shares").json()["shares"]
        assert {s["label"] for s in listed} == {"grant@uni.edu", "nope@uni.edu"}
        share_id = next(s["id"] for s in listed if s["label"] == "nope@uni.edu")
        # A bad grantee string is rejected.
        bad = client.post("/reports/rr/shares", data={"grantee": "not-valid"})
        assert bad.status_code == 400 and bad.json()["ok"] is False

        as_user(stranger_id)
        assert client.get("/result/rr").status_code == 200  # now granted

        as_user(owner_id)
        assert client.post(f"/reports/rr/shares/{share_id}/delete").json()["ok"] is True

        as_user(stranger_id)
        assert client.get("/result/rr").status_code == 403  # access revoked

        # A non-owner cannot manage sharing.
        as_user(grantee_id)
        assert client.post("/reports/rr/shares", data={"grantee": "x@uni.edu"}).status_code == 403
