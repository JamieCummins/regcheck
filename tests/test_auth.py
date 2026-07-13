from __future__ import annotations

import pytest

from backend.core.oauth import OAuthIdentityInfo, extract_identity
from backend.db.session import create_engine_from_url, create_sessionmaker, init_models
from backend.services import users


def test_extract_identity_google_and_orcid():
    g = extract_identity("google", {"userinfo": {"sub": "g-1", "email": "a@uni.edu", "email_verified": True, "name": "Dr A"}})
    assert g.provider == "google" and g.subject == "g-1" and g.email == "a@uni.edu" and g.email_verified
    o = extract_identity("orcid", {"orcid": "0000-0002-1825-0097", "name": "Dr O"})
    assert o.provider == "orcid" and o.subject == "0000-0002-1825-0097" and o.email is None
    with pytest.raises(ValueError):
        extract_identity("facebook", {})


async def _session(tmp_path):
    engine = create_engine_from_url(f"sqlite+aiosqlite:///{tmp_path / 'auth.db'}")
    await init_models(engine)
    return engine, create_sessionmaker(engine)


@pytest.mark.asyncio
async def test_upsert_creates_then_returns_same_user(tmp_path):
    engine, Session = await _session(tmp_path)
    try:
        ident = OAuthIdentityInfo("google", "sub-1", "a@uni.edu", True, "Dr A")
        async with Session() as s:
            u1 = await users.upsert_oauth_user(s, ident)
            await s.commit()
            uid = u1.id
            assert u1.handle  # auto-generated
        # Same identity again -> same user, no duplicate identity.
        async with Session() as s:
            u2 = await users.upsert_oauth_user(s, ident)
            await s.commit()
            assert u2.id == uid
            from sqlalchemy import func, select
            from backend.db import models
            count = (await s.execute(select(func.count()).select_from(models.OAuthIdentity))).scalar_one()
            assert count == 1
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_linking_second_provider_to_existing_user(tmp_path):
    engine, Session = await _session(tmp_path)
    try:
        async with Session() as s:
            user = await users.upsert_oauth_user(s, OAuthIdentityInfo("google", "g-9", "b@uni.edu", True, "B"))
            await s.commit()
            uid = user.id
        # ORCID identity (no email) linked to the already-signed-in user.
        async with Session() as s:
            current = await users.get_user(s, uid)
            linked = await users.upsert_oauth_user(
                s, OAuthIdentityInfo("orcid", "0000-x", None, False, "B"), link_to_user=current
            )
            await s.commit()
            assert linked.id == uid
        async with Session() as s:
            from sqlalchemy import select
            from backend.db import models
            rows = (await s.execute(select(models.OAuthIdentity).where(models.OAuthIdentity.user_id == uid))).scalars().all()
            assert {r.provider for r in rows} == {"google", "orcid"}
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_unverified_email_does_not_autolink(tmp_path):
    engine, Session = await _session(tmp_path)
    try:
        async with Session() as s:
            await users.upsert_oauth_user(s, OAuthIdentityInfo("google", "g-1", "shared@uni.edu", True, "A"))
            await s.commit()
        async with Session() as s:
            # A different provider asserting the same email but NOT verified must
            # create a separate account (no takeover by unverified email).
            u = await users.upsert_oauth_user(s, OAuthIdentityInfo("orcid", "o-1", "shared@uni.edu", False, "Imposter"))
            await s.commit()
            from sqlalchemy import func, select
            from backend.db import models
            user_count = (await s.execute(select(func.count()).select_from(models.User))).scalar_one()
            assert user_count == 2
            assert u.email is None or u.email == "shared@uni.edu"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_update_profile_and_handle_uniqueness(tmp_path):
    engine, Session = await _session(tmp_path)
    try:
        async with Session() as s:
            a = await users.upsert_oauth_user(s, OAuthIdentityInfo("google", "a", "a@x.edu", True, "Alpha"))
            b = await users.upsert_oauth_user(s, OAuthIdentityInfo("google", "b", "b@x.edu", True, "Beta"))
            await s.commit()
            aid, bid = a.id, b.id
        async with Session() as s:
            a = await users.get_user(s, aid)
            await users.update_profile(s, a, handle="myhandle", research_field="Psychology", display_name="Alpha A")
            await s.commit()
        async with Session() as s:
            b = await users.get_user(s, bid)
            # Blank handle must not clobber the existing one.
            await users.update_profile(s, b, handle="myhandle")  # collides with a's handle
            await s.commit()
            assert b.handle != "myhandle"  # deduped
        async with Session() as s:
            a = await users.get_user(s, aid)
            assert a.handle == "myhandle"
            assert a.research_field == "Psychology"
            assert a.display_name == "Alpha A"
    finally:
        await engine.dispose()
