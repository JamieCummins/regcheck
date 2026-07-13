"""Authorization regression tests for the P0 access-control fixes.

Covers the two stop-ship defects: browser-session ownership must NOT override DB
ownership (cross-account management), and a missing DB row must NOT fail open for
an owned/private report (enforced from Redis metadata instead)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from backend.routes.reports import _can_manage
from backend.services import sharing as sharing_service


def _run(coro):
    """Run on a private loop WITHOUT nulling the global current loop — asyncio.run()
    would, breaking the legacy get_event_loop() tests that run after this module."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _request(*, user_id: str | None, owned: list[str], redis=None):
    session = {"owned_reports": list(owned)}
    if user_id is not None:
        session["user_id"] = user_id
    state = SimpleNamespace(user=SimpleNamespace(id=user_id) if user_id else None)
    app = SimpleNamespace(state=SimpleNamespace(redis=redis, db_sessionmaker=None))
    return SimpleNamespace(session=session, state=state, app=app)


def _report(owner_id):
    return SimpleNamespace(owner_id=owner_id)


def test_can_manage_owner_matches():
    req = _request(user_id="A", owned=[])
    assert _can_manage(req, "t1", _report("A")) is True


def test_session_ownership_does_not_override_db_owner():
    # User B shares the browser; the session still lists A's report id. B must NOT
    # be able to manage A's DB-owned report.
    req = _request(user_id="B", owned=["t1"])
    assert _can_manage(req, "t1", _report("A")) is False
    # And an anonymous user (no account) with the stale session entry is also denied.
    anon = _request(user_id=None, owned=["t1"])
    assert _can_manage(anon, "t1", _report("A")) is False


def test_session_ownership_manages_anonymous_report():
    # No owned DB row → the creating browser session may manage it.
    req = _request(user_id=None, owned=["t1"])
    assert _can_manage(req, "t1", None) is True
    assert _can_manage(req, "other", None) is False


class _Redis:
    def __init__(self, hashes):
        self.hashes = hashes

    async def hgetall(self, key):
        return self.hashes.get(key, {})


def _viewable(hashes, task_id, *, user_id=None):
    redis = _Redis(hashes)
    req = _request(user_id=user_id, owned=[], redis=redis)
    return _run(sharing_service._viewable_from_redis_metadata(req, task_id))


def test_missing_row_anonymous_report_is_public():
    # No owner metadata → genuinely anonymous → public by link (designed).
    v = _viewable({"t": {"visibility": "public"}}, "t")
    assert v.allowed is True


def test_missing_row_owned_private_fails_closed():
    # Owned + private in Redis but no DB row (row creation failed / lost): must NOT
    # fail open. Anonymous viewer → needs login; non-owner → denied; owner → allowed.
    meta = {"t": {"owner_id": "A", "visibility": "private"}}
    assert _viewable(meta, "t").needs_login is True
    assert _viewable(meta, "t", user_id="B").allowed is False
    assert _viewable(meta, "t", user_id="A").allowed is True


def test_missing_row_owned_public_is_viewable():
    meta = {"t": {"owner_id": "A", "visibility": "public"}}
    assert _viewable(meta, "t").allowed is True
