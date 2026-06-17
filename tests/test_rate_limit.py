import pytest
from fastapi import Depends, FastAPI
from limits import parse
from limits.storage import MemoryStorage
from limits.strategies import MovingWindowRateLimiter
from starlette.testclient import TestClient

import backend.core.rate_limit as rl


@pytest.fixture
def limiter_state():
    """Save/restore the module-level limiter globals so toggling enabled/limit in
    one test never leaks into others (other tests POST to rate-limited routes)."""
    saved = (rl._ENABLED, rl._STORAGE, rl._STRATEGY, rl._COMPARISON_ITEM)
    try:
        yield
    finally:
        rl._ENABLED, rl._STORAGE, rl._STRATEGY, rl._COMPARISON_ITEM = saved


def _client() -> TestClient:
    app = FastAPI()

    @app.get("/go", dependencies=[Depends(rl.comparison_rate_limit)])
    async def go():
        return {"ok": True}

    return TestClient(app)


def test_disabled_is_passthrough(limiter_state):
    rl._ENABLED = False
    c = _client()
    codes = [c.get("/go", headers={"x-forwarded-for": "9.9.9.9"}).status_code for _ in range(20)]
    assert set(codes) == {200}


def test_enabled_blocks_after_limit_per_ip(limiter_state):
    rl._ENABLED = True
    rl._STORAGE = MemoryStorage()
    rl._STRATEGY = MovingWindowRateLimiter(rl._STORAGE)
    rl._COMPARISON_ITEM = parse("3/minute")
    c = _client()

    a = [c.get("/go", headers={"x-forwarded-for": "1.1.1.1, 10.0.0.1"}).status_code for _ in range(4)]
    assert a == [200, 200, 200, 429]
    # A different client IP is unaffected by A's usage.
    assert c.get("/go", headers={"x-forwarded-for": "2.2.2.2"}).status_code == 200


def test_client_key_prefers_left_most_forwarded_for():
    class Req:
        headers = {"x-forwarded-for": "203.0.113.5, 10.0.0.1"}
        client = None

    assert rl._client_key(Req()) == "203.0.113.5"
