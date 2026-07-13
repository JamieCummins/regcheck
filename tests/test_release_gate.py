"""Release-gate surface: /ready readiness probe and the OpenAPI/docs polish
(version source, 202 on compare, Bearer security scheme, self-hosted Swagger)."""
from __future__ import annotations

import os

import pytest


class _FakeRedis:
    def __init__(self, *, ping_ok=True, heartbeats=1):
        self.ping_ok = ping_ok
        self.heartbeats = heartbeats

    async def ping(self):
        if not self.ping_ok:
            raise ConnectionError("redis down")
        return True

    async def keys(self, pattern):
        assert pattern == "worker:heartbeat:*"
        return [f"worker:heartbeat:{i}" for i in range(self.heartbeats)]

    async def hgetall(self, k):
        return {}

    async def get(self, k):
        return None


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("release")
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{tmp}/release.db"
    os.environ["SESSION_SECRET"] = "release-secret"
    import backend.core.config as cfg

    cfg.get_settings.cache_clear()
    from starlette.testclient import TestClient

    from backend import create_app

    app = create_app()
    app.state.redis = _FakeRedis()
    with TestClient(app) as c:
        yield c, app
    cfg.get_settings.cache_clear()


# ── /ready ─────────────────────────────────────────────────────────────────────


def test_ready_ok_when_all_dependencies_up(client):
    c, app = client
    app.state.redis = _FakeRedis(ping_ok=True, heartbeats=2)
    resp = c.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["checks"]["redis"] == "ok"
    assert body["checks"]["database"] == "ok"
    assert body["checks"]["worker"].startswith("ok")


def test_ready_503_without_worker_heartbeat(client):
    c, app = client
    app.state.redis = _FakeRedis(ping_ok=True, heartbeats=0)
    resp = c.get("/ready")
    assert resp.status_code == 503
    assert "no live worker heartbeat" in resp.json()["checks"]["worker"]


def test_ready_503_when_redis_down_and_no_internals_leaked(client):
    c, app = client
    app.state.redis = _FakeRedis(ping_ok=False)
    resp = c.get("/ready")
    assert resp.status_code == 503
    checks = resp.json()["checks"]
    assert checks["redis"] == "error: ConnectionError"  # class name only, no detail
    assert checks["worker"] == "unknown: redis unavailable"
    assert "redis down" not in resp.text  # exception text must not leak


def test_health_stays_dependency_free(client):
    c, app = client
    app.state.redis = _FakeRedis(ping_ok=False)  # even with redis down...
    assert c.get("/health").json() == {"status": "ok"}  # ...liveness is green


# ── OpenAPI + self-hosted docs ─────────────────────────────────────────────────


def test_openapi_version_matches_single_source(client):
    c, _app = client
    from backend.main import APP_VERSION

    spec = c.get("/openapi.json").json()
    assert spec["info"]["title"] == "RegCheck"
    assert spec["info"]["version"] == APP_VERSION


def test_api_compare_documents_202_and_bearer_security(client):
    c, _app = client
    spec = c.get("/openapi.json").json()
    compare = spec["paths"]["/api/v1/compare"]["post"]
    assert "202" in compare["responses"]
    assert {"RegCheckApiKey": []} in compare.get("security", [])
    scheme = spec["components"]["securitySchemes"]["RegCheckApiKey"]
    assert scheme["type"] == "http"
    assert scheme["scheme"] == "bearer"


def test_docs_served_from_self_hosted_assets(client):
    c, _app = client
    resp = c.get("/docs")
    assert resp.status_code == 200
    assert "/static/vendor/swagger-ui/swagger-ui-bundle.js" in resp.text
    assert "cdn.jsdelivr.net" not in resp.text  # CSP would block the CDN build


def test_swagger_assets_exist_on_disk():
    from pathlib import Path

    vendor = Path(__file__).resolve().parents[1] / "static" / "vendor" / "swagger-ui"
    assert (vendor / "swagger-ui-bundle.js").stat().st_size > 100_000
    assert (vendor / "swagger-ui.css").stat().st_size > 10_000
