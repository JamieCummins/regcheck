"""APP_MODE=precheck: same codebase, second site. The PreCheck deployment must
serve ONLY the registration-quality tool (+ shared account/report plumbing)
under its own brand, and the default RegCheck mode must be byte-for-byte
unaffected."""
from __future__ import annotations

import os

import pytest


class _FakeRedis:
    async def ping(self):
        return True

    async def hgetall(self, k):
        return {}

    async def get(self, k):
        return None

    async def exists(self, k):
        return 0

    async def keys(self, pattern):
        return ["worker:heartbeat:x"]


def _build_client(tmp_path, mode: str | None):
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{tmp_path}/pc.db"
    os.environ["SESSION_SECRET"] = "pc-secret"
    if mode is None:
        os.environ.pop("APP_MODE", None)
    else:
        os.environ["APP_MODE"] = mode
    import backend.core.config as cfg

    cfg.get_settings.cache_clear()
    from starlette.testclient import TestClient

    from backend import create_app

    app = create_app()
    app.state.redis = _FakeRedis()
    return TestClient(app)


@pytest.fixture()
def precheck(tmp_path):
    with _build_client(tmp_path, "precheck") as c:
        yield c
    os.environ.pop("APP_MODE", None)
    import backend.core.config as cfg

    cfg.get_settings.cache_clear()


@pytest.fixture()
def regcheck(tmp_path):
    with _build_client(tmp_path, None) as c:
        yield c
    import backend.core.config as cfg

    cfg.get_settings.cache_clear()


# ── route gating ───────────────────────────────────────────────────────────────

GATED = [
    "/compare",
    "/clinical_trials",
    "/general_preregistration",
    "/animals_trials",
    "/demo",
    "/faq",
    "/api",
    "/api/v1/compare",
    "/docs",
    "/openapi.json",
    "/coming-soon/code-paper",
    "/jobs",
]

ALLOWED = [
    "/",
    "/evaluate_registration",
    "/login",
    "/privacy",
    "/contact",
    "/team",
    "/health",
]


def test_precheck_gates_comparison_suite(precheck):
    for path in GATED:
        resp = precheck.get(path, follow_redirects=False)
        assert resp.status_code == 404, f"{path} -> {resp.status_code}"


def test_precheck_gate_covers_post_routes(precheck):
    resp = precheck.post("/compare", data={}, follow_redirects=False)
    assert resp.status_code == 404
    resp = precheck.post("/api/v1/compare", data={}, follow_redirects=False)
    assert resp.status_code == 404


def test_precheck_serves_quality_surface(precheck):
    for path in ALLOWED:
        resp = precheck.get(path)
        assert resp.status_code == 200, f"{path} -> {resp.status_code}"


def test_precheck_gated_404_keeps_security_headers(precheck):
    resp = precheck.get("/compare")
    assert resp.status_code == 404
    assert "x-content-type-options" in {k.lower() for k in resp.headers}


# ── branding ───────────────────────────────────────────────────────────────────


def test_precheck_branding_on_served_pages(precheck):
    for path in ("/", "/evaluate_registration", "/login", "/privacy"):
        text = precheck.get(path).text
        assert "PreCheck" in text, path
        # The navbar wordmark and page copy must not leak the RegCheck brand.
        assert "<span>RegCheck</span>" not in text, path
        assert "RegCheck Privacy" not in text, path


def test_precheck_landing_has_no_comparison_nav(precheck):
    text = precheck.get("/").text
    assert "Evaluate a registration" in text
    assert "Registration-Paper Comparison" not in text
    assert "Example report" not in text
    assert 'href="/compare"' not in text


def test_precheck_footer_drops_gated_links(precheck):
    text = precheck.get("/").text
    assert 'href="/faq"' not in text
    assert ">API<" not in text


# ── default mode unaffected ────────────────────────────────────────────────────


def test_regcheck_mode_unchanged(regcheck):
    assert regcheck.get("/compare").status_code == 200
    assert regcheck.get("/demo").status_code == 200
    assert regcheck.get("/openapi.json").status_code == 200
    home = regcheck.get("/").text
    assert "<span>RegCheck</span>" in home
    assert "Compare study plans with final papers." in home


def test_unknown_app_mode_falls_back_to_regcheck(tmp_path):
    with _build_client(tmp_path, "nonsense") as c:
        assert c.get("/compare").status_code == 200
        assert "<span>RegCheck</span>" in c.get("/").text
    os.environ.pop("APP_MODE", None)
    import backend.core.config as cfg

    cfg.get_settings.cache_clear()
