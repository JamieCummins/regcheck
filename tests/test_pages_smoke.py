"""Render every public HTML page once.

The rest of the suite exercises services and APIs but rarely renders templates,
so a broken TemplateResponse call (e.g. the pre-starlette-1.x signature) can slip
through with a single point of failure. This smoke pass fails loudly instead."""
from __future__ import annotations

import os

import pytest


class FakeRedis:
    async def ping(self):
        return True

    async def hgetall(self, k):
        return {}

    async def get(self, k):
        return None

    async def exists(self, k):
        return 0


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pages")
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{tmp}/pages.db"
    os.environ["SESSION_SECRET"] = "pages-secret"
    import backend.core.config as cfg

    cfg.get_settings.cache_clear()
    from starlette.testclient import TestClient

    from backend import create_app

    app = create_app()
    app.state.redis = FakeRedis()
    with TestClient(app) as c:
        yield c
    cfg.get_settings.cache_clear()


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/compare",
        "/evaluate_registration",
        "/contact",
        "/demo",
        "/team",
        "/jobs",
        "/privacy",
        "/faq",
        "/api",
        "/login",
        "/coming-soon/code-paper",
        "/coming-soon/evaluate-registration",
    ],
)
def test_page_renders(client, path):
    resp = client.get(path)
    assert resp.status_code == 200, f"{path} -> {resp.status_code}"
    assert "<html" in resp.text.lower()


def test_legacy_wizard_paths_redirect_to_compare(client):
    for path in ("/clinical_trials", "/general_preregistration"):
        resp = client.get(path, follow_redirects=False)
        assert resp.status_code == 302
        assert resp.headers["location"].endswith("/compare")
