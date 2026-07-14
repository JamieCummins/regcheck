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


def test_og_image_is_single_absolute_url_and_resolves(client):
    # Regression: base_url used to be prepended to the already-absolute
    # url_for(), yielding "https://hosthttps://host/static/...".
    import re
    from urllib.parse import urlparse

    html = client.get("/").text
    m = re.search(r'property="og:image" content="([^"]+)"', html)
    assert m, "og:image missing"
    url = m.group(1)
    assert url.count("://") == 1, f"malformed og:image: {url}"
    parsed = urlparse(url)
    assert parsed.scheme in ("http", "https") and parsed.netloc
    # twitter:image uses the same value.
    assert f'name="twitter:image" content="{url}"' in html
    # The asset itself serves as an image.
    asset = client.get(parsed.path)
    assert asset.status_code == 200
    assert asset.headers["content-type"].startswith("image/")


def test_robots_noindex_only_on_private_surfaces(client):
    # Public/marketing pages must be indexable (the beta-era blanket noindex
    # is gone); report-bearing surfaces stay noindexed (public-by-link).
    for path in ("/", "/team", "/faq", "/privacy", "/compare"):
        assert "noindex" not in client.get(path).text, f"{path} still noindexed"
    assert "noindex" in client.get("/next-steps/some-task").text


def test_post_run_page_avoids_survey_selectors(client):
    # The post-submit page must not use survey-named URLs/classes/ids: ad-blocker
    # annoyance lists hide them, and this page is the only path wizard -> report.
    resp = client.get("/next-steps/some-task-id")
    assert resp.status_code == 200
    html = resp.text
    assert 'class="postrun-card"' in html
    assert 'id="report-direct"' in html  # escape hatch outside the hideable card
    assert 'class="survey' not in html and 'id="survey' not in html
    assert 'id="step-survey"' not in html and 'id="to-survey"' not in html
    # Legacy URL still lands somewhere sensible.
    legacy = client.get("/survey/some-task-id", follow_redirects=False)
    assert legacy.status_code == 301
    assert legacy.headers["location"].endswith("/next-steps/some-task-id")


def test_wizard_ships_extension_fallback_note(client):
    # The content-blocker fallback: present in markup, armed (hidden) by
    # wizard.js at runtime, revealed by CSS delay if the script never runs.
    html = client.get("/compare").text
    assert 'id="wizard-fallback-note"' in html
    assert "content blocker" in html
