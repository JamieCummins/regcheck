from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.testclient import TestClient

from backend.core.security_headers import FrameBustMiddleware, SecurityHeadersMiddleware


def _client(*, frame_bust: bool = False, canonical_base: str = "https://regcheck.app", **kwargs) -> TestClient:
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware, **kwargs)
    if frame_bust:
        # Mirrors create_app: FrameBust added after SecurityHeaders → outermost.
        app.add_middleware(FrameBustMiddleware, canonical_base=canonical_base)

    @app.get("/page", response_class=HTMLResponse)
    def page() -> str:
        return "<html><body>hi</body></html>"

    @app.get("/api")
    def api() -> JSONResponse:
        return JSONResponse({"ok": True})

    return TestClient(app)


def test_baseline_headers_and_csp_on_html():
    resp = _client(hsts=True).get("/page")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "SAMEORIGIN"
    assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert "Strict-Transport-Security" in resp.headers
    csp = resp.headers["Content-Security-Policy"]
    for directive in ("default-src 'self'", "frame-ancestors 'self'", "object-src 'none'", "base-uri 'self'"):
        assert directive in csp
    # Google Fonts + jsdelivr are allowed for styles/fonts (real app resources).
    assert "https://fonts.googleapis.com" in csp and "https://fonts.gstatic.com" in csp


def test_csp_applied_to_html_only_not_json():
    client = _client()
    assert "Content-Security-Policy" in client.get("/page").headers
    assert "Content-Security-Policy" not in client.get("/api").headers


def test_csp_report_only_mode():
    resp = _client(csp_report_only=True).get("/page")
    assert "Content-Security-Policy-Report-Only" in resp.headers
    assert "Content-Security-Policy" not in resp.headers


# ── frame busting (cross-site iframe embeds, e.g. the cached pre-2026 masking page) ──

_IFRAME = {"Sec-Fetch-Dest": "iframe", "Sec-Fetch-Site": "cross-site"}


def test_cross_site_iframe_gets_bust_page_without_frame_headers():
    resp = _client(frame_bust=True).get("/page", headers=_IFRAME)
    assert resp.status_code == 200
    assert 'top.location.replace("https://regcheck.app/page")' in resp.text
    # The bust page must itself be frameable: no CSP / XFO on this response.
    assert "Content-Security-Policy" not in resp.headers
    assert "X-Frame-Options" not in resp.headers
    assert resp.headers["Cache-Control"] == "no-store"


def test_bust_preserves_path_and_query():
    resp = _client(frame_bust=True).get("/report/abc123?view=evidence&x=1", headers=_IFRAME)
    assert 'top.location.replace("https://regcheck.app/report/abc123?view=evidence&x=1")' in resp.text


def test_bust_target_is_escaped_against_injection():
    resp = _client(frame_bust=True).get(
        '/x/"</script><script>alert(1)</script>', headers=_IFRAME
    )
    assert resp.status_code == 200
    assert "<script>alert(1)</script>" not in resp.text
    # Exactly the one legitimate script block.
    assert resp.text.count("<script>") == 1


def test_same_or_own_origin_iframes_and_navigations_pass_through():
    client = _client(frame_bust=True)
    # Same-origin frame (e.g. the report viewer's print frame): normal response.
    same = client.get("/page", headers={"Sec-Fetch-Dest": "iframe", "Sec-Fetch-Site": "same-origin"})
    assert "top.location.replace" not in same.text
    assert "Content-Security-Policy" in same.headers
    # Plain navigation: untouched.
    nav = client.get("/page")
    assert nav.text == "<html><body>hi</body></html>"
    # Non-GET is never intercepted, even with iframe fetch metadata.
    post = client.post("/page", headers=_IFRAME)
    assert post.status_code == 405


def test_bust_respects_canonical_base_override():
    resp = _client(frame_bust=True, canonical_base="https://staging.example.org/").get(
        "/page", headers=_IFRAME
    )
    assert 'top.location.replace("https://staging.example.org/page")' in resp.text
