from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.testclient import TestClient

from backend.core.security_headers import SecurityHeadersMiddleware


def _client(**kwargs) -> TestClient:
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware, **kwargs)

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
