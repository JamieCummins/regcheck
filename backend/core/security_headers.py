from __future__ import annotations

import json
from urllib.parse import quote

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Baseline Content-Security-Policy. Tuned to the app's actual resource loading:
# self-hosted assets, Google Fonts (googleapis CSS + gstatic font files), and the
# jsdelivr docsearch CSS on /jobs. Inline <script>/<style> exist on a few pages, so
# script/style allow 'unsafe-inline' (no nonces with static templates) — but only
# 'self' + inline, never external script origins. blob:/data: cover the report
# viewer's CSV download + rendered pages. No cross-origin form actions, so
# form-action stays 'self'. frame-ancestors/object-src/base-uri close the common
# clickjacking / plugin / base-injection holes. Only applied to HTML documents
# (static assets and JSON API responses don't need it).
DEFAULT_CSP = "; ".join(
    [
        "default-src 'self'",
        "script-src 'self' 'unsafe-inline'",
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net",
        "font-src 'self' https://fonts.gstatic.com data:",
        "img-src 'self' data: blob:",
        "connect-src 'self'",
        "frame-src 'self' blob:",
        "frame-ancestors 'self'",
        "base-uri 'self'",
        "object-src 'none'",
        "form-action 'self'",
    ]
)


class FrameBustMiddleware:
    """Serve a tiny frame-busting page to cross-site iframe embeds.

    ``frame-ancestors 'self'`` (below) means a cross-site ``<iframe>`` of this
    app can never render — the browser fetches the document and then blanks the
    frame. The main real-world source of such embeds is the pre-July-2026
    regcheck.app setup: a registrar "URL masking" page that iframed the
    herokuapp URL. That page was served without ``Cache-Control``, so browsers
    heuristically cache it for weeks — returning visitors keep rendering it
    from disk long after DNS moved, and see a dead frame.

    Instead of letting those embeds die, answer them (identified by the
    browser-set, unforgeable ``Sec-Fetch-Dest: iframe`` + ``Sec-Fetch-Site:
    cross-site`` fetch-metadata headers) with a minimal page that navigates the
    TOP window to the canonical site, preserving path and query. For a stale
    masking page this transparently lands the user on the real site; for a
    hostile embedder it is classic frame-busting — strictly better than a
    blank frame in both cases. Same-/own-origin frames (``about:blank`` print
    frames, ``frame-src`` blob viewers) are untouched.

    Must be registered OUTSIDE SecurityHeadersMiddleware (added after it) so
    the bust response short-circuits before frame-ancestors/X-Frame-Options
    are stamped — the bust page must itself be allowed to render in the frame.
    """

    def __init__(self, app: ASGIApp, *, canonical_base: str = "https://regcheck.app") -> None:
        self.app = app
        self.canonical_base = canonical_base.rstrip("/")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "GET"
            or Headers(scope=scope).get("sec-fetch-dest") != "iframe"
            or Headers(scope=scope).get("sec-fetch-site") != "cross-site"
        ):
            await self.app(scope, receive, send)
            return

        target = self.canonical_base + quote(scope.get("path") or "/", safe="/")
        query = (scope.get("query_string") or b"").decode("latin-1")
        if query:
            target += "?" + query
        # json.dumps + < keeps attacker-influenced path/query inert inside
        # both the <script> block and the href attribute.
        js_target = json.dumps(target).replace("<", "\\u003c")
        html_target = target.replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")
        body = (
            "<!doctype html><html><head><meta charset=\"utf-8\">"
            f"<script>top.location.replace({js_target});</script></head>"
            f"<body><a href=\"{html_target}\" target=\"_top\">Continue to RegCheck</a>"
            "</body></html>"
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"text/html; charset=utf-8"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"cache-control", b"no-store"),
                    (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"strict-origin-when-cross-origin"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


class SecurityHeadersMiddleware:
    """Attach baseline security headers to every HTTP response.

    Implemented as pure ASGI (not ``BaseHTTPMiddleware``) so it composes cleanly
    with streaming/file responses and never buffers a body.

    ``hsts`` is opt-in because ``Strict-Transport-Security`` must only be sent
    when the site is actually served over HTTPS (production). Browsers ignore it
    over plain HTTP, but gating it avoids pinning an HTTPS upgrade for
    ``localhost`` during development. ``X-Frame-Options: SAMEORIGIN`` is safe for
    the report viewer's print flow, which uses a src-less ``about:blank`` iframe
    (frame-ancestor policy only governs navigated documents).

    ``csp`` is the Content-Security-Policy string (defaults to ``DEFAULT_CSP``);
    set ``csp_report_only=True`` to emit it as ``Content-Security-Policy-Report-Only``
    instead — useful for validating a tightened policy on staging before enforcing.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        hsts: bool = False,
        csp: str | None = DEFAULT_CSP,
        csp_report_only: bool = False,
    ) -> None:
        self.app = app
        self.hsts = hsts
        self.csp = csp
        self.csp_header = (
            "Content-Security-Policy-Report-Only" if csp_report_only else "Content-Security-Policy"
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers.setdefault("X-Content-Type-Options", "nosniff")
                headers.setdefault("X-Frame-Options", "SAMEORIGIN")
                headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
                if self.hsts:
                    headers.setdefault(
                        "Strict-Transport-Security",
                        "max-age=31536000; includeSubDomains",
                    )
                # CSP only matters for HTML documents; skip it for static assets and
                # JSON so it doesn't bloat every response or constrain non-document fetches.
                if self.csp and "text/html" in headers.get("content-type", ""):
                    headers.setdefault(self.csp_header, self.csp)
            await send(message)

        await self.app(scope, receive, send_with_headers)
