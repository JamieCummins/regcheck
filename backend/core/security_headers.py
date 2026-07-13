from __future__ import annotations

from starlette.datastructures import MutableHeaders
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
