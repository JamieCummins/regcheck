from __future__ import annotations

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send


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
    """

    def __init__(self, app: ASGIApp, *, hsts: bool = False) -> None:
        self.app = app
        self.hsts = hsts

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
            await send(message)

        await self.app(scope, receive, send_with_headers)
