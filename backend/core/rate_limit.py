from __future__ import annotations

import os

from fastapi import HTTPException, Request
from limits import parse
from limits.storage import MemoryStorage
from limits.strategies import MovingWindowRateLimiter

from .config import get_settings

# Per-IP cap on cost-bearing comparison submissions (LLM + embeddings spend).
# Configurable via env; sensible default.
COMPARISON_RATE_LIMIT = (os.environ.get("COMPARISON_RATE_LIMIT") or "10/minute").strip()

# Gated to production: disabled outside Heroku/HTTPS so local development and the
# test suite (which reuse a single client IP) are never throttled.
_ENABLED = get_settings().is_production

# In-memory storage = per worker process (see review notes on the multi-process
# multiplier). Moving window for smoother enforcement than fixed buckets.
_STORAGE = MemoryStorage()
_STRATEGY = MovingWindowRateLimiter(_STORAGE)
_COMPARISON_ITEM = parse(COMPARISON_RATE_LIMIT)


def _client_key(request: Request) -> str:
    """Identify the caller for rate limiting.

    Behind Heroku's router the socket peer is the router, not the user, so the
    real client is the left-most entry of ``X-Forwarded-For`` (Heroku sets this
    to the originating IP). Falls back to the direct peer address off-platform.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    client = request.client
    return client.host if client else "anonymous"


async def comparison_rate_limit(request: Request) -> None:
    """FastAPI dependency: per-IP throttle for cost-bearing submissions.

    Implemented as a dependency (not slowapi's decorator) so it never rewrites
    the endpoint signature — the decorator is incompatible with our
    File/UploadFile parameters under ``from __future__ import annotations``.
    """
    if not _ENABLED:
        return
    if not _STRATEGY.hit(_COMPARISON_ITEM, "compare", _client_key(request)):
        raise HTTPException(
            status_code=429,
            detail="Too many comparison requests. Please wait a moment and try again.",
        )
