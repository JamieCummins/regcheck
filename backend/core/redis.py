from __future__ import annotations

import os

from redis import asyncio as aioredis


def _tls_insecure() -> bool:
    """Opt-OUT of certificate verification for managed Redis providers that ship
    self-signed per-instance certs (e.g. Heroku Data for Redis). Verification is
    the DEFAULT; disabling it is an explicit, documented deployment choice
    (REDIS_TLS_INSECURE=1)."""
    return (os.environ.get("REDIS_TLS_INSECURE") or "").strip().lower() in {"1", "true", "yes", "on"}


def create_redis_client(
    redis_url: str,
    *,
    socket_connect_timeout: float | None = 5,
    socket_timeout: float | None = 5,
):
    """Create an asyncio Redis client from a configured URL."""
    kwargs = {
        "decode_responses": True,
        # Make connection issues fail fast and allow redis-py to retry on timeouts.
        "socket_connect_timeout": socket_connect_timeout,
        # NOTE: For blocking commands (e.g. BRPOPLPUSH), socket_timeout must be
        # greater than the command's server-side timeout to avoid false timeouts.
        "socket_timeout": socket_timeout,
        "retry_on_timeout": True,
        # Keep connections warm to reduce first-request failures after idling.
        "health_check_interval": 30,
    }
    if redis_url.startswith("rediss://") and _tls_insecure():
        return aioredis.from_url(redis_url, ssl_cert_reqs=None, **kwargs)
    # rediss:// without the opt-out gets the library's verified-TLS default.
    return aioredis.from_url(redis_url, **kwargs)
