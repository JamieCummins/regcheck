from __future__ import annotations

from redis import asyncio as aioredis


def create_redis_client(redis_url: str):
    """Create an asyncio Redis client from a configured URL."""
    kwargs = {
        "decode_responses": True,
        # Make connection issues fail fast and allow redis-py to retry on timeouts.
        "socket_connect_timeout": 5,
        "socket_timeout": 5,
        "retry_on_timeout": True,
        # Keep connections warm to reduce first-request failures after idling.
        "health_check_interval": 30,
    }
    if redis_url.startswith("rediss://"):
        return aioredis.from_url(redis_url, ssl_cert_reqs=None, **kwargs)
    return aioredis.from_url(redis_url, **kwargs)
