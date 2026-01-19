from __future__ import annotations

from redis import asyncio as aioredis


def create_redis_client(redis_url: str):
    """Create an asyncio Redis client from a configured URL."""
    if redis_url.startswith("rediss://"):
        return aioredis.from_url(redis_url, decode_responses=True, ssl_cert_reqs=None)
    return aioredis.from_url(redis_url, decode_responses=True)
