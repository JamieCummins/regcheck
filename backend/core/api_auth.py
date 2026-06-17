"""Bearer-token authentication for the keyed public API (/api/v1).

A request must carry a RegCheck API key (issued by a signed-in user) as
`Authorization: Bearer rc_live_…` (or `X-API-Key: …`). Keys are stored as
SHA-256 hashes; each successful call bumps the key's usage counter.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import Depends, HTTPException, Request
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import models
from ..db.session import get_db
from .security import hash_api_key, looks_like_api_key

_UNAUTHORIZED_HEADERS = {"WWW-Authenticate": "Bearer"}


def _extract_token(request: Request) -> str:
    auth = request.headers.get("Authorization") or ""
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip()
    return (request.headers.get("X-API-Key") or "").strip()


async def require_api_key(
    request: Request, db: AsyncSession = Depends(get_db)
) -> models.User:
    token = _extract_token(request)
    if not looks_like_api_key(token):
        raise HTTPException(
            status_code=401,
            detail="Missing or malformed API key. Send 'Authorization: Bearer rc_live_…'.",
            headers=_UNAUTHORIZED_HEADERS,
        )
    result = await db.execute(
        select(models.ApiKey).where(models.ApiKey.key_hash == hash_api_key(token))
    )
    key = result.scalar_one_or_none()
    if key is None or key.revoked_at is not None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or revoked API key.",
            headers=_UNAUTHORIZED_HEADERS,
        )
    user = await db.get(models.User, key.user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="API key owner not found.", headers=_UNAUTHORIZED_HEADERS)

    # Bump usage atomically (request_count = request_count + 1) so concurrent
    # requests on the same key don't lose increments via read-modify-write.
    await db.execute(
        update(models.ApiKey)
        .where(models.ApiKey.id == key.id)
        .values(
            request_count=models.ApiKey.request_count + 1,
            last_used_at=datetime.now(timezone.utc),
        )
    )
    await db.commit()

    request.state.api_user = user
    request.state.api_key = key
    return user
