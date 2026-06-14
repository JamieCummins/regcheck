from __future__ import annotations

import re
import secrets

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.oauth import OAuthIdentityInfo
from ..db import models

_SLUG_RE = re.compile(r"[^a-z0-9]+")


async def get_user(session: AsyncSession, user_id: str | None) -> models.User | None:
    if not user_id:
        return None
    return await session.get(models.User, user_id)


async def find_user_by_email(session: AsyncSession, email: str | None) -> models.User | None:
    if not email:
        return None
    result = await session.execute(select(models.User).where(models.User.email == email))
    return result.scalar_one_or_none()


async def find_identity(
    session: AsyncSession, provider: str, subject: str
) -> models.OAuthIdentity | None:
    result = await session.execute(
        select(models.OAuthIdentity).where(
            models.OAuthIdentity.provider == provider,
            models.OAuthIdentity.subject == subject,
        )
    )
    return result.scalar_one_or_none()


def _slugify(value: str | None) -> str:
    base = _SLUG_RE.sub("-", (value or "").strip().lower()).strip("-")
    return base[:40] or "user"


async def generate_unique_handle(session: AsyncSession, base: str | None) -> str:
    candidate = _slugify(base)
    suffix = 0
    while True:
        handle = candidate if suffix == 0 else f"{candidate}-{suffix}"
        existing = await session.execute(
            select(models.User.id).where(models.User.handle == handle)
        )
        if existing.scalar_one_or_none() is None:
            return handle
        suffix += 1
        if suffix > 50:  # extreme collision; fall back to random
            return f"{candidate}-{secrets.token_hex(3)}"


async def upsert_oauth_user(
    session: AsyncSession,
    identity: OAuthIdentityInfo,
    *,
    link_to_user: models.User | None = None,
) -> models.User:
    """Find or create the user for an OAuth identity.

    - If the (provider, subject) identity already exists, return its user.
    - If `link_to_user` is given (a logged-in user adding a provider), link to it.
    - Otherwise match by *verified* email, else create a new user.
    """
    if not identity.subject:
        raise ValueError("OAuth identity is missing a stable subject id")

    existing = await find_identity(session, identity.provider, identity.subject)
    if existing is not None:
        user = await session.get(models.User, existing.user_id)
        if user is not None:
            return user

    if link_to_user is not None:
        user = link_to_user
    else:
        user = None
        if identity.email and identity.email_verified:
            user = await find_user_by_email(session, identity.email)
        if user is None:
            handle_base = identity.display_name or (identity.email.split("@")[0] if identity.email else None)
            # Only claim an email on the account when the provider verified it,
            # so an unverified identity can't collide with / take over another
            # account's email (the raw email is still recorded on the identity).
            account_email = identity.email if identity.email_verified else None
            user = models.User(
                email=account_email,
                display_name=identity.display_name,
                handle=await generate_unique_handle(session, handle_base),
            )
            session.add(user)
            await session.flush()

    session.add(
        models.OAuthIdentity(
            user_id=user.id,
            provider=identity.provider,
            subject=identity.subject,
            email=identity.email,
        )
    )
    await session.flush()
    return user


async def update_profile(
    session: AsyncSession,
    user: models.User,
    *,
    display_name: str | None = None,
    handle: str | None = None,
    is_public_profile: bool | None = None,
    use_case: str | None = None,
    academic_position: str | None = None,
    research_field: str | None = None,
) -> models.User:
    if display_name is not None:
        user.display_name = display_name.strip() or None
    if handle and handle.strip():
        desired = _slugify(handle)
        if desired != user.handle:
            taken = await session.execute(
                select(models.User.id).where(
                    models.User.handle == desired, models.User.id != user.id
                )
            )
            user.handle = desired if taken.scalar_one_or_none() is None else await generate_unique_handle(session, desired)
    if is_public_profile is not None:
        user.is_public_profile = is_public_profile
    if use_case is not None:
        user.use_case = use_case.strip() or None
    if academic_position is not None:
        user.academic_position = academic_position.strip() or None
    if research_field is not None:
        user.research_field = research_field.strip() or None
    await session.flush()
    return user


# ── API keys ─────────────────────────────────────────────────────────────────
from datetime import datetime, timezone  # noqa: E402

from ..core import security  # noqa: E402


async def create_api_key(session: AsyncSession, user: models.User, name: str | None):
    """Create an API key for a user. Returns (ApiKey, plaintext) — the plaintext
    is shown to the user exactly once and never stored."""
    raw = security.generate_api_key()
    key = models.ApiKey(
        user_id=user.id,
        name=(name or "").strip() or None,
        prefix=security.api_key_display_prefix(raw),
        key_hash=security.hash_api_key(raw),
    )
    session.add(key)
    await session.flush()
    return key, raw


async def list_api_keys(session: AsyncSession, user_id: str) -> list[models.ApiKey]:
    result = await session.execute(
        select(models.ApiKey)
        .where(models.ApiKey.user_id == user_id)
        .order_by(models.ApiKey.created_at.desc())
    )
    return list(result.scalars().all())


async def revoke_api_key(session: AsyncSession, user_id: str, key_id: str) -> bool:
    key = await session.get(models.ApiKey, key_id)
    if key is None or key.user_id != user_id or key.revoked_at is not None:
        return False
    key.revoked_at = datetime.now(timezone.utc)
    return True
