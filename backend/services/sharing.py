"""Per-person view access for ``private`` reports.

A private report is viewable only by its owner and the people the owner has
granted, named by **verified email** (Google sign-in) or **ORCID iD** (ORCID
sign-in does not return an email). Grants live in the ``report_shares`` table;
matching happens at view time against the signed-in viewer's account.

This module also exposes the single access gate (:func:`ensure_viewable`) used by
every endpoint that serves report content. Only ``private`` reports are gated —
``public`` reports stay open by link.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import models
from . import reports as reports_service

# ORCID iDs are 16 digits in 4 groups; the final character may be a checksum "X".
_ORCID_RE = re.compile(r"^(\d{4}-\d{4}-\d{4}-\d{3}[\dX])$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class ShareError(ValueError):
    """Raised when a grantee string is not a usable email or ORCID iD."""


def parse_grantee(raw: str | None) -> tuple[str | None, str | None]:
    """Classify a grantee string into ``(email, orcid)`` (exactly one set).

    Accepts a bare ORCID iD or an ``orcid.org`` URL, otherwise a plain email.
    Raises :class:`ShareError` if it is neither.
    """
    value = (raw or "").strip()
    if not value:
        raise ShareError("Enter an email address or ORCID iD.")

    # Pull an ORCID iD out of a full ORCID URL if one was pasted.
    url_match = re.search(r"orcid\.org/(\d{4}-\d{4}-\d{4}-\d{3}[\dX])", value, re.IGNORECASE)
    candidate = url_match.group(1) if url_match else value

    orcid_match = _ORCID_RE.match(candidate.upper())
    if orcid_match:
        return None, orcid_match.group(1).upper()

    email = value.lower()
    if _EMAIL_RE.match(email):
        return email, None

    raise ShareError("That doesn't look like an email address or ORCID iD.")


def share_label(share: models.ReportShare) -> str:
    """Human label for the UI / API."""
    return share.grantee_email or share.grantee_orcid or "Unknown"


async def list_shares(db: AsyncSession, task_id: str) -> list[models.ReportShare]:
    result = await db.execute(
        select(models.ReportShare)
        .where(models.ReportShare.task_id == task_id)
        .order_by(models.ReportShare.created_at.asc())
    )
    return list(result.scalars().all())


async def add_share(db: AsyncSession, task_id: str, grantee: str) -> models.ReportShare:
    """Grant view access. Idempotent — returns the existing row if already granted."""
    email, orcid = parse_grantee(grantee)
    existing = await db.execute(
        select(models.ReportShare).where(
            models.ReportShare.task_id == task_id,
            (models.ReportShare.grantee_email == email)
            if email is not None
            else (models.ReportShare.grantee_orcid == orcid),
        )
    )
    found = existing.scalar_one_or_none()
    if found is not None:
        return found
    share = models.ReportShare(task_id=task_id, grantee_email=email, grantee_orcid=orcid)
    db.add(share)
    await db.flush()
    return share


async def remove_share(db: AsyncSession, task_id: str, share_id: str) -> bool:
    share = await db.get(models.ReportShare, share_id)
    if share is None or share.task_id != task_id:
        return False
    await db.delete(share)
    return True


async def _viewer_orcids(db: AsyncSession, user: models.User) -> set[str]:
    result = await db.execute(
        select(models.OAuthIdentity.subject).where(
            models.OAuthIdentity.user_id == user.id,
            models.OAuthIdentity.provider == "orcid",
        )
    )
    return {s.upper() for s in result.scalars().all() if s}


async def user_can_view(
    db: AsyncSession, report: models.Report, user: models.User | None
) -> bool:
    """True if ``user`` is the owner of, or has been granted access to, ``report``."""
    if user is None:
        return False
    if report.owner_id and report.owner_id == user.id:
        return True
    shares = await list_shares(db, report.task_id)
    if not shares:
        return False
    emails = {user.email.lower()} if user.email else set()
    orcids = await _viewer_orcids(db, user)
    for share in shares:
        if share.grantee_email and share.grantee_email in emails:
            return True
        if share.grantee_orcid and share.grantee_orcid.upper() in orcids:
            return True
    return False


@dataclass
class ViewVerdict:
    allowed: bool
    needs_login: bool = False  # private + anonymous: send to sign-in
    report: models.Report | None = None


async def _viewable_from_redis_metadata(request, task_id: str) -> ViewVerdict:
    """Access decision when there is no DB ``Report`` row, driven by the Redis task
    hash. Anonymous reports (no ``owner_id``) are public by link. An owned report
    with missing row is enforced from its Redis ``owner_id``/``visibility`` — it
    must not fail open just because the row is absent."""
    redis_client = getattr(request.app.state, "redis", None)
    meta = {}
    if redis_client is not None:
        try:
            meta = await redis_client.hgetall(task_id) or {}
        except Exception:  # pragma: no cover - defensive; treat as no metadata
            meta = {}
    owner_id = (meta.get("owner_id") or "").strip()
    if not owner_id:
        return ViewVerdict(allowed=True, report=None)  # anonymous → public by link
    visibility = reports_service.normalize_visibility(meta.get("visibility"))
    if visibility != "private":
        return ViewVerdict(allowed=True, report=None)
    user = getattr(request.state, "user", None)
    if user is None:
        return ViewVerdict(allowed=False, needs_login=True, report=None)
    # No row → no grantee list to consult; owner-id match is the available check.
    return ViewVerdict(allowed=(str(user.id) == owner_id), report=None)


async def ensure_viewable(request, task_id: str) -> ViewVerdict:
    """The single access gate for report-content endpoints.

    Opens its own short DB session so callers don't need a ``get_db`` dependency.

    - No accounts DB configured → allowed (degrades to pre-accounts behavior).
    - No ``Report`` row: consult the Redis task hash. A genuinely anonymous report
      (no ``owner_id`` metadata) is public by link, by design. But an OWNED report
      whose row is missing (row-creation failed, or the row was lost while Redis
      data survived) must NOT fail open — it is enforced from its Redis
      ``owner_id``/``visibility`` metadata instead.
    - ``public`` → allowed (open by link).
    - ``private`` → allowed only for the owner or a granted viewer; anonymous
      viewers get ``needs_login`` so the caller can redirect to sign-in.
    """
    sessionmaker = getattr(request.app.state, "db_sessionmaker", None)
    if sessionmaker is None:
        return ViewVerdict(allowed=True)

    async with sessionmaker() as db:
        report = await reports_service.get_report_row(db, task_id)
        if report is None:
            return await _viewable_from_redis_metadata(request, task_id)

        visibility = reports_service.normalize_visibility(report.visibility)
        if visibility != "private":
            return ViewVerdict(allowed=True, report=report)

        user = getattr(request.state, "user", None)
        if user is None:
            return ViewVerdict(allowed=False, needs_login=True, report=report)
        allowed = await user_can_view(db, report, user)
        return ViewVerdict(allowed=allowed, report=report)
