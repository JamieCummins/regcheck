"""Report ownership/visibility metadata + lifecycle.

The durable ownership record lives in Postgres (``reports`` table); the report
content (items, manifest, evidence) lives in Redis under ``task_id``. Anonymous
runs are Redis-only (no row here) and expire on a short TTL.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import models
from .report_artifacts import delete_report_artifacts, migrate_artifacts_to_persist

logger = logging.getLogger(__name__)

VISIBILITIES = {"public", "private"}
DEFAULT_VISIBILITY = "private"
# Two-tier model: public (listed + open by link) vs private (allow-list; only the
# owner and people they grant, who must sign in). Earlier vocabularies collapse
# here — "restricted" was the allow-list tier, "unlisted" was link-only (now
# folded into private so nothing that wasn't public stays openly linkable).
_VISIBILITY_ALIASES = {"restricted": "private", "unlisted": "private"}


def _clean_title(raw: str | None) -> str:
    text = (raw or "").replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:200]


def generate_default_title(
    *,
    comparison_type: str | None = None,
    paper_filename: str | None = None,
    registration_id: str | None = None,
) -> str:
    """Auto-generate a human-friendly report name (users can rename later)."""
    stem = _clean_title(Path(paper_filename).stem) if paper_filename else ""
    if not stem and registration_id:
        stem = _clean_title(registration_id)
    if stem:
        return stem
    return "Comparison " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


def normalize_visibility(value: str | None) -> str:
    v = (value or "").strip().lower()
    v = _VISIBILITY_ALIASES.get(v, v)
    return v if v in VISIBILITIES else DEFAULT_VISIBILITY


async def create_report_row(
    db: AsyncSession,
    *,
    task_id: str,
    owner_id: str | None,
    visibility: str,
    title: str | None,
    comparison_type: str | None,
    source: str = "ui",
) -> models.Report:
    report = models.Report(
        task_id=task_id,
        owner_id=owner_id,
        visibility=normalize_visibility(visibility),
        title=title,
        comparison_type=comparison_type,
        source=source,
    )
    db.add(report)
    return report


async def get_report_row(db: AsyncSession, task_id: str) -> models.Report | None:
    return await db.get(models.Report, task_id)


async def filter_session_manageable(db: AsyncSession, task_ids: list[str]) -> list[str]:
    """Of ``task_ids``, keep only those a browser session may still manage — i.e.
    with no owned DB row. Used after login to drop any session handles to reports
    now owned by an account (this user's are on the dashboard; others' are not the
    session's to touch)."""
    keep: list[str] = []
    for task_id in task_ids:
        row = await get_report_row(db, task_id)
        if row is None or not (row.owner_id or "").strip():
            keep.append(task_id)
    return keep


async def claim_anonymous_reports(
    redis_client,
    db: AsyncSession,
    *,
    owner_id: str,
    task_ids: list[str],
    visibility: str = DEFAULT_VISIBILITY,
) -> list[str]:
    """Assign anonymous reports (created in this browser session) to a user who has
    just signed in: flip the Redis hash to owned + persistent, migrate evidence off
    the auto-expiring temp bucket, and create the durable ownership row. Skips
    reports that are gone/expired or already owned. Best effort per report; returns
    the task_ids actually claimed. Caller commits the DB session."""
    claimed: list[str] = []
    vis = normalize_visibility(visibility)
    for task_id in task_ids:
        try:
            existing = await get_report_row(db, task_id)
            if existing is not None and (existing.owner_id or "").strip():
                # Owned by someone. If that someone is THIS user, treat as claimed
                # (so it's pruned from the session list); otherwise leave it alone
                # but it will still be pruned below (a session must not retain a
                # handle to a report owned by another account).
                if existing.owner_id == owner_id:
                    claimed.append(task_id)
                continue
            meta = await redis_client.hgetall(task_id)
            if not meta:
                continue  # expired or never existed
            if (meta.get("owner_id") or "").strip():
                continue  # already owned (defensive)
            await redis_client.hset(
                task_id,
                mapping={"owner_id": owner_id, "visibility": vis, "retention": "persist"},
            )
            await redis_client.persist(task_id)
            await migrate_artifacts_to_persist(redis_client, task_id)
            if existing is None:
                await create_report_row(
                    db,
                    task_id=task_id,
                    owner_id=owner_id,
                    visibility=vis,
                    title=meta.get("title"),
                    comparison_type=meta.get("comparison_type"),
                    source="ui",
                )
            else:
                existing.owner_id = owner_id
                existing.visibility = vis
            claimed.append(task_id)
        except Exception:  # pragma: no cover - per-report best effort
            logger.warning("Failed to claim report %s on login", task_id, exc_info=True)
            continue
    return claimed


async def list_reports_for_owner(db: AsyncSession, owner_id: str) -> list[models.Report]:
    result = await db.execute(
        select(models.Report)
        .where(models.Report.owner_id == owner_id)
        .order_by(models.Report.created_at.desc())
    )
    return list(result.scalars().all())


async def list_public_reports_for_owner(db: AsyncSession, owner_id: str) -> list[models.Report]:
    result = await db.execute(
        select(models.Report)
        .where(models.Report.owner_id == owner_id, models.Report.visibility == "public")
        .order_by(models.Report.created_at.desc())
    )
    return list(result.scalars().all())


async def rename_report(redis_client, db: AsyncSession, report: models.Report, title: str) -> str:
    cleaned = _clean_title(title) or report.title or "Untitled report"
    report.title = cleaned
    # Mirror onto the Redis hash so the viewer shows it without a DB read.
    try:
        await redis_client.hset(report.task_id, mapping={"title": cleaned})
    except Exception as exc:  # pragma: no cover - best effort mirror
        logger.warning("Failed to mirror report title to Redis", exc_info=exc)
    return cleaned


async def set_report_visibility(
    redis_client, db: AsyncSession, report: models.Report, visibility: str
) -> str:
    vis = normalize_visibility(visibility)
    report.visibility = vis
    try:
        await redis_client.hset(report.task_id, mapping={"visibility": vis})
    except Exception as exc:  # pragma: no cover - best effort mirror
        logger.warning("Failed to mirror report visibility to Redis", exc_info=exc)
    return vis


# A deletion tombstone lives this long: comfortably longer than the slowest job,
# so a worker that finishes after a delete still sees it and cleans up.
DELETION_TOMBSTONE_TTL_SECONDS = 3 * 60 * 60


def deletion_tombstone_key(task_id: str) -> str:
    return f"deleted:{task_id}"


async def mark_deleted(redis_client, task_id: str) -> None:
    """Record a deletion tombstone so an in-flight worker won't resurrect the
    report after it's been deleted (see worker job-completion guard)."""
    try:
        await redis_client.set(
            deletion_tombstone_key(task_id), "1", ex=DELETION_TOMBSTONE_TTL_SECONDS
        )
    except Exception:  # pragma: no cover - best-effort
        logger.warning("Failed to set deletion tombstone for %s", task_id, exc_info=True)


async def is_deleted(redis_client, task_id: str) -> bool:
    try:
        return bool(await redis_client.exists(deletion_tombstone_key(task_id)))
    except Exception:  # pragma: no cover - fail open on the CHECK is safe: a missed
        # tombstone just means the worker finishes normally, and delete already
        # removed the content; the guard is belt-and-braces.
        return False


async def delete_report_everywhere(
    redis_client,
    db: AsyncSession | None,
    task_id: str,
    report: models.Report | None = None,
) -> None:
    """The single, complete report-deletion path (used by UI, API, and account
    erasure): tombstone + retained uploads + evidence artifacts + Redis hash +
    survey record + ownership row. Idempotent and best-effort per resource."""
    await mark_deleted(redis_client, task_id)
    await _cleanup_retained_uploads(redis_client, task_id)
    await delete_report_artifacts(redis_client, task_id)
    try:
        await redis_client.delete(task_id, f"survey:{task_id}")
        await redis_client.srem("survey:task_ids", task_id)
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Failed to delete report content from Redis", exc_info=exc)
    if db is not None and report is not None:
        await db.delete(report)


async def _cleanup_retained_uploads(redis_client, task_id: str) -> None:
    """Remove the source uploads kept for regeneration (Redis blobs + S3 objects).
    Lives here (not in the route) so UI and API deletion share one implementation."""
    import json as _json

    from ..core.storage import get_s3_config, s3_delete_key

    s3_keys: dict = {}
    try:
        raw = await redis_client.hget(task_id, "regen_job")
        if raw:
            payload = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
            s3_keys = (_json.loads(payload).get("s3_keys") or {})
    except Exception:  # pragma: no cover - defensive
        s3_keys = {}
    for suffix in ("paper", "prereg", "csv"):
        try:
            await redis_client.delete(f"upload:{task_id}:{suffix}")
        except Exception:  # pragma: no cover - best-effort
            pass
    cfg = get_s3_config()
    if cfg is not None:
        import asyncio as _asyncio

        for key in s3_keys.values():
            if key:
                try:
                    await _asyncio.to_thread(s3_delete_key, cfg, key=key)
                except Exception:  # pragma: no cover - best-effort
                    pass
