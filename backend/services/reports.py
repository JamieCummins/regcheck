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
from .report_artifacts import delete_report_artifacts

logger = logging.getLogger(__name__)

VISIBILITIES = {"public", "unlisted", "restricted"}
DEFAULT_VISIBILITY = "unlisted"
# Legacy value: "private" historically meant "unlisted, link-only".
_VISIBILITY_ALIASES = {"private": "unlisted"}


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


async def delete_report_everywhere(
    redis_client,
    db: AsyncSession | None,
    task_id: str,
    report: models.Report | None = None,
) -> None:
    """Remove a report's content (Redis hash + evidence artifacts + survey) and
    its ownership row (if any)."""
    await delete_report_artifacts(redis_client, task_id)
    try:
        await redis_client.delete(task_id, f"survey:{task_id}")
        await redis_client.srem("survey:task_ids", task_id)
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Failed to delete report content from Redis", exc_info=exc)
    if db is not None and report is not None:
        await db.delete(report)
