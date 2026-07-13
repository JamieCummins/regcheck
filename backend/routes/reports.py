from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.rate_limit import comparison_rate_limit
from ..core.storage import get_s3_config
from ..db import models
from ..db.session import get_db
from ..services import reports as reports_service
from ..services import sharing as sharing_service

router = APIRouter()
logger = logging.getLogger(__name__)


def _current_user(request: Request) -> models.User | None:
    return getattr(request.state, "user", None)


def _session_owned(request: Request) -> set[str]:
    owned = request.session.get("owned_reports")
    return set(owned) if isinstance(owned, list) else set()


def _can_manage(request: Request, task_id: str, report: models.Report | None) -> bool:
    """Who may rename/delete/regenerate a report.

    An OWNED report (DB row with an ``owner_id``) is manageable ONLY by that owner —
    browser-session ownership never overrides DB ownership, so a second account (or
    an anonymous user) sharing a browser cannot manage the first account's reports.
    A report with no owned row (anonymous / Redis-only) is manageable by the browser
    session that created it."""
    user = _current_user(request)
    if report is not None and (report.owner_id or "").strip():
        return user is not None and report.owner_id == user.id
    return task_id in _session_owned(request)


@router.get("/reports", name="my_reports")
async def my_reports(request: Request, db: AsyncSession = Depends(get_db)):
    user = _current_user(request)
    if user is None:
        return RedirectResponse(url=f"{request.url_for('login')}?next=/reports", status_code=302)
    reports = await reports_service.list_reports_for_owner(db, user.id)
    return request.app.state.templates.TemplateResponse(
        request,
        "dashboard.html",
        {"user": user, "reports": reports},
    )


@router.get("/u/{handle}", name="public_profile")
async def public_profile(request: Request, handle: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(models.User).where(models.User.handle == handle))
    profile_user = result.scalar_one_or_none()
    if profile_user is None or not profile_user.is_public_profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    reports = await reports_service.list_public_reports_for_owner(db, profile_user.id)
    return request.app.state.templates.TemplateResponse(
        request,
        "public_profile.html",
        {"profile_user": profile_user, "reports": reports},
    )


@router.post("/reports/{task_id}/rename", name="report_rename")
async def report_rename(
    request: Request,
    task_id: str,
    title: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    report = await reports_service.get_report_row(db, task_id)
    if not _can_manage(request, task_id, report):
        raise HTTPException(status_code=403, detail="Not allowed to rename this report")
    redis_client = request.app.state.redis
    new_title = await reports_service.rename_report(redis_client, db, report, title) if report else None
    if report is None:
        # Anonymous, Redis-only report owned by this session: rename in Redis.
        new_title = (title or "").strip()[:200] or "Untitled report"
        await redis_client.hset(task_id, mapping={"title": new_title})
    else:
        await db.commit()
    return JSONResponse({"ok": True, "title": new_title})


@router.post("/reports/{task_id}/visibility", name="report_visibility")
async def report_visibility(
    request: Request,
    task_id: str,
    visibility: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    user = _current_user(request)
    report = await reports_service.get_report_row(db, task_id)
    # Visibility is an account feature (anonymous reports are always public).
    if user is None or report is None or report.owner_id != user.id:
        raise HTTPException(status_code=403, detail="Not allowed to change visibility")
    vis = await reports_service.set_report_visibility(request.app.state.redis, db, report, visibility)
    await db.commit()
    return JSONResponse({"ok": True, "visibility": vis})


async def _owned_report_or_403(request: Request, task_id: str, db: AsyncSession) -> models.Report:
    """Sharing is an account-owner feature (restricted reports always have a row)."""
    user = _current_user(request)
    report = await reports_service.get_report_row(db, task_id)
    if user is None or report is None or report.owner_id != user.id:
        raise HTTPException(status_code=403, detail="Not allowed to manage sharing for this report")
    return report


def _share_payload(share: models.ReportShare) -> dict:
    return {
        "id": share.id,
        "label": sharing_service.share_label(share),
        "type": "orcid" if share.grantee_orcid else "email",
    }


@router.get("/reports/{task_id}/shares", name="report_shares_list")
async def report_shares_list(request: Request, task_id: str, db: AsyncSession = Depends(get_db)):
    await _owned_report_or_403(request, task_id, db)
    shares = await sharing_service.list_shares(db, task_id)
    return JSONResponse({"ok": True, "shares": [_share_payload(s) for s in shares]})


@router.post("/reports/{task_id}/shares", name="report_share_add")
async def report_share_add(
    request: Request,
    task_id: str,
    grantee: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    await _owned_report_or_403(request, task_id, db)
    try:
        share = await sharing_service.add_share(db, task_id, grantee)
    except sharing_service.ShareError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    await db.commit()
    return JSONResponse({"ok": True, "share": _share_payload(share)})


@router.post("/reports/{task_id}/shares/{share_id}/delete", name="report_share_delete")
async def report_share_delete(
    request: Request,
    task_id: str,
    share_id: str,
    db: AsyncSession = Depends(get_db),
):
    await _owned_report_or_403(request, task_id, db)
    removed = await sharing_service.remove_share(db, task_id, share_id)
    if removed:
        await db.commit()
    return JSONResponse({"ok": removed})


@router.post("/reports/{task_id}/delete", name="report_delete")
async def report_delete(
    request: Request,
    task_id: str,
    db: AsyncSession = Depends(get_db),
):
    report = await reports_service.get_report_row(db, task_id)
    if not _can_manage(request, task_id, report):
        raise HTTPException(status_code=403, detail="Not allowed to delete this report")
    # Centralized: tombstone + retained uploads + artifacts + survey + row.
    await reports_service.delete_report_everywhere(request.app.state.redis, db, task_id, report)
    if report is not None:
        await db.commit()
    # Drop it from the session's owned list.
    owned = request.session.get("owned_reports")
    if isinstance(owned, list) and task_id in owned:
        request.session["owned_reports"] = [t for t in owned if t != task_id]
    return JSONResponse({"ok": True})


@router.post(
    "/reports/{task_id}/regenerate",
    name="report_regenerate",
    dependencies=[Depends(comparison_rate_limit)],
)
async def report_regenerate(
    request: Request,
    task_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Re-run a report from its ORIGINAL uploaded files (no re-upload needed) — e.g.
    after an evidence-generation failure. Re-queues the stored job verbatim; the
    uploads it references are retained for the report's window."""
    report = await reports_service.get_report_row(db, task_id)
    if not _can_manage(request, task_id, report):
        raise HTTPException(status_code=403, detail="Not allowed to regenerate this report")

    redis_client = request.app.state.redis
    if await reports_service.is_deleted(redis_client, task_id):
        raise HTTPException(status_code=409, detail="This report has been deleted.")

    # Idempotency: one regeneration at a time per report. A short NX lock absorbs
    # double-clicks / retries so two POSTs don't queue (and pay for) the same job
    # twice; it clears on its own so a genuine later re-run is never blocked.
    lock_key = f"regen:lock:{task_id}"
    if not await redis_client.set(lock_key, "1", nx=True, ex=120):
        raise HTTPException(status_code=409, detail="A regeneration is already in progress.")

    current_state = (await redis_client.hget(task_id, "state") or "").strip().upper()
    if current_state in {"PENDING", "IN_PROGRESS"}:
        raise HTTPException(status_code=409, detail="This report is already being generated.")

    raw = await redis_client.hget(task_id, "regen_job")
    if not raw:
        raise HTTPException(
            status_code=409,
            detail="This report can no longer be regenerated. Please run a new comparison.",
        )
    payload = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw

    # Best-effort: when uploads live in Redis (no S3), make sure they're still present
    # before re-queuing, so the user gets a clear message rather than a failed re-run.
    if get_s3_config() is None:
        try:
            job = json.loads(payload)
            paper_key = (job.get("upload_keys") or {}).get("paper")
            if paper_key and not await redis_client.exists(paper_key):
                raise HTTPException(
                    status_code=409,
                    detail="The uploaded files for this report have expired. Please run a new comparison.",
                )
        except json.JSONDecodeError:  # pragma: no cover - defensive
            pass

    await redis_client.hset(
        task_id,
        mapping={
            "state": "PENDING",
            "status": "Re-running the comparison",
            "result_json": json.dumps({"items": []}),
            "processed_dimensions": 0,
            "evidence_status": "pending",
            "evidence_error": "",
        },
    )
    await redis_client.rpush("comparison:queue", payload)
    return JSONResponse({"ok": True})
