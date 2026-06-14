"""Keyed public API (/api/v1). All endpoints require a RegCheck API key
(`Authorization: Bearer rc_live_…`) issued by a signed-in user. Reports created
or queried here are owner-scoped to the key's user.
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.api_auth import require_api_key
from ..db import models
from ..db.session import get_db
from ..services import reports as reports_service
from . import comparisons as comparisons_routes

router = APIRouter(prefix="/api/v1")
logger = logging.getLogger(__name__)


def _report_links(request: Request, task_id: str) -> dict[str, str]:
    base = str(request.base_url).rstrip("/")
    return {
        "status_url": f"{base}/api/v1/status/{task_id}",
        "report_url": f"{base}/api/v1/reports/{task_id}",
        "view_url": f"{base}/result/{task_id}",
    }


async def _owned_report_or_404(db: AsyncSession, user: models.User, task_id: str) -> models.Report:
    report = await reports_service.get_report_row(db, task_id)
    if report is None or report.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


@router.post("/compare")
async def api_compare(
    request: Request,
    user: models.User = Depends(require_api_key),
    parser_choice: str = Form("pymupdf"),
    client: str = Form("openai"),
    reasoning_effort: str | None = Form(None),
    append_previous_output: str = Form("no"),
    multiple_experiments: str = Form("no"),
    experiment_number: str | None = Form(None),
    experiment_text: str | None = Form(None),
    clinical_registration: str = Form("no"),
    registration_id: str | None = Form(None),
    preregistration: UploadFile | None = File(None),
    paper: UploadFile | None = File(None),
    dimensions_data: str = Form(...),
    visibility: str | None = Form(None),
):
    comparison_type = (
        "clinical_trials"
        if comparisons_routes._bool_from_yes(clinical_registration)
        else "general_preregistration"
    )
    task_id = await comparisons_routes._queue_comparison(
        request,
        comparison_type=comparison_type,
        parser_choice=parser_choice,
        client=client,
        reasoning_effort=reasoning_effort,
        append_previous_output=append_previous_output,
        multiple_experiments=multiple_experiments,
        experiment_number=experiment_number,
        experiment_text=experiment_text,
        registration_id=registration_id,
        preregistration=preregistration,
        paper=paper,
        dimensions_data=dimensions_data,
        visibility=visibility,
        owner_override=user,
        source="api",
    )
    return JSONResponse(
        {"task_id": task_id, "state": "PENDING", **_report_links(request, task_id)},
        status_code=202,
    )


def _decode_hash(data: dict) -> dict:
    out = {}
    for k, v in (data or {}).items():
        out[k.decode() if isinstance(k, bytes) else k] = v.decode() if isinstance(v, bytes) else v
    return out


@router.get("/status/{task_id}")
async def api_status(
    request: Request,
    task_id: str,
    user: models.User = Depends(require_api_key),
    db: AsyncSession = Depends(get_db),
):
    await _owned_report_or_404(db, user, task_id)
    data = _decode_hash(await request.app.state.redis.hgetall(task_id))
    if not data:
        raise HTTPException(status_code=404, detail="Report content not found (it may have expired)")

    def _as_int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    return {
        "task_id": task_id,
        "state": data.get("state"),
        "status": data.get("status"),
        "processed_dimensions": _as_int(data.get("processed_dimensions")),
        "total_dimensions": _as_int(data.get("total_dimensions")),
        "title": data.get("title"),
        "visibility": data.get("visibility"),
        **_report_links(request, task_id),
    }


@router.get("/reports")
async def api_list_reports(
    request: Request,
    user: models.User = Depends(require_api_key),
    db: AsyncSession = Depends(get_db),
):
    rows = await reports_service.list_reports_for_owner(db, user.id)
    return {
        "reports": [
            {
                "task_id": r.task_id,
                "title": r.title,
                "visibility": r.visibility,
                "comparison_type": r.comparison_type,
                "source": r.source,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                **_report_links(request, r.task_id),
            }
            for r in rows
        ]
    }


@router.get("/reports/{task_id}")
async def api_get_report(
    request: Request,
    task_id: str,
    user: models.User = Depends(require_api_key),
    db: AsyncSession = Depends(get_db),
):
    report = await _owned_report_or_404(db, user, task_id)
    data = _decode_hash(await request.app.state.redis.hgetall(task_id))
    if not data:
        raise HTTPException(status_code=404, detail="Report content not found (it may have expired)")
    result = None
    if data.get("result_json"):
        try:
            result = json.loads(data["result_json"])
        except json.JSONDecodeError:
            result = None
    return {
        "task_id": task_id,
        "title": report.title,
        "visibility": report.visibility,
        "comparison_type": report.comparison_type,
        "state": data.get("state"),
        "result": result,
        **_report_links(request, task_id),
    }


@router.delete("/reports/{task_id}")
async def api_delete_report(
    request: Request,
    task_id: str,
    user: models.User = Depends(require_api_key),
    db: AsyncSession = Depends(get_db),
):
    report = await _owned_report_or_404(db, user, task_id)
    await reports_service.delete_report_everywhere(request.app.state.redis, db, task_id, report)
    await db.commit()
    return {"ok": True}
