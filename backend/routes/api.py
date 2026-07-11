"""Keyed public API (/api/v1). All endpoints require a RegCheck API key
(`Authorization: Bearer rc_live_…`) issued by a signed-in user. Reports created
or queried here are owner-scoped to the key's user.
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.api_auth import require_api_key
from ..core.rate_limit import comparison_rate_limit
from ..db import models
from ..db.session import get_db
from ..services import reports as reports_service
from ..services.dimensions import discipline_keys, get_discipline_dimensions
from . import comparisons as comparisons_routes

# OpenAPI-only security declaration: documents the Bearer requirement on every
# /api/v1 endpoint and powers the Swagger "Authorize" button. auto_error=False so
# enforcement stays with require_api_key (which also accepts X-API-Key and returns
# a precise 401 message).
_bearer_scheme = HTTPBearer(
    auto_error=False,
    scheme_name="RegCheckApiKey",
    description="RegCheck API key issued from your profile page: `Authorization: Bearer rc_live_…`",
)

router = APIRouter(prefix="/api/v1", dependencies=[Depends(_bearer_scheme)])
logger = logging.getLogger(__name__)


def _report_links(request: Request, task_id: str) -> dict[str, str]:
    base = str(request.base_url).rstrip("/")
    return {
        "status_url": f"{base}/api/v1/status/{task_id}",
        "report_url": f"{base}/api/v1/reports/{task_id}",
        "view_url": f"{base}/result/{task_id}",
    }


def _resolve_api_dimensions(dimension_set: str | None, dimensions_data: str | None) -> str:
    """Resolve the dimensions for an API request into a dimensions_data JSON string.
    Callers may pass either ``dimension_set`` (a built-in discipline preset, same as
    the web app / CLI) or ``dimensions_data`` (a custom JSON list) — exactly one."""
    set_key = (dimension_set or "").strip()
    raw = (dimensions_data or "").strip()
    if set_key and raw:
        raise HTTPException(status_code=400, detail="Provide only one of 'dimension_set' or 'dimensions_data'.")
    if set_key:
        dims = get_discipline_dimensions(set_key)
        if not dims:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown dimension_set '{dimension_set}'. Choose one of: {', '.join(discipline_keys())}.",
            )
        return json.dumps(dims)
    if raw:
        return raw
    raise HTTPException(status_code=400, detail="Provide dimensions via 'dimension_set' or 'dimensions_data'.")


async def _owned_report_or_404(db: AsyncSession, user: models.User, task_id: str) -> models.Report:
    report = await reports_service.get_report_row(db, task_id)
    if report is None or report.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


@router.post("/compare", status_code=202, dependencies=[Depends(comparison_rate_limit)])
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
    osf_url: str | None = Form(None),
    paper: UploadFile | None = File(None),
    dimensions_data: str | None = Form(None),
    dimension_set: str | None = Form(None),
    visibility: str | None = Form(None),
):
    comparison_type = (
        "clinical_trials"
        if comparisons_routes._bool_from_yes(clinical_registration)
        else "general_preregistration"
    )
    dimensions_data = _resolve_api_dimensions(dimension_set, dimensions_data)
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
        osf_url=osf_url,
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
