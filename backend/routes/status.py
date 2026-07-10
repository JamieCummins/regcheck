from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError
from redis.exceptions import TimeoutError as RedisTimeoutError

from ..services import reports as reports_service
from ..services import sharing as sharing_service
from ..services.report_artifacts import manifest_key

router = APIRouter()
logger = logging.getLogger(__name__)


async def _safe_hgetall(redis_client, key: str, *, retries: int = 2) -> dict:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return await redis_client.hgetall(key)
        except (RedisConnectionError, RedisTimeoutError, OSError, RedisError) as exc:
            last_error = exc
            logger.warning(
                "Redis hgetall failed (attempt %s/%s)",
                attempt + 1,
                retries + 1,
                extra={"key": key},
                exc_info=exc,
            )
            await asyncio.sleep(0.2 * (attempt + 1))
    logger.error("Redis hgetall exhausted retries", extra={"key": key}, exc_info=last_error)
    return {}


async def _safe_exists(redis_client, key: str) -> bool:
    try:
        return bool(await redis_client.exists(key))
    except (RedisConnectionError, RedisTimeoutError, OSError, RedisError):
        return False


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _coerce_int(value):
    value = _decode(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int,)):
        return int(value)
    if isinstance(value, float):
        return int(value)
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _status_field(data: dict, name: str) -> str | None:
    value = _decode(data.get(name))
    if value is None:
        return None
    value = str(value)
    return value if value else None


@router.get("/task_status/{task_id}")
async def task_status(request: Request, task_id: str):
    verdict = await sharing_service.ensure_viewable(request, task_id)
    if not verdict.allowed:
        return JSONResponse({"state": "FORBIDDEN", "status": "You do not have access to this report."}, status_code=403)
    redis_client = request.app.state.redis
    data = await _safe_hgetall(redis_client, task_id)
    if not data:
        return JSONResponse({"state": "UNKNOWN", "status": "Task not found or temporarily unavailable"})

    result_json = data.get("result_json")
    parsed_result = None
    if result_json:
        try:
            parsed_result = json.loads(result_json)
        except json.JSONDecodeError:  # pragma: no cover - defensive
            parsed_result = None
    total_dimensions = _coerce_int(data.get("total_dimensions"))
    if total_dimensions is None:
        dimensions_raw = _decode(data.get("dimensions"))
        if dimensions_raw:
            try:
                total_dimensions = len(json.loads(dimensions_raw))
            except (json.JSONDecodeError, TypeError):
                total_dimensions = None
    processed_dimensions = _coerce_int(data.get("processed_dimensions"))
    state = _decode(data.get("state"))
    status_text = _decode(data.get("status")) or "Pending..."
    evidence_available = await _safe_exists(redis_client, manifest_key(task_id))
    evidence_status = _status_field(data, "evidence_status")
    evidence_error = _status_field(data, "evidence_error")
    evidence_storage = _status_field(data, "evidence_storage")
    evidence_source_count = _coerce_int(data.get("evidence_source_count"))
    evidence_chunk_count = _coerce_int(data.get("evidence_chunk_count"))
    evidence_artifact_count = _coerce_int(data.get("evidence_artifact_count"))
    evidence_artifact_bytes = _coerce_int(data.get("evidence_artifact_bytes"))
    if evidence_available and evidence_status != "error":
        evidence_status = "ready"
        evidence_error = None
        evidence_storage = evidence_storage or "redis"
    elif state == "SUCCESS":
        if evidence_status in {None, "", "pending", "preparing"}:
            evidence_status = "missing"
        if not evidence_error:
            evidence_error = (
                "Evidence artifacts were not found for this completed report. "
                "The worker may not be running the current evidence-enabled release, "
                "or artifact storage failed before the manifest was saved."
            )

    settings_raw = data.get("settings_json")
    parsed_settings = None
    if settings_raw:
        try:
            if isinstance(settings_raw, bytes):
                settings_raw = settings_raw.decode("utf-8", "replace")
            parsed_settings = json.loads(settings_raw)
        except (json.JSONDecodeError, TypeError):  # pragma: no cover - defensive
            parsed_settings = None

    # Registered Report runs: Track A carried-forward-text integrity payload.
    rr_raw = data.get("rr_integrity")
    parsed_rr = None
    if rr_raw:
        try:
            if isinstance(rr_raw, bytes):
                rr_raw = rr_raw.decode("utf-8", "replace")
            parsed_rr = json.loads(rr_raw)
        except (json.JSONDecodeError, TypeError):  # pragma: no cover - defensive
            parsed_rr = None

    return JSONResponse(
        {
            "state": state,
            "status": status_text,
            "result": parsed_result,
            "settings": parsed_settings,
            "rr_integrity": parsed_rr,
            "total_dimensions": total_dimensions,
            "processed_dimensions": processed_dimensions,
            "evidence_available": evidence_available,
            "evidence_status": evidence_status,
            "evidence_error": evidence_error,
            "evidence_storage": evidence_storage,
            "evidence_source_count": evidence_source_count,
            "evidence_chunk_count": evidence_chunk_count,
            "evidence_artifact_count": evidence_artifact_count,
            "evidence_artifact_bytes": evidence_artifact_bytes,
        }
    )


@router.get("/result/{task_id}", response_class=HTMLResponse)
async def result(request: Request, task_id: str):
    templates = request.app.state.templates

    verdict = await sharing_service.ensure_viewable(request, task_id)
    if not verdict.allowed:
        if verdict.needs_login:
            login_url = request.url_for("login")
            return RedirectResponse(url=f"{login_url}?next=/result/{task_id}", status_code=302)
        return templates.TemplateResponse(
            "report_no_access.html",
            {"request": request, "task_id": task_id},
            status_code=403,
        )

    redis_client = request.app.state.redis
    data = await _safe_hgetall(redis_client, task_id)

    state = _decode(data.get("state")) if data else None
    total_dimensions = None
    processed_dimensions = None
    if data:
        total_dimensions = _coerce_int(data.get("total_dimensions"))
        if total_dimensions is None:
            dimensions_raw = _decode(data.get("dimensions"))
            if dimensions_raw:
                try:
                    total_dimensions = len(json.loads(dimensions_raw))
                except (json.JSONDecodeError, TypeError):
                    total_dimensions = None
        processed_dimensions = _coerce_int(data.get("processed_dimensions"))

    # Ownership / naming context for the viewer (rename, visibility, delete).
    report_title = (_decode(data.get("title")) if data else None) or "RegCheck Report"
    raw_visibility = _decode(data.get("visibility")) if data else None
    # Anonymous/legacy reports (no stored visibility) are public; otherwise map to
    # the current three-tier vocabulary (private -> unlisted).
    report_visibility = reports_service.normalize_visibility(raw_visibility) if raw_visibility else "public"
    owner_id = _decode(data.get("owner_id")) if data else None
    user = getattr(request.state, "user", None)
    owned = request.session.get("owned_reports")
    owned_in_session = isinstance(owned, list) and task_id in owned
    is_owner = bool((user is not None and owner_id and user.id == owner_id) or owned_in_session)
    # The visibility toggle is an account feature (anonymous reports stay public).
    can_set_visibility = bool(user is not None and owner_id and user.id == owner_id)

    # Owners always get the current allow-list so the share dialog is accurate
    # even before switching the report to restricted.
    shares_ctx: list[dict] = []
    sessionmaker = getattr(request.app.state, "db_sessionmaker", None)
    if can_set_visibility and sessionmaker is not None:
        async with sessionmaker() as db:
            shares_ctx = [
                {"id": s.id, "label": sharing_service.share_label(s), "type": "orcid" if s.grantee_orcid else "email"}
                for s in await sharing_service.list_shares(db, task_id)
            ]

    base_context = {
        "request": request,
        "task_id": task_id,
        "total_dimensions": total_dimensions or 0,
        "processed_dimensions": processed_dimensions or 0,
        "report_title": report_title,
        "report_visibility": report_visibility,
        "is_owner": is_owner,
        "can_set_visibility": can_set_visibility,
        "shares": shares_ctx,
        # Multi-study isolation outcome, so a run that silently fell back to the full
        # paper is visible rather than mistaken for a study-specific comparison.
        "multi_study_isolation": (_decode(data.get("multi_study_isolation")) if data else None),
        "multi_study_isolation_error": (_decode(data.get("multi_study_isolation_error")) if data else None),
        "multi_study_label": (_decode(data.get("multi_study_label")) if data else None),
    }

    if state == "SUCCESS":
        result_json = data.get("result_json")
        parsed_result = None
        if result_json:
            try:
                parsed_result = json.loads(result_json)
            except json.JSONDecodeError:
                # Avoid 500s on transient/partial reads; let the page load and the
                # frontend poll for the final payload.
                logger.warning("Invalid result_json for task; rendering empty result", extra={"task_id": task_id})
                parsed_result = None
        return templates.TemplateResponse("result.html", {**base_context, "result": parsed_result})

    return templates.TemplateResponse("result.html", {**base_context, "result": []})


@router.post("/append_result/{task_id}")
async def append_result(task_id: str, request: Request):
    return JSONResponse({"message": "Appending results is no longer supported."})
