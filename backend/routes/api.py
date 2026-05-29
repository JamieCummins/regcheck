from __future__ import annotations

import json
import secrets
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from ..services.dimensions import default_dimensions_for
from ..services.trials import extract_nct_id
from .comparisons import ComparisonType, _enqueue_comparison, _parse_dimensions
from .status import get_task_status_payload

router = APIRouter(prefix="/api/v1")


def _api_error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def _auth_error(request: Request) -> JSONResponse | None:
    configured_token = getattr(request.app.state.settings, "api_token", None)
    if not configured_token:
        return _api_error(
            503,
            "API_AUTH_NOT_CONFIGURED",
            "REGCHECK_API_TOKEN is not configured.",
        )

    supplied_token = (
        _extract_bearer_token(request.headers.get("authorization"))
        or (request.headers.get("x-api-key") or "").strip()
    )
    if not supplied_token:
        return _api_error(401, "MISSING_API_AUTH", "Provide an API token.")
    if not secrets.compare_digest(supplied_token, configured_token):
        return _api_error(401, "INVALID_API_AUTH", "Invalid API token.")
    return None


def _detail_message(detail: Any) -> str:
    if isinstance(detail, str):
        return detail
    try:
        return json.dumps(detail)
    except TypeError:
        return str(detail)


def _yes_no(value: str | None, *, default: str) -> str:
    normalized = (value or "").strip().lower()
    if normalized in {"yes", "true", "1", "on"}:
        return "yes"
    if normalized in {"no", "false", "0", "off"}:
        return "no"
    return default


def _normalize_task_state(state: str | None) -> str:
    state_map = {
        "PENDING": "queued",
        "IN_PROGRESS": "in_progress",
        "SUCCESS": "success",
        "FAILURE": "failure",
    }
    raw_state = (state or "").strip()
    return state_map.get(raw_state.upper(), raw_state.lower() or "unknown")


def _normalize_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if isinstance(result, list):
        return {"items": result}
    return {"items": []}


def _dimensions_for_request(
    comparison_type: ComparisonType,
    dimensions: str | None,
) -> list[dict[str, str]]:
    if dimensions and dimensions.strip():
        return _parse_dimensions(dimensions)
    if comparison_type == "clinical_trials":
        return default_dimensions_for("clinical_trials")
    return default_dimensions_for("general_preregistration")


@router.post("/comparisons")
async def create_comparison(
    request: Request,
    paper: UploadFile | None = File(None),
    registration_id: str | None = Form(None),
    registration_file: UploadFile | None = File(None),
    dimensions: str | None = Form(None),
    parser_choice: str = Form("grobid"),
    client: str = Form("openai"),
    reasoning_effort: str | None = Form("medium"),
    append_previous_output: str | None = Form("yes"),
    multiple_experiments: str | None = Form("no"),
    experiment_number: str | None = Form(None),
) -> JSONResponse:
    auth_error = _auth_error(request)
    if auth_error is not None:
        return auth_error

    if paper is None:
        return _api_error(400, "MISSING_PAPER", "Paper upload is required.")

    registration_id_value = (registration_id or "").strip()
    has_registration_id = bool(registration_id_value)
    has_registration_file = registration_file is not None
    if not has_registration_id and not has_registration_file:
        return _api_error(
            400,
            "MISSING_REGISTRATION_INPUT",
            "Provide either registration_id or registration_file.",
        )
    if has_registration_id and has_registration_file:
        return _api_error(
            400,
            "AMBIGUOUS_REGISTRATION_INPUT",
            "Provide either registration_id or registration_file.",
        )

    comparison_type: ComparisonType = (
        "clinical_trials" if has_registration_id else "general_preregistration"
    )
    normalized_registration_id: str | None = None
    if has_registration_id:
        try:
            normalized_registration_id = extract_nct_id(registration_id_value)
        except ValueError:
            return _api_error(
                400,
                "INVALID_REGISTRATION_ID",
                "registration_id must contain a valid NCT identifier.",
            )

    try:
        selected_dimensions = _dimensions_for_request(comparison_type, dimensions)
    except HTTPException as exc:
        return _api_error(exc.status_code, "INVALID_DIMENSIONS", _detail_message(exc.detail))

    try:
        queued = await _enqueue_comparison(
            request,
            comparison_type=comparison_type,
            parser_choice=(parser_choice or "grobid").strip() or "grobid",
            client=(client or "openai").strip() or "openai",
            reasoning_effort=(reasoning_effort or "medium").strip() or "medium",
            append_previous_output=_yes_no(append_previous_output, default="yes"),
            selected_dimensions=selected_dimensions,
            registration_id=normalized_registration_id,
            preregistration=registration_file,
            paper=paper,
            multiple_experiments=_yes_no(multiple_experiments, default="no"),
            experiment_number=experiment_number,
        )
    except HTTPException as exc:
        return _api_error(exc.status_code, "REQUEST_FAILED", _detail_message(exc.detail))

    status_url = f"/api/v1/comparisons/{queued.task_id}"
    return JSONResponse(
        status_code=202,
        content={
            "task_id": queued.task_id,
            "state": queued.state,
            "status": queued.status,
            "status_url": status_url,
        },
    )


@router.get("/comparisons/{task_id}")
async def get_comparison(request: Request, task_id: str) -> JSONResponse:
    auth_error = _auth_error(request)
    if auth_error is not None:
        return auth_error

    payload = await get_task_status_payload(request.app.state.redis, task_id)
    if payload is None:
        return _api_error(404, "TASK_NOT_FOUND", "Task not found.")

    return JSONResponse(
        {
            "task_id": task_id,
            "state": _normalize_task_state(payload.get("state")),
            "status": payload.get("status") or "Pending...",
            "processed_dimensions": payload.get("processed_dimensions") or 0,
            "total_dimensions": payload.get("total_dimensions") or 0,
            "result": _normalize_result(payload.get("result")),
        }
    )
