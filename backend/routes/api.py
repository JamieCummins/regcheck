from __future__ import annotations

from io import BytesIO
import json
import secrets
from typing import Any

from fastapi import APIRouter, Body, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from starlette.datastructures import Headers

from ..services.dimensions import default_dimensions_for
from ..services.trials import extract_nct_id
from .comparisons import ComparisonType, _enqueue_comparison, _parse_dimensions
from .status import get_task_status_payload

router = APIRouter(prefix="/api/v1")


class TextComparisonRequest(BaseModel):
    paper_text: str
    registration_text: str | None = None
    registration_id: str | None = None
    dimensions: Any | None = None
    parser_choice: str = "grobid"
    client: str = "openai"
    reasoning_effort: str | None = "medium"
    append_previous_output: Any | None = True
    multiple_experiments: Any | None = False
    experiment_number: str | None = None


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


def _yes_no_from_any(value: Any, *, default: str) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return default
    return _yes_no(str(value), default=default)


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


def _dimensions_from_payload(
    comparison_type: ComparisonType,
    dimensions: Any | None,
) -> list[dict[str, str]]:
    if dimensions is None or dimensions == "":
        return _dimensions_for_request(comparison_type, None)
    if isinstance(dimensions, str):
        return _dimensions_for_request(comparison_type, dimensions)
    return _parse_dimensions(json.dumps(dimensions))


def _text_upload(filename: str, text: str) -> UploadFile:
    payload = text.encode("utf-8")
    return UploadFile(
        BytesIO(payload),
        size=len(payload),
        filename=filename,
        headers=Headers({"content-type": "text/plain; charset=utf-8"}),
    )


def _json_request(request: Request) -> bool:
    content_type = request.headers.get("content-type", "").split(";", 1)[0]
    return content_type.strip().lower() == "application/json"


def _validate_text_payload(payload: Any) -> TextComparisonRequest | JSONResponse:
    try:
        return TextComparisonRequest.model_validate(payload)
    except ValidationError as exc:
        return _api_error(400, "INVALID_JSON", exc.json())


async def _create_text_comparison(
    request: Request,
    payload: TextComparisonRequest,
) -> JSONResponse:
    if not payload.paper_text or not payload.paper_text.strip():
        return _api_error(400, "MISSING_PAPER_TEXT", "paper_text is required.")

    registration_id_value = (payload.registration_id or "").strip()
    registration_text_value = payload.registration_text or ""
    has_registration_id = bool(registration_id_value)
    has_registration_text = bool(registration_text_value.strip())
    if not has_registration_id and not has_registration_text:
        return _api_error(
            400,
            "MISSING_REGISTRATION_INPUT",
            "Provide either registration_id or registration_text.",
        )
    if has_registration_id and has_registration_text:
        return _api_error(
            400,
            "AMBIGUOUS_REGISTRATION_INPUT",
            "Provide either registration_id or registration_text.",
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
        selected_dimensions = _dimensions_from_payload(comparison_type, payload.dimensions)
    except HTTPException as exc:
        return _api_error(exc.status_code, "INVALID_DIMENSIONS", _detail_message(exc.detail))

    try:
        queued = await _enqueue_comparison(
            request,
            comparison_type=comparison_type,
            parser_choice=(payload.parser_choice or "grobid").strip() or "grobid",
            client=(payload.client or "openai").strip() or "openai",
            reasoning_effort=(payload.reasoning_effort or "medium").strip() or "medium",
            append_previous_output=_yes_no_from_any(
                payload.append_previous_output,
                default="yes",
            ),
            selected_dimensions=selected_dimensions,
            registration_id=normalized_registration_id,
            preregistration=(
                None
                if has_registration_id
                else _text_upload("registration.txt", registration_text_value)
            ),
            paper=_text_upload("paper.txt", payload.paper_text),
            multiple_experiments=_yes_no_from_any(
                payload.multiple_experiments,
                default="no",
            ),
            experiment_number=payload.experiment_number,
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

    if paper is None and _json_request(request):
        try:
            raw_payload = await request.json()
        except json.JSONDecodeError:
            return _api_error(400, "INVALID_JSON", "Request body must be valid JSON.")
        payload = _validate_text_payload(raw_payload)
        if isinstance(payload, JSONResponse):
            return payload
        return await _create_text_comparison(request, payload)

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


@router.post("/comparisons/text")
async def create_text_comparison(
    request: Request,
    payload_data: dict[str, Any] = Body(...),
) -> JSONResponse:
    auth_error = _auth_error(request)
    if auth_error is not None:
        return auth_error

    payload = _validate_text_payload(payload_data)
    if isinstance(payload, JSONResponse):
        return payload
    return await _create_text_comparison(request, payload)


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
