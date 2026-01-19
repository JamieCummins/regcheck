from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

router = APIRouter()


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


@router.get("/task_status/{task_id}")
async def task_status(request: Request, task_id: str):
    redis_client = request.app.state.redis
    data = await redis_client.hgetall(task_id)
    if not data:
        return JSONResponse({"state": "UNKNOWN", "status": "Task not found"})

    result_json = data.get("result_json")
    parsed_result = None
    if result_json:
        try:
            parsed_result = json.loads(result_json)
        except json.JSONDecodeError:
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

    return JSONResponse(
        {
            "state": state,
            "status": status_text,
            "result": parsed_result,
            "total_dimensions": total_dimensions,
            "processed_dimensions": processed_dimensions,
        }
    )


@router.get("/result/{task_id}", response_class=HTMLResponse)
async def result(request: Request, task_id: str):
    redis_client = request.app.state.redis
    data = await redis_client.hgetall(task_id)
    templates = request.app.state.templates

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

    if state == "SUCCESS":
        result_json = data.get("result_json")
        parsed_result = json.loads(result_json) if result_json else None
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "result": parsed_result,
                "task_id": task_id,
                "total_dimensions": total_dimensions or 0,
                "processed_dimensions": processed_dimensions or 0,
            },
        )

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "result": [],
            "task_id": task_id,
            "total_dimensions": total_dimensions or 0,
            "processed_dimensions": processed_dimensions or 0,
        },
    )


@router.post("/append_result/{task_id}")
async def append_result(task_id: str, request: Request):
    return JSONResponse({"message": "Appending results is no longer supported."})
