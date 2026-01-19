from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse

from ..services.comparisons import (
    clinical_trial_comparison,
    general_preregistration_comparison,
    animals_trial_comparison,
)

router = APIRouter()
logger = logging.getLogger(__name__)

ComparisonType = Literal[
    "clinical_trials",
    "general_preregistration",
    "animals_trials",
]


async def _store_upload(destination: Path, upload: UploadFile) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    contents = await upload.read()
    with open(destination, "wb") as handle:
        handle.write(contents)
    return str(destination)


def _safe_filename(filename: str | None) -> str:
    name = Path(filename or "").name.strip()
    return name or "upload"


def _file_ext(filename: str | None) -> str:
    return Path(_safe_filename(filename)).suffix.lower()


async def _save_upload(
    upload_dir: Path,
    upload: UploadFile,
    *,
    prefix: str,
) -> tuple[str, str]:
    filename = _safe_filename(upload.filename)
    destination = upload_dir / f"{prefix}_{filename}"
    stored = await _store_upload(destination, upload)
    return stored, _file_ext(filename)


def _bool_from_yes(value: str | None) -> bool:
    return (value or "").strip().lower() == "yes"


def _parse_dimensions(dimensions_data: str) -> list[dict[str, str]]:
    try:
        payload = json.loads(dimensions_data)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid dimension payload") from exc

    if not isinstance(payload, list):
        raise HTTPException(status_code=400, detail="Invalid dimension payload")

    selected_dimensions: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = (item.get("dimension") or item.get("name") or "").strip()
        definition = (item.get("definition") or "").strip()
        if name:
            selected_dimensions.append({"dimension": name, "definition": definition})

    if not selected_dimensions:
        raise HTTPException(status_code=400, detail="At least one dimension must be selected")

    return selected_dimensions


def _normalize_parser_choice(parser_choice: str) -> str:
    normalized = (parser_choice or "").strip().lower()
    if normalized not in {"grobid", "dpt2"}:
        raise HTTPException(status_code=400, detail="Unsupported parser choice")
    return normalized


def _normalize_reasoning_effort(client: str, reasoning_effort: str | None) -> str | None:
    effort_normalized = (reasoning_effort or "").strip().lower()
    if client == "openai":
        if effort_normalized not in {"low", "medium", "high"}:
            effort_normalized = "medium"
        return effort_normalized
    return None


async def _queue_comparison(
    request: Request,
    *,
    comparison_type: ComparisonType,
    parser_choice: str,
    client: str,
    reasoning_effort: str | None,
    append_previous_output: str | None,
    dimensions_data: str,
    registration_id: str | None = None,
    preregistration: UploadFile | None = None,
    paper: UploadFile | None = None,
    registration_csv: UploadFile | None = None,
    multiple_experiments: str | None = None,
    experiment_number: str | None = None,
    experiment_text: str | None = None,
) -> RedirectResponse:
    settings = request.app.state.settings
    upload_dir = Path(settings.upload_dir)

    selected_dimensions = _parse_dimensions(dimensions_data)
    dimension_names = [item["dimension"] for item in selected_dimensions]

    append_previous = _bool_from_yes(append_previous_output)
    parser_choice_normalized = _normalize_parser_choice(parser_choice)
    effort_normalized = _normalize_reasoning_effort(client, reasoning_effort)
    logger.info(
        "queue_comparison normalized reasoning effort",
        extra={"client": client, "reasoning_effort": effort_normalized, "comparison_type": comparison_type},
    )

    if paper is None:
        raise HTTPException(status_code=400, detail="Paper upload is required")
    task_id = str(uuid.uuid4())
    paper_path, paper_ext = await _save_upload(upload_dir, paper, prefix=f"{task_id}_paper")

    stored_prereg_path: str | None = None
    prereg_ext: str | None = None
    stored_csv_path: str | None = None

    if comparison_type == "clinical_trials":
        if not registration_id or not registration_id.strip():
            raise HTTPException(
                status_code=400, detail="ClinicalTrials.gov link or ID is required for this option"
            )
    elif comparison_type == "general_preregistration":
        if preregistration is None:
            raise HTTPException(
                status_code=400, detail="Preregistration upload is required for this option"
            )
        stored_prereg_path, prereg_ext = await _save_upload(
            upload_dir, preregistration, prefix=f"{task_id}_prereg"
        )
    elif comparison_type == "animals_trials":
        if not registration_id or not registration_id.strip():
            raise HTTPException(status_code=400, detail="Registration ID is required for this option")
        if registration_csv is None:
            raise HTTPException(
                status_code=400,
                detail="CSV required for animals trials until API retrieval is implemented.",
            )
        stored_csv_path, _ = await _save_upload(
            upload_dir, registration_csv, prefix=f"{task_id}_registration"
        )
    else:
        raise HTTPException(status_code=400, detail="Unsupported comparison type")

    redis_client = request.app.state.redis
    initial_payload = {
        "state": "PENDING",
        "status": "Task queued",
        "result_json": json.dumps({"items": []}),
        "total_dimensions": len(dimension_names),
        "processed_dimensions": 0,
        "dimensions": json.dumps(dimension_names),
        "comparison_type": comparison_type,
    }
    try:
        await redis_client.hset(task_id, mapping=initial_payload)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Redis failed to set initial state", exc_info=exc)

    if comparison_type == "clinical_trials":
        asyncio.create_task(
            clinical_trial_comparison(
                registration_id,  # type: ignore[arg-type]
                paper_path,
                paper_ext,
                client,
                task_id,
                redis_client,
                parser_choice=parser_choice_normalized,
                reasoning_effort=effort_normalized,
                selected_dimensions=selected_dimensions,
                append_previous_output=append_previous,
            )
        )
    elif comparison_type == "general_preregistration":
        multiple_experiments_flag = _bool_from_yes(multiple_experiments)
        asyncio.create_task(
            general_preregistration_comparison(
                stored_prereg_path,  # type: ignore[arg-type]
                prereg_ext or "",
                paper_path,
                paper_ext,
                client,
                parser_choice_normalized,
                task_id,
                redis_client,
                selected_dimensions,
                append_previous_output=append_previous,
                reasoning_effort=effort_normalized,
                multiple_experiments=multiple_experiments_flag,
                experiment_number=experiment_number,
                experiment_text=experiment_text,
            )
        )
    else:
        asyncio.create_task(
            animals_trial_comparison(
                registration_id or "",
                paper_path,
                paper_ext,
                client,
                registration_csv_path=stored_csv_path,  # type: ignore[arg-type]
                parser_choice=parser_choice_normalized,
                task_id=task_id,
                redis_client=redis_client,
                selected_dimensions=selected_dimensions,
                append_previous_output=append_previous,
                reasoning_effort=effort_normalized,
            )
        )

    await redis_client.hset(task_id, mapping={"partial": "enabled"})
    return RedirectResponse(url=f"/survey/{task_id}", status_code=302)


@router.post("/compare", name="compare_post")
async def compare_post(
    request: Request,
    parser_choice: str = Form(...),
    client: str = Form(...),
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
):
    comparison_type: ComparisonType = (
        "clinical_trials" if _bool_from_yes(clinical_registration) else "general_preregistration"
    )
    return await _queue_comparison(
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
    )


@router.post("/clinical_trials")
async def clinical_trials_post(
    request: Request,
    parser_choice: str = Form(...),
    client: str = Form(...),
    reasoning_effort: str | None = Form(None),
    append_previous_output: str = Form("no"),
    registration_id: str = Form(...),
    paper: UploadFile = File(...),
    dimensions_data: str = Form(...),
):
    return await _queue_comparison(
        request,
        comparison_type="clinical_trials",
        parser_choice=parser_choice,
        client=client,
        reasoning_effort=reasoning_effort,
        append_previous_output=append_previous_output,
        registration_id=registration_id,
        paper=paper,
        dimensions_data=dimensions_data,
    )


@router.post("/general_preregistration")
async def general_preregistration_post(
    request: Request,
    parser_choice: str = Form(...),
    client: str = Form(...),
    reasoning_effort: str | None = Form(None),
    append_previous_output: str = Form("no"),
    multiple_experiments: str = Form("no"),
    experiment_number: str | None = Form(None),
    experiment_text: str | None = Form(None),
    preregistration: UploadFile = File(...),
    paper: UploadFile = File(...),
    dimensions_data: str = Form(...),
):
    return await _queue_comparison(
        request,
        comparison_type="general_preregistration",
        parser_choice=parser_choice,
        client=client,
        reasoning_effort=reasoning_effort,
        append_previous_output=append_previous_output,
        multiple_experiments=multiple_experiments,
        experiment_number=experiment_number,
        experiment_text=experiment_text,
        preregistration=preregistration,
        paper=paper,
        dimensions_data=dimensions_data,
    )


@router.post("/animals_trials")
async def animals_trials_post(
    request: Request,
    parser_choice: str = Form(...),
    client: str = Form(...),
    reasoning_effort: str | None = Form(None),
    append_previous_output: str = Form("no"),
    registration_id: str = Form(...),
    paper: UploadFile = File(...),
    registration_csv: UploadFile | None = File(None),
    dimensions_data: str = Form(...),
):
    return await _queue_comparison(
        request,
        comparison_type="animals_trials",
        parser_choice=parser_choice,
        client=client,
        reasoning_effort=reasoning_effort,
        append_previous_output=append_previous_output,
        registration_id=registration_id,
        paper=paper,
        registration_csv=registration_csv,
        dimensions_data=dimensions_data,
    )
