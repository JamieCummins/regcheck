from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse

from ..core.rate_limit import comparison_rate_limit
from ..core.storage import get_s3_config, guess_content_type, s3_upload_fileobj
from ..services import reports as reports_service
from ..services.documents import read_file
from ..services.llm import HOSTED_CLIENTS
from ..services.osf import extract_osf_guid, fetch_osf_preregistration

router = APIRouter()
logger = logging.getLogger(__name__)

# Sentinel: distinguishes "owner not supplied (use the session user)" from an
# explicit owner of None (anonymous).
_UNSET = object()


async def _compare_and_redirect(*args, **kwargs) -> RedirectResponse:
    """UI wrapper: queue a comparison and redirect to the post-run survey."""
    task_id = await _queue_comparison(*args, **kwargs)
    return RedirectResponse(url=f"/survey/{task_id}", status_code=302)


def _session_track_report(request: Request, task_id: str) -> None:
    """Record a report created in this browser session so the anonymous (or
    signed-in) creator can manage it from the same browser. Capped to avoid
    unbounded session-cookie growth."""
    try:
        owned = request.session.get("owned_reports")
        if not isinstance(owned, list):
            owned = []
        if task_id not in owned:
            owned.append(task_id)
        request.session["owned_reports"] = owned[-50:]
    except Exception:  # pragma: no cover - session is best-effort
        pass
DEFAULT_MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


def _upload_limit() -> int:
    raw = os.getenv("MAX_UPLOAD_BYTES")
    if raw is None:
        return DEFAULT_MAX_UPLOAD_BYTES
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_MAX_UPLOAD_BYTES
    return max(1, parsed)


MAX_UPLOAD_BYTES = _upload_limit()

# Document types the comparison pipeline can read (mirrors documents.read_file).
_SUPPORTED_DOC_EXTS = {".pdf", ".docx", ".txt", ".html", ".htm"}

ComparisonType = Literal[
    "clinical_trials",
    "general_preregistration",
    "registered_report",
    "animals_trials",
]


async def _store_upload(
    destination: Path,
    upload: UploadFile,
    *,
    max_bytes: int | None = None,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    size_limit = max_bytes or MAX_UPLOAD_BYTES
    total_read = 0
    try:
        with open(destination, "wb") as handle:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total_read += len(chunk)
                if size_limit and total_read > size_limit:
                    raise HTTPException(
                        status_code=413,
                        detail="Uploaded file exceeds the permitted size limit.",
                    )
                handle.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        try:
            await upload.seek(0)
        except Exception:
            pass
    return str(destination)


def _safe_filename(filename: str | None) -> str:
    name = Path(filename or "").name.strip()
    return name or "upload"


def _file_ext(filename: str | None) -> str:
    return Path(_safe_filename(filename)).suffix.lower()


def _validate_doc_ext(ext: str, *, kind: str) -> None:
    """Reject documents the comparison pipeline can't read, with a clear message.

    Without this, an unsupported (or extension-less) upload passes submit and only
    fails deep in the worker as a cryptic ``Worker error: Unsupported file type``.
    """
    if (ext or "").lower() not in _SUPPORTED_DOC_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported {kind} file type ('{ext or 'unknown'}'). Upload a PDF, DOCX, TXT, or HTML file.",
        )


async def _save_upload(
    upload_dir: Path,
    upload: UploadFile,
    *,
    prefix: str,
    max_bytes: int | None = None,
) -> tuple[str, str]:
    filename = _safe_filename(upload.filename)
    destination = upload_dir / f"{prefix}_{filename}"
    stored = await _store_upload(destination, upload, max_bytes=max_bytes)
    return stored, _file_ext(filename)


def _non_empty_uploads(files: list[UploadFile] | UploadFile | None) -> list[UploadFile]:
    """Form fields arrive as a single value, a list, or an empty-filename placeholder;
    normalise to the list of uploads that actually carry a file."""
    if files is None:
        return []
    items = files if isinstance(files, list) else [files]
    return [f for f in items if f is not None and (getattr(f, "filename", "") or "").strip()]


async def _coalesce_uploads(
    files: list[UploadFile],
    *,
    upload_dir: Path,
    kind: str,
    max_bytes: int | None = None,
) -> UploadFile | None:
    """Combine multiple uploaded documents into ONE for the comparison.

    A single file is returned untouched (keeps its native parser + PDF rendering).
    Two or more are each extracted to text and concatenated, with labelled separators,
    into one plain-text upload — so "upload several files as the paper/registration"
    becomes one registration vs one paper without touching the worker or comparison.
    (Several PDFs can't share a single page view anyway, hence text.)"""
    real = _non_empty_uploads(files)
    if not real:
        return None
    if len(real) == 1:
        return real[0]

    parts: list[str] = []
    tmp_paths: list[str] = []
    try:
        for index, upload in enumerate(real, start=1):
            name = _safe_filename(upload.filename)
            _validate_doc_ext(_file_ext(name), kind=kind)
            path, ext = await _save_upload(
                upload_dir, upload, prefix=f"combine_{uuid.uuid4()}", max_bytes=max_bytes
            )
            tmp_paths.append(path)
            try:
                text = await asyncio.to_thread(read_file, path, ext)
            except Exception as exc:
                raise HTTPException(
                    status_code=400, detail=f"Could not read {kind} file '{name}'."
                ) from exc
            parts.append(f"===== Document {index}: {name} =====\n\n{(text or '').strip()}")
        combined = ("\n\n\n".join(parts)).encode("utf-8")
        return UploadFile(file=io.BytesIO(combined), filename=f"{kind}-combined.txt", size=len(combined))
    finally:
        for stale in tmp_paths:
            try:
                Path(stale).unlink(missing_ok=True)
            except Exception:  # pragma: no cover - best-effort cleanup
                pass


async def _store_upload_to_redis(redis_client, redis_key: str, file_path: str, ttl_seconds: int = 86400) -> None:
    """Store an uploaded file's contents in Redis (compressed + base64) so workers can reconstruct it."""
    try:
        raw = Path(file_path).read_bytes()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read uploaded file: {exc}") from exc
    import base64
    import gzip

    compressed = gzip.compress(raw)
    encoded = base64.b64encode(compressed).decode("ascii")
    try:
        await redis_client.set(redis_key, encoded, ex=ttl_seconds)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to persist upload to queue store") from exc


async def _store_upload_to_s3(task_id: str, file_path: str, *, label: str) -> str:
    cfg = get_s3_config()
    if cfg is None:
        raise RuntimeError("S3_BUCKET not configured")
    ext = Path(file_path).suffix.lower()
    key = f"regcheck/uploads/{task_id}/{label}{ext}"

    def _upload() -> None:
        with open(file_path, "rb") as handle:
            s3_upload_fileobj(
                cfg,
                key=key,
                fileobj=handle,
                content_type=guess_content_type(file_path),
            )

    await asyncio.to_thread(_upload)
    return key


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
    if normalized not in {"grobid", "dpt2", "pymupdf", "external"}:
        raise HTTPException(status_code=400, detail="Unsupported parser choice")
    return normalized


def _normalize_client(client: str) -> str:
    """Validate the model provider for the hosted app/API. gpustack is rejected here
    because the Heroku worker can't reach the Uni Bern network — it's CLI-only."""
    normalized = (client or "").strip().lower()
    if normalized not in HOSTED_CLIENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model provider '{client}'.",
        )
    return normalized


def _normalize_reasoning_effort(client: str, reasoning_effort: str | None) -> str | None:
    effort_normalized = (reasoning_effort or "").strip().lower()
    # ChatGPT (gpt-5.5) is the only reasoning-effort model exposed to users.
    if client == "openai":
        if effort_normalized not in {"low", "medium", "high"}:
            effort_normalized = "medium"
        return effort_normalized
    return None


async def _safe_hset(redis_client, task_id: str, mapping: dict, retries: int = 2) -> None:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            await redis_client.hset(task_id, mapping=mapping)
            return
        except Exception as exc:  # pragma: no cover - defensive logging
            last_error = exc
            logger.error("Redis hset failed (attempt %s/%s)", attempt + 1, retries + 1, exc_info=exc)
            await asyncio.sleep(0.2 * (attempt + 1))
    if last_error:
        raise last_error


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
    osf_url: str | None = None,
    paper: UploadFile | None = None,
    registration_csv: UploadFile | None = None,
    multiple_experiments: str | None = None,
    experiment_number: str | None = None,
    experiment_text: str | None = None,
    visibility: str | None = None,
    owner_override: object = _UNSET,
    source: str = "ui",
) -> str:
    settings = request.app.state.settings
    upload_dir = Path(settings.upload_dir)
    redis_client = request.app.state.redis
    # Keep the stored uploads alive for the report's lifetime (capped at the anon
    # window) so a report can be REGENERATED from its original files without a
    # re-upload. They still expire, so storage stays bounded.
    _regen_window = int(getattr(settings, "anonymous_task_ttl_seconds", 7 * 24 * 60 * 60))
    upload_ttl = max(86400, _regen_window)

    # Basic backpressure: refuse new jobs if queue + in-flight exceeds configured limit.
    try:
        queued = await redis_client.llen("comparison:queue")
        in_flight = await redis_client.llen("comparison:processing")
        if queued + in_flight >= settings.max_queue_length:
            raise HTTPException(status_code=503, detail="System is busy; please retry shortly.")
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Failed to compute queue depth; proceeding without backpressure", exc_info=exc)

    selected_dimensions = _parse_dimensions(dimensions_data)
    dimension_names = [item["dimension"] for item in selected_dimensions]

    append_previous = _bool_from_yes(append_previous_output)
    client = _normalize_client(client)
    parser_choice_normalized = _normalize_parser_choice(parser_choice)
    effort_normalized = _normalize_reasoning_effort(client, reasoning_effort)
    logger.info(
        "queue_comparison normalized reasoning effort",
        extra={"client": client, "reasoning_effort": effort_normalized, "comparison_type": comparison_type},
    )

    # An empty file input arrives as an UploadFile with a blank filename (not
    # None), so guard on the filename rather than identity.
    if paper is None or not (getattr(paper, "filename", "") or "").strip():
        raise HTTPException(status_code=400, detail="Paper upload is required")
    task_id = str(uuid.uuid4())
    paper_path, paper_ext = await _save_upload(
        upload_dir, paper, prefix=f"{task_id}_paper", max_bytes=MAX_UPLOAD_BYTES
    )
    # Validate the paper type at submit (the prereg is validated below). Otherwise
    # an unsupported paper only fails later in the worker with a cryptic error.
    if (paper_ext or "").lower() not in _SUPPORTED_DOC_EXTS:
        Path(paper_path).unlink(missing_ok=True)
        _validate_doc_ext(paper_ext, kind="paper")
    paper_redis_key = f"upload:{task_id}:paper"
    prereg_redis_key: str | None = None
    csv_redis_key: str | None = None

    # Prefer durable object storage (S3) so worker dynos can always access uploads.
    # Fall back to storing compressed blobs in Redis when S3 isn't configured.
    s3_keys: dict[str, str | None] = {"paper": None, "prereg": None, "csv": None}
    if get_s3_config() is not None:
        s3_keys["paper"] = await _store_upload_to_s3(task_id, paper_path, label="paper")
        try:
            Path(paper_path).unlink(missing_ok=True)
        except Exception:
            pass
    else:
        await _store_upload_to_redis(redis_client, paper_redis_key, paper_path, ttl_seconds=upload_ttl)
        try:
            Path(paper_path).unlink(missing_ok=True)
        except Exception:
            pass

    stored_prereg_path: str | None = None
    prereg_ext: str | None = None
    stored_csv_path: str | None = None

    if comparison_type == "clinical_trials":
        if not registration_id or not registration_id.strip():
            raise HTTPException(
                status_code=400, detail="ClinicalTrials.gov link or ID is required for this option"
            )
    elif comparison_type in ("general_preregistration", "registered_report"):
        if osf_url and osf_url.strip():
            # Preregistration comes from an OSF link; validate it here and fetch
            # it in the worker (no file to store).
            if extract_osf_guid(osf_url) is None:
                raise HTTPException(
                    status_code=400,
                    detail="Enter a valid OSF link, e.g. https://osf.io/abc12/.",
                )
        elif preregistration is None or not (getattr(preregistration, "filename", "") or "").strip():
            # An empty file input (e.g. the OSF source was active but its link
            # field was disabled/not submitted) arrives as an UploadFile with a
            # blank filename — not None — so guard on the filename, not identity.
            raise HTTPException(
                status_code=400, detail="Provide a preregistration file or an OSF link."
            )
        elif _file_ext(preregistration.filename) not in _SUPPORTED_DOC_EXTS:
            raise HTTPException(
                status_code=400,
                detail="Unsupported preregistration file type. Upload a PDF, DOCX, TXT, or HTML file, or paste an OSF link.",
            )
        else:
            stored_prereg_path, prereg_ext = await _save_upload(
                upload_dir, preregistration, prefix=f"{task_id}_prereg", max_bytes=MAX_UPLOAD_BYTES
            )
            prereg_redis_key = f"upload:{task_id}:prereg"
            if get_s3_config() is not None:
                s3_keys["prereg"] = await _store_upload_to_s3(task_id, stored_prereg_path, label="prereg")
                try:
                    Path(stored_prereg_path).unlink(missing_ok=True)
                except Exception:
                    pass
            else:
                await _store_upload_to_redis(redis_client, prereg_redis_key, stored_prereg_path, ttl_seconds=upload_ttl)
                try:
                    Path(stored_prereg_path).unlink(missing_ok=True)
                except Exception:
                    pass
    elif comparison_type == "animals_trials":
        if not registration_id or not registration_id.strip():
            raise HTTPException(status_code=400, detail="Registration ID is required for this option")
        if registration_csv is None:
            raise HTTPException(
                status_code=400,
                detail="CSV required for animals trials until API retrieval is implemented.",
            )
        stored_csv_path, _ = await _save_upload(
            upload_dir, registration_csv, prefix=f"{task_id}_registration", max_bytes=MAX_UPLOAD_BYTES
        )
        csv_redis_key = f"upload:{task_id}:csv"
        if get_s3_config() is not None:
            s3_keys["csv"] = await _store_upload_to_s3(task_id, stored_csv_path, label="registration")
            try:
                Path(stored_csv_path).unlink(missing_ok=True)
            except Exception:
                pass
        else:
            await _store_upload_to_redis(redis_client, csv_redis_key, stored_csv_path, ttl_seconds=upload_ttl)
            try:
                Path(stored_csv_path).unlink(missing_ok=True)
            except Exception:
                pass
    else:
        raise HTTPException(status_code=400, detail="Unsupported comparison type")

    # Ownership + retention: signed-in users own a persistent report (their
    # chosen visibility); anonymous runs are public and auto-delete on a short
    # TTL. Title is auto-generated and renamable later.
    owner = getattr(request.state, "user", None) if owner_override is _UNSET else owner_override
    report_title = reports_service.generate_default_title(
        comparison_type=comparison_type,
        paper_filename=getattr(paper, "filename", None),
        registration_id=registration_id,
    )
    if owner is not None:
        report_visibility = reports_service.normalize_visibility(visibility)
        retention = "persist"
        anon_ttl = None
    else:
        report_visibility = "public"
        anon_ttl = int(getattr(settings, "anonymous_task_ttl_seconds", 7 * 24 * 60 * 60))
        retention = str(anon_ttl)

    initial_payload = {
        "state": "PENDING",
        "status": "Task queued",
        "result_json": json.dumps({"items": []}),
        "total_dimensions": len(dimension_names),
        "processed_dimensions": 0,
        "dimensions": json.dumps(dimension_names),
        "comparison_type": comparison_type,
        "evidence_status": "pending",
        "evidence_error": "",
        "title": report_title,
        "visibility": report_visibility,
        "owner_id": owner.id if owner is not None else "",
        "retention": retention,
        # Settings used for this report, surfaced read-only in the viewer ("View settings").
        "settings_json": json.dumps({
            "comparison_type": comparison_type,
            "client": client,
            "parser_choice": parser_choice_normalized,
            "reasoning_effort": effort_normalized,
            "append_previous_output": bool(append_previous),
            "multiple_experiments": _bool_from_yes(multiple_experiments),
            "experiment_number": (experiment_number or "").strip() or None,
            "dimensions": dimension_names,
        }),
    }
    try:
        await _safe_hset(redis_client, task_id, initial_payload)
        if anon_ttl is not None:
            await redis_client.expire(task_id, anon_ttl)
        else:
            # Signed-in reports persist until the owner deletes them.
            await redis_client.persist(task_id)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Redis failed to set initial state", exc_info=exc)
        raise HTTPException(status_code=503, detail="Failed to queue task; please retry.") from exc

    # Durable ownership record for signed-in users (anonymous runs are Redis-only).
    if owner is not None:
        sessionmaker = getattr(request.app.state, "db_sessionmaker", None)
        if sessionmaker is not None:
            try:
                async with sessionmaker() as db:
                    await reports_service.create_report_row(
                        db,
                        task_id=task_id,
                        owner_id=owner.id,
                        visibility=report_visibility,
                        title=report_title,
                        comparison_type=comparison_type,
                        source=source,
                    )
                    await db.commit()
            except Exception as exc:  # pragma: no cover - don't fail the run on metadata
                logger.error("Failed to persist report ownership row", exc_info=exc, extra={"task_id": task_id})

    if source == "ui":
        _session_track_report(request, task_id)

    job_payload = {
        "comparison_type": comparison_type,
        "task_id": task_id,
        "client": client,
        "parser_choice": parser_choice_normalized,
        "reasoning_effort": effort_normalized,
        "append_previous_output": append_previous,
        "selected_dimensions": selected_dimensions,
        "upload_keys": {"paper": paper_redis_key, "prereg": prereg_redis_key, "csv": csv_redis_key},
        "s3_keys": s3_keys,
    }

    if comparison_type == "clinical_trials":
        job_payload.update(
            {
                "registration_id": registration_id,
                "paper_path": paper_path,
                "paper_ext": paper_ext,
            }
        )
    elif comparison_type in ("general_preregistration", "registered_report"):
        multiple_experiments_flag = _bool_from_yes(multiple_experiments)
        job_payload.update(
            {
                "prereg_path": stored_prereg_path,
                "prereg_ext": prereg_ext or "",
                "osf_url": (osf_url or "").strip() or None,
                "paper_path": paper_path,
                "paper_ext": paper_ext,
                "multiple_experiments": multiple_experiments_flag,
                "experiment_number": experiment_number,
                "experiment_text": experiment_text,
            }
        )
    else:
        job_payload.update(
            {
                "registration_id": registration_id,
                "paper_path": paper_path,
                "paper_ext": paper_ext,
                "registration_csv_path": stored_csv_path,
            }
        )

    # Persist the job so the report can be re-queued verbatim by "Regenerate"
    # (the stored uploads it references are kept for the same window).
    try:
        await _safe_hset(redis_client, task_id, {"regen_job": json.dumps(job_payload)})
    except Exception as exc:  # pragma: no cover - non-fatal: regenerate just won't be offered
        logger.warning("Failed to persist regen job payload", exc_info=exc, extra={"task_id": task_id})

    try:
        await redis_client.rpush("comparison:queue", json.dumps(job_payload))
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to enqueue comparison job", exc_info=exc, extra={"task_id": task_id})
        await redis_client.hset(
            task_id,
            mapping={
                "state": "FAILURE",
                "status": "Failed to enqueue job; please retry.",
            },
        )
        raise HTTPException(status_code=503, detail="Failed to queue comparison. Please retry.") from exc

    return task_id


def _preflight_min_chars() -> int:
    """Below this many extractable characters, a registration is treated as
    "thin" and the wizard warns (but never blocks). Env-tunable so it can be
    calibrated against real OSF stubs in production without a redeploy."""
    raw = (os.getenv("PREFLIGHT_MIN_REGISTRATION_CHARS") or "").strip()
    if not raw:
        return 400
    try:
        return max(0, int(raw))
    except ValueError:
        return 400


@router.post("/preflight/registration")
async def preflight_registration(
    request: Request,
    prereg_source: str = Form("upload"),
    osf_url: str | None = Form(None),
    preregistration: UploadFile | None = File(None),
):
    """Cheap pre-submit probe of how much text we can extract from the chosen
    registration, so the wizard can WARN (not block) when it is near-empty —
    e.g. an OSF *registry* link whose substantive content is an attached file.

    Best-effort by design: on any resolution/parse problem we return
    ``ok: false`` and the wizard proceeds to a normal submit (the worker stays
    the source of truth). Uses the lightweight PyMuPDF/stdlib readers, never
    GROBID, so it stays fast in the request path."""
    settings = request.app.state.settings
    upload_dir = Path(settings.upload_dir)
    threshold = _preflight_min_chars()
    source = (prereg_source or "").strip().lower()
    osf_link = (osf_url or "").strip()

    cleanup: list[str] = []
    try:
        if source == "osf" or osf_link:
            if not osf_link:
                return JSONResponse({"ok": False, "reason": "no_osf_url"})
            try:
                path, ext = await asyncio.to_thread(fetch_osf_preregistration, osf_link, upload_dir)
            except Exception as exc:  # network / unresolved / unsupported file
                logger.info("preflight OSF resolution failed", exc_info=exc)
                return JSONResponse({"ok": False, "reason": "osf_unresolved"})
            cleanup.append(path)
            probe_source = "osf"
        elif preregistration is not None and (getattr(preregistration, "filename", "") or "").strip():
            if _file_ext(preregistration.filename) not in _SUPPORTED_DOC_EXTS:
                return JSONResponse({"ok": False, "reason": "unsupported_type"})
            path, ext = await _save_upload(
                upload_dir, preregistration, prefix=f"preflight_{uuid.uuid4()}_prereg", max_bytes=MAX_UPLOAD_BYTES
            )
            cleanup.append(path)
            probe_source = "upload"
        else:
            # Nothing probe-able here (e.g. ClinicalTrials.gov, or no input yet).
            return JSONResponse({"ok": False, "reason": "no_input"})

        try:
            text = await asyncio.to_thread(read_file, path, ext)
        except Exception as exc:
            logger.info("preflight text extraction failed", exc_info=exc)
            return JSONResponse({"ok": False, "reason": "parse_failed"})

        chars = len((text or "").strip())
        return JSONResponse(
            {"ok": True, "chars": chars, "threshold": threshold, "thin": chars < threshold, "source": probe_source}
        )
    finally:
        for stale in cleanup:
            try:
                Path(stale).unlink(missing_ok=True)
            except Exception:  # pragma: no cover - best-effort cleanup
                pass


@router.post("/compare", name="compare_post", dependencies=[Depends(comparison_rate_limit)])
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
    comparison_mode: str = Form("standard"),
    prereg_source: str = Form("upload"),
    registration_id: str | None = Form(None),
    preregistration: list[UploadFile] = File([]),
    osf_url: str | None = Form(None),
    paper: list[UploadFile] = File([]),
    dimensions_data: str = Form(...),
    visibility: str | None = Form(None),
):
    # Preregistration source: a file upload, a ClinicalTrials.gov registration, or
    # an OSF link. (clinical_registration kept for backward compatibility.)
    source = (prereg_source or "upload").strip().lower()
    is_clinical = source == "clinical" or _bool_from_yes(clinical_registration)
    is_rr = (comparison_mode or "").strip().lower() == "registered_report"
    comparison_type: ComparisonType = (
        "clinical_trials"
        if is_clinical
        else ("registered_report" if is_rr else "general_preregistration")
    )
    # Multiple uploaded files per side are concatenated into one paper / one
    # registration before the (unchanged) single-document pipeline runs.
    upload_dir = Path(request.app.state.settings.upload_dir)
    paper_file = await _coalesce_uploads(paper, upload_dir=upload_dir, kind="paper", max_bytes=MAX_UPLOAD_BYTES)
    prereg_file = await _coalesce_uploads(
        preregistration, upload_dir=upload_dir, kind="registration", max_bytes=MAX_UPLOAD_BYTES
    )
    return await _compare_and_redirect(
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
        preregistration=prereg_file,
        osf_url=osf_url,
        paper=paper_file,
        dimensions_data=dimensions_data,
        visibility=visibility,
    )


@router.post("/clinical_trials", dependencies=[Depends(comparison_rate_limit)])
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
    return await _compare_and_redirect(
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


@router.post("/general_preregistration", dependencies=[Depends(comparison_rate_limit)])
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
    return await _compare_and_redirect(
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


@router.post("/animals_trials", dependencies=[Depends(comparison_rate_limit)])
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
    return await _compare_and_redirect(
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
