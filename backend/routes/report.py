from __future__ import annotations

import json
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from datetime import date

from ..services import sharing as sharing_service
from ..services.report_artifacts import (
    load_artifact_bytes,
    load_manifest,
    offline_bundle,
    public_manifest,
)
from ..services.report_html import build_standalone_report_html

try:  # pragma: no cover - optional dependency
    import fitz
except ModuleNotFoundError:  # pragma: no cover
    fitz = None

router = APIRouter()

PRIVATE_HEADERS = {
    "Cache-Control": "private, no-store, max-age=0",
    "Pragma": "no-cache",
    "X-Robots-Tag": "noindex, nofollow",
}


async def _ensure_access(request: Request, task_id: str) -> None:
    """403 unless the viewer may see this report (restricted reports only)."""
    verdict = await sharing_service.ensure_viewable(request, task_id)
    if not verdict.allowed:
        raise HTTPException(status_code=403, detail="You do not have access to this report")


async def _manifest_or_404(request: Request, task_id: str) -> dict:
    manifest = await load_manifest(request.app.state.redis, task_id)
    if not manifest:
        raise HTTPException(status_code=404, detail="Report evidence artifacts not found")
    return manifest


def _source_or_404(manifest: dict, source_id: str) -> dict:
    source = (manifest.get("sources") or {}).get(source_id)
    if not isinstance(source, dict):
        raise HTTPException(status_code=404, detail="Report source not found")
    return source


@router.get("/report/{task_id}/manifest")
async def report_manifest(request: Request, task_id: str):
    await _ensure_access(request, task_id)
    manifest = await _manifest_or_404(request, task_id)
    return JSONResponse(public_manifest(manifest, task_id), headers=PRIVATE_HEADERS)


@router.get("/report/{task_id}/export.html")
async def report_export_html(request: Request, task_id: str):
    """Download the report as a single self-contained HTML file — the same interactive
    viewer the CLI's --report-html produces, with the evidence bundle inlined so it
    renders fully offline (text rendering + quote-string tracing, no server)."""
    await _ensure_access(request, task_id)
    redis_client = request.app.state.redis
    manifest = await _manifest_or_404(request, task_id)

    # Items come from the same place the live viewer reads them (the status hash), so
    # the export is consistent with what the report currently shows.
    items: list = []
    title = "RegCheck Report"
    try:
        data = await redis_client.hgetall(task_id)
    except Exception:  # pragma: no cover - defensive
        data = None
    if data:
        def _f(key: str):
            return data.get(key) if key in data else data.get(key.encode())
        result_raw = _f("result_json")
        if isinstance(result_raw, bytes):
            result_raw = result_raw.decode("utf-8", "replace")
        if result_raw:
            try:
                items = (json.loads(result_raw) or {}).get("items") or []
            except json.JSONDecodeError:  # pragma: no cover - defensive
                items = []
        name_raw = _f("report_name")
        if isinstance(name_raw, bytes):
            name_raw = name_raw.decode("utf-8", "replace")
        if name_raw and str(name_raw).strip():
            title = str(name_raw).strip()

    offline_manifest, render_data = await offline_bundle(redis_client, manifest)
    document = build_standalone_report_html(
        title=title,
        items=items,
        manifest=offline_manifest,
        render_data=render_data,
        meta=f"Exported from RegCheck on {date.today().isoformat()}",
    )
    filename = "regcheck-report.html"
    headers = {
        **PRIVATE_HEADERS,
        "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}",
    }
    return Response(content=document, media_type="text/html; charset=utf-8", headers=headers)


@router.get("/report/{task_id}/sources/{source_id}/render-data")
async def report_source_render_data(request: Request, task_id: str, source_id: str):
    await _ensure_access(request, task_id)
    manifest = await _manifest_or_404(request, task_id)
    source = _source_or_404(manifest, source_id)
    artifact = (source.get("_artifacts") or {}).get("render")
    data = await load_artifact_bytes(request.app.state.redis, artifact)
    if data is None:
        raise HTTPException(status_code=404, detail="Source render data not found")
    try:
        payload = json.loads(data.decode("utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail="Source render data is invalid") from exc
    return JSONResponse(payload, headers=PRIVATE_HEADERS)


@router.get("/report/{task_id}/sources/{source_id}/raw")
async def report_source_raw(request: Request, task_id: str, source_id: str):
    await _ensure_access(request, task_id)
    manifest = await _manifest_or_404(request, task_id)
    source = _source_or_404(manifest, source_id)
    artifact = (source.get("_artifacts") or {}).get("raw")
    data = await load_artifact_bytes(request.app.state.redis, artifact)
    if data is None:
        raise HTTPException(status_code=404, detail="Raw source not found")
    content_type = artifact.get("content_type") or "application/octet-stream"
    filename = artifact.get("filename") or source.get("raw_filename") or f"{source_id}.bin"
    headers = {
        **PRIVATE_HEADERS,
        "Content-Disposition": f"inline; filename*=UTF-8''{quote(str(filename))}",
    }
    return Response(content=data, media_type=content_type, headers=headers)


@router.get("/report/{task_id}/sources/{source_id}/pages/{page_number}.png")
async def report_source_page_png(request: Request, task_id: str, source_id: str, page_number: int):
    if fitz is None:
        raise HTTPException(status_code=503, detail="PDF rendering is unavailable")
    if page_number < 1:
        raise HTTPException(status_code=404, detail="Page not found")

    await _ensure_access(request, task_id)
    manifest = await _manifest_or_404(request, task_id)
    source = _source_or_404(manifest, source_id)
    if source.get("kind") != "pdf":
        raise HTTPException(status_code=404, detail="Source is not a PDF")
    artifact = (source.get("_artifacts") or {}).get("raw")
    data = await load_artifact_bytes(request.app.state.redis, artifact)
    if data is None:
        raise HTTPException(status_code=404, detail="PDF source not found")

    doc = fitz.open(stream=data, filetype="pdf")
    try:
        if page_number > len(doc):
            raise HTTPException(status_code=404, detail="Page not found")
        page = doc.load_page(page_number - 1)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
        png = pixmap.tobytes("png")
    finally:
        doc.close()

    return Response(content=png, media_type="image/png", headers=PRIVATE_HEADERS)
