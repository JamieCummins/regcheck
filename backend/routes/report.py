from __future__ import annotations

import json
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from ..services.report_artifacts import load_artifact_bytes, load_manifest, public_manifest

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
    manifest = await _manifest_or_404(request, task_id)
    return JSONResponse(public_manifest(manifest, task_id), headers=PRIVATE_HEADERS)


@router.get("/report/{task_id}/sources/{source_id}/render-data")
async def report_source_render_data(request: Request, task_id: str, source_id: str):
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
