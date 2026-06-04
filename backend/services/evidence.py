from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

from .embeddings import extract_chunks_tokens_with_spans

try:  # pragma: no cover - optional dependency
    import fitz
except ModuleNotFoundError:  # pragma: no cover
    fitz = None


def _chunk_id(prefix: str, index: int) -> str:
    return f"{prefix.upper()}_{index:04d}"


def _content_type_for_path(path: str) -> str:
    guessed, _encoding = mimetypes.guess_type(path)
    return guessed or "application/octet-stream"


def _plain_text_render_data(text: str, *, kind: str = "text", metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "kind": kind,
        "text": text or "",
        "metadata": metadata or {},
    }


def build_text_evidence_source(
    *,
    source_id: str,
    label: str,
    text: str,
    chunk_prefix: str,
    kind: str = "text",
    metadata: dict[str, Any] | None = None,
    raw_bytes: bytes | None = None,
    raw_content_type: str | None = None,
    raw_filename: str | None = None,
    max_chunk_tokens: int = 300,
    embedding_model: str = "text-embedding-3-large",
) -> dict[str, Any]:
    chunks = extract_chunks_tokens_with_spans(
        text or "",
        max_chunk_tokens=max_chunk_tokens,
        encoding_name=embedding_model,
    )
    segments = [chunk.text for chunk in chunks]
    chunk_metadata: list[dict[str, Any]] = []
    manifest_chunks: dict[str, dict[str, Any]] = {}
    for index, chunk in enumerate(chunks, start=1):
        cid = _chunk_id(chunk_prefix, index)
        location = {
            "kind": "text",
            "start": chunk.start,
            "end": chunk.end,
        }
        item = {
            "id": cid,
            "source_id": source_id,
            "source_label": label,
            "source_kind": kind,
            "text": chunk.text,
            "locations": [location],
            "relevance_scores_by_dimension": {},
        }
        manifest_chunks[cid] = item
        chunk_metadata.append(item)

    source = {
        "id": source_id,
        "label": label,
        "kind": kind,
        "render_mode": "text",
        "raw_filename": raw_filename,
        "metadata": metadata or {},
    }
    return {
        "source": source,
        "text": text or "",
        "segments": segments,
        "metadata": chunk_metadata,
        "chunks": manifest_chunks,
        "raw_bytes": raw_bytes,
        "raw_content_type": raw_content_type,
        "render_data": _plain_text_render_data(text or "", kind=kind, metadata=metadata),
    }


def _pdf_page_text_and_offsets(doc) -> tuple[str, list[dict[str, Any]]]:
    pages: list[dict[str, Any]] = []
    parts: list[str] = []
    offset = 0
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        page_text = page.get_text("text") or ""
        rect = page.rect
        if parts:
            parts.append("\n\n")
            offset += 2
        start = offset
        parts.append(page_text)
        offset += len(page_text)
        pages.append(
            {
                "page_number": page_index + 1,
                "start": start,
                "end": offset,
                "width": float(rect.width),
                "height": float(rect.height),
            }
        )
    return "".join(parts), pages


def _text_slice_for_page(text: str, chunk_start: int, chunk_end: int, page_info: dict[str, Any]) -> str:
    start = max(chunk_start, int(page_info["start"]))
    end = min(chunk_end, int(page_info["end"]))
    if start >= end:
        return ""
    return text[start:end].strip()


def _search_page_rects(page, text: str) -> list[dict[str, float]]:
    candidate = " ".join((text or "").split())
    if not candidate:
        return []
    attempts = [candidate]
    if len(candidate) > 240:
        attempts.append(candidate[:240].rsplit(" ", 1)[0] or candidate[:240])
    if len(candidate) > 120:
        attempts.append(candidate[:120].rsplit(" ", 1)[0] or candidate[:120])
    for attempt in attempts:
        try:
            rects = page.search_for(attempt)
        except Exception:
            rects = []
        if rects:
            return [
                {
                    "x0": float(rect.x0),
                    "y0": float(rect.y0),
                    "x1": float(rect.x1),
                    "y1": float(rect.y1),
                }
                for rect in rects
            ]
    return []


def _pdf_locations_for_chunk(doc, full_text: str, chunk_start: int, chunk_end: int, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    locations: list[dict[str, Any]] = []
    for page_info in pages:
        if chunk_end < page_info["start"] or chunk_start > page_info["end"]:
            continue
        page_text = _text_slice_for_page(full_text, chunk_start, chunk_end, page_info)
        if not page_text:
            continue
        page = doc.load_page(int(page_info["page_number"]) - 1)
        rects = _search_page_rects(page, page_text)
        if rects:
            locations.append(
                {
                    "kind": "pdf",
                    "page": page_info["page_number"],
                    "page_width": page_info["width"],
                    "page_height": page_info["height"],
                    "rects": rects,
                }
            )
    return locations


def build_pdf_evidence_source(
    *,
    source_id: str,
    label: str,
    pdf_path: str,
    chunk_prefix: str,
    fallback_text: str | None = None,
    metadata: dict[str, Any] | None = None,
    max_chunk_tokens: int = 300,
    embedding_model: str = "text-embedding-3-large",
) -> dict[str, Any]:
    path = Path(pdf_path)
    raw_bytes = path.read_bytes() if path.exists() else None
    raw_filename = path.name or f"{source_id}.pdf"
    base_source = {
        "id": source_id,
        "label": label,
        "kind": "pdf",
        "raw_filename": raw_filename,
        "metadata": metadata or {},
    }

    if fitz is None or raw_bytes is None:
        return build_text_evidence_source(
            source_id=source_id,
            label=label,
            text=fallback_text or "",
            chunk_prefix=chunk_prefix,
            kind="text",
            metadata={"fallback_reason": "PDF rendering unavailable", **(metadata or {})},
            raw_bytes=raw_bytes,
            raw_content_type="application/pdf" if raw_bytes else None,
            raw_filename=raw_filename if raw_bytes else None,
            max_chunk_tokens=max_chunk_tokens,
            embedding_model=embedding_model,
        )

    try:
        doc = fitz.open(str(path))
    except Exception:
        return build_text_evidence_source(
            source_id=source_id,
            label=label,
            text=fallback_text or "",
            chunk_prefix=chunk_prefix,
            kind="text",
            metadata={"fallback_reason": "PDF could not be opened for highlighting", **(metadata or {})},
            raw_bytes=raw_bytes,
            raw_content_type="application/pdf",
            raw_filename=raw_filename,
            max_chunk_tokens=max_chunk_tokens,
            embedding_model=embedding_model,
        )
    try:
        pdf_text, pages = _pdf_page_text_and_offsets(doc)
        use_pdf_text = bool((pdf_text or "").strip())
        source_text = pdf_text if use_pdf_text else (fallback_text or "")
        render_mode = "pdf" if use_pdf_text else "text"
        chunks = extract_chunks_tokens_with_spans(
            source_text,
            max_chunk_tokens=max_chunk_tokens,
            encoding_name=embedding_model,
        )
        segments = [chunk.text for chunk in chunks]
        chunk_metadata: list[dict[str, Any]] = []
        manifest_chunks: dict[str, dict[str, Any]] = {}
        for index, chunk in enumerate(chunks, start=1):
            cid = _chunk_id(chunk_prefix, index)
            if use_pdf_text:
                locations = _pdf_locations_for_chunk(doc, source_text, chunk.start, chunk.end, pages)
                if not locations:
                    locations = [{"kind": "text", "start": chunk.start, "end": chunk.end}]
            else:
                locations = [{"kind": "text", "start": chunk.start, "end": chunk.end}]
            item = {
                "id": cid,
                "source_id": source_id,
                "source_label": label,
                "source_kind": "pdf",
                "text": chunk.text,
                "locations": locations,
                "relevance_scores_by_dimension": {},
            }
            manifest_chunks[cid] = item
            chunk_metadata.append(item)

        source = {
            **base_source,
            "render_mode": render_mode,
            "page_count": len(doc),
            "pages": [
                {
                    "page_number": page["page_number"],
                    "width": page["width"],
                    "height": page["height"],
                }
                for page in pages
            ],
        }
        render_data = {
            "kind": "pdf",
            "render_mode": render_mode,
            "text": source_text,
            "pages": source["pages"],
            "metadata": metadata or {},
        }
        return {
            "source": source,
            "text": source_text,
            "segments": segments,
            "metadata": chunk_metadata,
            "chunks": manifest_chunks,
            "raw_bytes": raw_bytes,
            "raw_content_type": "application/pdf",
            "render_data": render_data,
        }
    finally:
        doc.close()


def _flatten_json_rows(data: dict[str, Any], parent: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, value in data.items():
        path = f"{parent} → {key}" if parent else str(key)
        if isinstance(value, dict):
            rows.extend(_flatten_json_rows(value, path))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                list_path = f"{path}[{index}]"
                if isinstance(item, dict):
                    rows.extend(_flatten_json_rows(item, list_path))
                elif item not in [None, ""]:
                    rows.append({"path": list_path, "value": str(item)})
        elif value not in [None, ""]:
            rows.append({"path": path, "value": str(value)})
    return rows


def build_json_evidence_source(
    *,
    source_id: str,
    label: str,
    data: dict[str, Any],
    chunk_prefix: str,
    metadata: dict[str, Any] | None = None,
    max_chunk_tokens: int = 300,
    embedding_model: str = "text-embedding-3-large",
) -> dict[str, Any]:
    rows = _flatten_json_rows(data)
    text_parts: list[str] = []
    offset = 0
    render_rows: list[dict[str, Any]] = []
    for row in rows:
        line = f"{row['path']}\n{row['value']}"
        if text_parts:
            text_parts.append("\n\n")
            offset += 2
        start = offset
        text_parts.append(line)
        offset += len(line)
        render_rows.append(
            {
                "path": row["path"],
                "value": row["value"],
                "start": start,
                "end": offset,
            }
        )
    text = "".join(text_parts)
    payload = build_text_evidence_source(
        source_id=source_id,
        label=label,
        text=text,
        chunk_prefix=chunk_prefix,
        kind="json",
        metadata=metadata,
        raw_bytes=text.encode("utf-8"),
        raw_content_type="text/plain; charset=utf-8",
        raw_filename=f"{source_id}.txt",
        max_chunk_tokens=max_chunk_tokens,
        embedding_model=embedding_model,
    )
    payload["source"]["kind"] = "json"
    payload["source"]["render_mode"] = "json"
    payload["render_data"] = {
        "kind": "json",
        "text": text,
        "rows": render_rows,
        "metadata": metadata or {},
    }
    for chunk in payload["chunks"].values():
        chunk["source_kind"] = "json"
        for location in chunk["locations"]:
            location["kind"] = "json"
    return payload


def build_file_evidence_source(
    *,
    source_id: str,
    label: str,
    file_path: str,
    file_ext: str,
    text: str,
    chunk_prefix: str,
    metadata: dict[str, Any] | None = None,
    max_chunk_tokens: int = 300,
    embedding_model: str = "text-embedding-3-large",
) -> dict[str, Any]:
    ext = (file_ext or Path(file_path).suffix).lower()
    if ext == ".pdf":
        return build_pdf_evidence_source(
            source_id=source_id,
            label=label,
            pdf_path=file_path,
            chunk_prefix=chunk_prefix,
            fallback_text=text,
            metadata=metadata,
            max_chunk_tokens=max_chunk_tokens,
            embedding_model=embedding_model,
        )

    path = Path(file_path)
    raw_bytes = path.read_bytes() if path.exists() else None
    return build_text_evidence_source(
        source_id=source_id,
        label=label,
        text=text,
        chunk_prefix=chunk_prefix,
        kind="text",
        metadata=metadata,
        raw_bytes=raw_bytes,
        raw_content_type=_content_type_for_path(str(path)) if raw_bytes is not None else None,
        raw_filename=path.name if raw_bytes is not None else None,
        max_chunk_tokens=max_chunk_tokens,
        embedding_model=embedding_model,
    )
