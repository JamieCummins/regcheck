from __future__ import annotations

import json
import os
import logging
from typing import Any

import httpx
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import fitz  # PyMuPDF
except ModuleNotFoundError:  # pragma: no cover
    fitz = None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _has_usable_text(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if cleaned.lower() == "body tag not found.":
        return False
    return True


def _is_low_text(text: str, *, min_chars: int | None = None) -> bool:
    threshold = min_chars if min_chars is not None else _env_int("MIN_PDF_EXTRACTED_TEXT_CHARS", 200)
    cleaned = (text or "").strip()
    if not cleaned or cleaned.lower() == "body tag not found.":
        return True
    return len(cleaned) < max(0, threshold)


def is_likely_scanned_pdf(
    filename: str,
    *,
    max_pages: int | None = None,
    min_text_chars: int | None = None,
    min_images: int | None = None,
) -> bool:
    """Heuristic: returns True when PDF pages have images but little/no selectable text."""
    if fitz is None:  # pragma: no cover - optional dependency
        return False
    page_cap = max_pages if max_pages is not None else _env_int("SCANNED_PDF_SCAN_PAGES", 3)
    text_threshold = min_text_chars if min_text_chars is not None else _env_int("SCANNED_PDF_MIN_TEXT_CHARS", 40)
    image_threshold = min_images if min_images is not None else _env_int("SCANNED_PDF_MIN_IMAGES", 1)
    try:
        doc = fitz.open(filename)
    except Exception:  # pragma: no cover - defensive
        return False
    try:
        pages = min(len(doc), max(1, page_cap))
        total_text = 0
        total_images = 0
        for i in range(pages):
            page = doc.load_page(i)
            try:
                total_text += len((page.get_text("text") or "").strip())
            except Exception:
                pass
            try:
                total_images += len(page.get_images(full=True) or [])
            except Exception:
                pass
        return total_images >= image_threshold and total_text < text_threshold
    finally:
        try:
            doc.close()
        except Exception:
            pass


def _fallback_mode() -> str:
    return (os.environ.get("SCANNED_PDF_FALLBACK") or "none").strip().lower()


async def pdf2grobid(
    filename: str,
    grobid_url: str | None = None,
) -> str:
    grobid_url = (grobid_url or os.environ.get("GROBID_URL") or "").strip() or (
        "https://kermitt2-grobid.hf.space/api/processFulltextDocument"
    )
    timeout = httpx.Timeout(60.0, read=60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        with open(filename, "rb") as file:
            files = {"input": file}
            response = await client.post(grobid_url, files=files)
    response.raise_for_status()
    return response.text


async def pdf2dpt(
    filename: str,
    dpt_url: str | None = None,
) -> dict[str, Any]:
    api_key = (os.environ.get("DPT_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Missing DPT_API_KEY")
    dpt_url = (dpt_url or os.environ.get("DPT_URL") or "").strip() or (
        "https://api.va.eu-west-1.landing.ai/v1/ade/parse"
    )
    headers = {"Authorization": api_key}
    data = {"model": "dpt-2-latest"}
    timeout_seconds = float(os.environ.get("DPT_TIMEOUT_SECONDS", "240") or 240)
    timeout = httpx.Timeout(timeout_seconds, read=timeout_seconds, connect=30.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            with open(filename, "rb") as document:
                files = {"document": document}
                response = await client.post(
                    dpt_url, headers=headers, data=data, files=files
                )
    except httpx.ReadTimeout as exc:
        raise RuntimeError("DPT parsing timed out; please retry or use grobid parser") from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"DPT parsing failed: {exc}") from exc
    response.raise_for_status()
    return response.json()


def extract_dpt_text(payload: Any) -> str:
    """Recursively collect textual content from a DPT response payload."""
    chunks: list[str] = []

    def _walk(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, str):
            text = node.strip()
            if text:
                chunks.append(text)
            return
        if isinstance(node, dict):
            for key, value in node.items():
                key_lower = str(key).lower()
                if isinstance(value, str) and any(token in key_lower for token in ("text", "content", "paragraph", "body")):
                    _walk(value)
                else:
                    _walk(value)
            return
        if isinstance(node, list):
            for item in node:
                _walk(item)
            return

    _walk(payload)
    if chunks:
        return "\n\n".join(chunks)
    try:
        return json.dumps(payload, ensure_ascii=False)
    except Exception:
        return str(payload)


def extract_body_text(xml_content: str) -> str:
    namespace = {"tei": "http://www.tei-c.org/ns/1.0"}
    root = ET.fromstring(xml_content)
    body = root.find(".//tei:body", namespace)
    if body is not None:
        return "".join(body.itertext()).strip()
    return "Body tag not found."


async def extract_pdf_text(
    filename: str,
    *,
    parser_choice: str = "grobid",
    pdf_parser: Any | None = None,
    dpt_parser: Any | None = None,
) -> tuple[str, str]:
    """Extract paper text from PDF; optionally fall back for scanned PDFs.

    Returns (extracted_text, used_parser_label).
    """
    normalized = (parser_choice or "grobid").strip().lower()
    if normalized not in {"grobid", "dpt2"}:
        raise ValueError(f"Unsupported parser choice: {parser_choice}")

    if normalized == "dpt2":
        parser_callable = dpt_parser or pdf2dpt
        payload = await parser_callable(filename)
        extracted = extract_dpt_text(payload)
        if not _has_usable_text(extracted):
            raise ValueError("Parsed PDF but extracted no usable text (DPT2).")
        return extracted, "dpt2"

    parser_callable = pdf_parser or pdf2grobid
    xml_payload = await parser_callable(filename)
    extracted = extract_body_text(xml_payload)
    if _has_usable_text(extracted) and not (_is_low_text(extracted) and is_likely_scanned_pdf(filename)):
        return extracted, "grobid"

    # Grobid produced little/no text. If this looks like a scanned PDF, either instruct or fall back.
    if is_likely_scanned_pdf(filename):
        mode = _fallback_mode()
        if mode == "dpt2":
            parser_callable = dpt_parser or pdf2dpt
            if parser_callable is pdf2dpt and not (os.environ.get("DPT_API_KEY") or "").strip():
                raise RuntimeError("SCANNED_PDF_FALLBACK=dpt2 requires DPT_API_KEY.")
            logger.info("Scanned PDF detected; falling back to DPT2", extra={"filename": filename})
            payload = await parser_callable(filename)
            extracted2 = extract_dpt_text(payload)
            if not _has_usable_text(extracted2):
                raise ValueError("Scanned PDF fallback (DPT2) produced no usable text.")
            return extracted2, "dpt2_fallback"
        raise ValueError(
            "PDF appears to be scanned (images with little/no selectable text). "
            "Select the DPT2 parser (OCR) or set SCANNED_PDF_FALLBACK=dpt2 with DPT_API_KEY."
        )

    raise ValueError(
        "Parsed PDF but extracted no usable text. "
        "If this is a scanned PDF, select the DPT2 parser (OCR)."
    )
