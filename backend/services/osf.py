"""Fetch a preregistration from an OSF (Open Science Framework) link.

A pasted OSF URL can point to either:
  * a **registration** ("template") — we read its form responses via the OSF API
    and flatten them into readable text; or
  * a **file** hosted on OSF storage — we download it and process it like any
    upload (PDF/DOCX/TXT/HTML).

OSF web pages are JS single-page apps, so registration content must come from the
API (https://api.osf.io/v2), not the page HTML. Only the public API is used by
default; set OSF_TOKEN for private content. Mirrors the sync `requests` pattern in
trials.py; the worker calls this via asyncio.to_thread.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Any, Callable

import requests

__all__ = ["extract_osf_guid", "fetch_osf_preregistration"]

_TIMEOUT = 30
_SUPPORTED_EXTS = {".pdf", ".docx", ".txt", ".html", ".htm"}
# OSF GUID right after the host (optionally behind a /download/ segment).
_GUID_RE = re.compile(r"osf\.io/(?:download/)?([a-z0-9]{5,})", re.IGNORECASE)
_BARE_GUID_RE = re.compile(r"^[a-z0-9]{5,}$", re.IGNORECASE)


def _api_base() -> str:
    return (os.environ.get("OSF_API_BASE") or "https://api.osf.io/v2").rstrip("/")


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.api+json"}
    token = (os.environ.get("OSF_TOKEN") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _download_headers() -> dict[str, str]:
    """Headers for binary file downloads. This must NOT request JSON — OSF's file
    server (WaterButler) content-negotiates, so an `Accept: application/vnd.api+json`
    can make it return JSON *metadata* instead of the file bytes (which then isn't a
    valid PDF/DOCX and breaks parsing). Only auth is forwarded."""
    token = (os.environ.get("OSF_TOKEN") or "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def extract_osf_guid(url_or_id: str) -> str | None:
    """Pull the OSF GUID from a URL (osf.io/<guid>, .../<guid>/download,
    osf.io/download/<guid>, …) or accept a bare GUID. Returns None if absent."""
    text = (url_or_id or "").strip()
    if not text:
        return None
    match = _GUID_RE.search(text)
    if match:
        return match.group(1).lower()
    if _BARE_GUID_RE.match(text):
        return text.lower()
    return None


def _resolve_guid(guid: str) -> dict[str, Any]:
    """Resolve a GUID to its typed object. /v2/guids/{guid}/ redirects to the
    typed endpoint, so the returned `data` carries the full attributes."""
    resp = requests.get(f"{_api_base()}/guids/{guid}/", headers=_headers(), timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("data", {}) or {}


def _flatten_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_flatten_value(item) for item in value if item not in (None, ""))
    if isinstance(value, dict):
        if "value" in value:
            return _flatten_value(value.get("value"))
        return "; ".join(f"{k}: {_flatten_value(v)}" for k, v in value.items())
    return "" if value is None else str(value)


def _registration_to_text(data: dict[str, Any]) -> str:
    """Flatten an OSF registration's form responses into readable text. The
    answers are the content; the qN keys are opaque labels but kept for context."""
    attrs = data.get("attributes", {}) or {}
    lines: list[str] = []
    if attrs.get("title"):
        lines.append(f"Title: {attrs['title']}")
    if attrs.get("description"):
        lines.append(f"Description: {attrs['description']}")

    responses = attrs.get("registration_responses") or {}
    if not responses:
        # Older registrations expose answers under registered_meta[key]["value"].
        meta = attrs.get("registered_meta") or {}
        responses = {key: val.get("value") if isinstance(val, dict) else val for key, val in meta.items()}

    if responses:
        lines.append("")
        for key in sorted(responses):
            flat = _flatten_value(responses[key]).strip()
            if flat:
                lines.append(f"{key}: {flat}")
    return "\n\n".join(line for line in lines if line is not None).strip()


def _download_file(data: dict[str, Any], dest_dir: str | Path, guid: str) -> tuple[str, str]:
    attrs = data.get("attributes", {}) or {}
    links = data.get("links", {}) or {}
    name = attrs.get("name") or ""
    download_url = links.get("download") or f"https://osf.io/{guid}/download"
    resp = requests.get(download_url, headers=_download_headers(), timeout=_TIMEOUT, allow_redirects=True)
    resp.raise_for_status()

    # Guard against content negotiation returning JSON metadata instead of the file.
    resp_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
    if resp_type in ("application/json", "application/vnd.api+json"):
        raise ValueError(
            "OSF returned file metadata instead of the file itself. Use the file's "
            "direct link (or its download URL) and ensure the file is public."
        )

    ext = Path(name).suffix.lower()
    if not ext:
        content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
        ext = {
            "application/pdf": ".pdf",
            "text/html": ".html",
            "text/plain": ".txt",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        }.get(content_type, "")
    if ext not in _SUPPORTED_EXTS:
        raise ValueError(
            f"That OSF file type ('{ext or 'unknown'}') isn't supported. "
            "Link to a PDF, DOCX, TXT, or HTML file."
        )

    dest = Path(dest_dir) / f"osf_{guid}_{uuid.uuid4().hex}{ext}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    return str(dest), ext


def fetch_osf_preregistration(
    url: str,
    *,
    dest_dir: str | Path = "uploads",
    resolver: Callable[[str], dict[str, Any]] | None = None,
    file_downloader: Callable[[dict[str, Any], str | Path, str], tuple[str, str]] | None = None,
) -> tuple[str, str]:
    """Resolve an OSF link to a local (path, extension) for the comparison pipeline.

    Registration → flattened form text written to a .txt. File → downloaded with
    its real extension. Project/unknown → ValueError with guidance.
    `resolver`/`file_downloader` are injectable for testing.
    """
    guid = extract_osf_guid(url)
    if not guid:
        raise ValueError(
            "Couldn't find an OSF identifier in that link. Paste an OSF registration "
            "or file URL, e.g. https://osf.io/abc12/."
        )

    data = (resolver or _resolve_guid)(guid)
    osf_type = (data.get("type") or "").lower()

    if osf_type == "registrations":
        text = _registration_to_text(data)
        if not text.strip():
            raise ValueError("That OSF registration has no readable form responses to compare.")
        dest = Path(dest_dir) / f"osf_{guid}_{uuid.uuid4().hex}.txt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        return str(dest), ".txt"

    if osf_type == "files":
        return (file_downloader or _download_file)(data, dest_dir, guid)

    if osf_type == "nodes":
        raise ValueError(
            "That link points to an OSF project, not a registration or file. "
            "Link to a specific registration, or to a file (its page or download URL)."
        )

    raise ValueError(
        f"Unsupported OSF item type ('{osf_type or 'unknown'}'). "
        "Link to an OSF registration or a hosted file."
    )
