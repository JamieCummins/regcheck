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

import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import requests

logger = logging.getLogger(__name__)

__all__ = ["extract_osf_guid", "fetch_osf_preregistration"]

def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int((os.environ.get(name) or "").strip()))
    except (TypeError, ValueError):
        return default


# OSF — especially the WaterButler file-download path — is occasionally slow to
# respond. Use a short connect timeout but a generous read timeout, and retry
# transient failures so a momentary OSF hiccup doesn't fail the whole job.
_CONNECT_TIMEOUT = 10
_API_READ_TIMEOUT = _int_env("OSF_API_TIMEOUT", 30)
_DOWNLOAD_READ_TIMEOUT = _int_env("OSF_DOWNLOAD_TIMEOUT", 60)
_MAX_ATTEMPTS = _int_env("OSF_MAX_ATTEMPTS", 3)
_RETRY_STATUS = {502, 503, 504}
# Cap the fetched file like direct uploads (MAX_UPLOAD_BYTES in the upload route)
# so a huge OSF file can't fill the dyno's disk/memory.
_MAX_DOWNLOAD_BYTES = _int_env("OSF_MAX_DOWNLOAD_BYTES", 20 * 1024 * 1024)
_SUPPORTED_EXTS = {".pdf", ".docx", ".txt", ".html", ".htm"}
_BARE_GUID_RE = re.compile(r"^[a-z0-9]{5,}$", re.IGNORECASE)
_PATH_AFTER_HOST_RE = re.compile(r"osf\.io/(.+)", re.IGNORECASE)
_GUID_SEGMENT_RE = re.compile(r"^[a-z0-9]{5,}$", re.IGNORECASE)
# Path words that look GUID-like but are OSF routes/storage providers, not GUIDs.
_RESERVED_SEGMENTS = {
    "download", "files", "osfstorage", "github", "dropbox", "googledrive",
    "figshare", "owncloud", "onedrive", "bitbucket", "gitlab", "dataverse",
    "render", "settings", "registrations", "forks", "metadata", "resources",
    "components", "analytics", "addons", "project", "quickfiles", "preprints",
    "preprint", "wiki",
}


def _is_guid_segment(segment: str) -> bool:
    return bool(_GUID_SEGMENT_RE.match(segment)) and segment.lower() not in _RESERVED_SEGMENTS


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
    """Pull the OSF GUID from a URL or accept a bare GUID. Returns None if absent.

    Handles the common shapes: ``osf.io/<guid>``, ``osf.io/<guid>/download``,
    ``osf.io/download/<guid>``, and file-browser URLs like
    ``osf.io/<node-or-registration>/files/[<provider>/]<file-guid>``. For the
    last, the *file* GUID (the entity the user pointed at) is returned rather
    than the containing node/registration — otherwise a link to a specific file
    would resolve to its parent project and never fetch the file.
    """
    text = (url_or_id or "").strip()
    if not text:
        return None
    if _BARE_GUID_RE.match(text):
        return text.lower()

    match = _PATH_AFTER_HOST_RE.search(text)
    if not match:
        return None
    path = match.group(1).split("?", 1)[0].split("#", 1)[0]
    segments = [s for s in path.split("/") if s]
    if not segments:
        return None
    lowered = [s.lower() for s in segments]

    # File-browser URL: the file GUID is the last guid-like segment after /files/.
    if "files" in lowered:
        after = [s for s in segments[lowered.index("files") + 1 :] if _is_guid_segment(s)]
        if after:
            return after[-1].lower()

    # Explicit download URLs: the GUID sits adjacent to the 'download' segment.
    if "download" in lowered:
        di = lowered.index("download")
        if di + 1 < len(segments) and _is_guid_segment(segments[di + 1]):
            return segments[di + 1].lower()
        if di - 1 >= 0 and _is_guid_segment(segments[di - 1]):
            return segments[di - 1].lower()

    # Otherwise the first guid-like segment (e.g. osf.io/<guid>).
    for segment in segments:
        if _is_guid_segment(segment):
            return segment.lower()
    return None


def _get_with_retry(url: str, *, headers: dict[str, str], read_timeout: int, stream: bool = False):
    """GET with a short connect + generous read timeout, retrying transient
    network errors and 5xx responses so OSF's occasional slowness doesn't fail
    the job. Raises a friendly ValueError if OSF stays unreachable."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            resp = requests.get(
                url,
                headers=headers,
                timeout=(_CONNECT_TIMEOUT, read_timeout),
                allow_redirects=True,
                stream=stream,
            )
            if resp.status_code in _RETRY_STATUS and attempt < _MAX_ATTEMPTS - 1:
                last_exc = requests.HTTPError(f"OSF returned {resp.status_code}")
                logger.warning(
                    "OSF %s on attempt %d/%d for %s; retrying",
                    resp.status_code, attempt + 1, _MAX_ATTEMPTS, url,
                )
                time.sleep(1.5 * (attempt + 1))
                continue
            return resp
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            logger.warning(
                "OSF request failed on attempt %d/%d for %s: %s",
                attempt + 1, _MAX_ATTEMPTS, url, exc,
            )
            if attempt < _MAX_ATTEMPTS - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
    raise ValueError(
        "Couldn't reach OSF to fetch the preregistration — it timed out or was "
        "unavailable after several attempts. OSF may be slow right now; try again "
        "in a moment, or download the file and upload it directly."
    ) from last_exc


def _resolve_guid(guid: str) -> dict[str, Any]:
    """Resolve a GUID to its typed object. /v2/guids/{guid}/ redirects to the
    typed endpoint, so the returned `data` carries the full attributes."""
    resp = _get_with_retry(f"{_api_base()}/guids/{guid}/", headers=_headers(), read_timeout=_API_READ_TIMEOUT)
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
        for key in sorted(responses, key=_response_sort_key):
            flat = _flatten_value(responses[key]).strip()
            if flat:
                lines.append(f"{key}: {flat}")
    return "\n\n".join(line for line in lines if line is not None).strip()


def _response_sort_key(key: str) -> tuple:
    """Order registration responses by their question NUMBER, not lexicographically
    — plain `sorted` puts "q10" before "q2", scrambling the prereg so sections read
    out of order (and content looks missing). e.g. "q17.question" → (17, "question")."""
    m = re.match(r"[A-Za-z]*(\d+)(?:\.(.*))?$", str(key))
    if m:
        return (0, int(m.group(1)), m.group(2) or "")
    return (1, 0, str(key))


def _download_file(data: dict[str, Any], dest_dir: str | Path, guid: str) -> tuple[str, str]:
    attrs = data.get("attributes", {}) or {}
    links = data.get("links", {}) or {}
    name = attrs.get("name") or ""
    download_url = links.get("download") or f"https://osf.io/{guid}/download"
    resp = _get_with_retry(
        download_url,
        headers=_download_headers(),
        read_timeout=_DOWNLOAD_READ_TIMEOUT,
        stream=True,
    )
    resp.raise_for_status()

    # Guard against content negotiation returning JSON metadata instead of the file.
    resp_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
    if resp_type in ("application/json", "application/vnd.api+json"):
        resp.close()
        raise ValueError(
            "OSF returned file metadata instead of the file itself. Use the file's "
            "direct link (or its download URL) and ensure the file is public."
        )

    # Fail fast when OSF declares a size beyond the cap (streaming still enforces
    # the cap when Content-Length is absent or wrong).
    declared = (resp.headers.get("content-length") or "").strip()
    if declared.isdigit() and int(declared) > _MAX_DOWNLOAD_BYTES:
        resp.close()
        raise ValueError(
            f"That OSF file is larger than the {_MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB "
            "limit. Upload a smaller version of the document directly."
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
        resp.close()
        raise ValueError(
            f"That OSF file type ('{ext or 'unknown'}') isn't supported. "
            "Link to a PDF, DOCX, TXT, or HTML file."
        )

    dest = Path(dest_dir) / f"osf_{guid}_{uuid.uuid4().hex}{ext}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                written += len(chunk)
                if written > _MAX_DOWNLOAD_BYTES:
                    raise ValueError(
                        f"That OSF file is larger than the {_MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB "
                        "limit. Upload a smaller version of the document directly."
                    )
                fh.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    finally:
        resp.close()
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
