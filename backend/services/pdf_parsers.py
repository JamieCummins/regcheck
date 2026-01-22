from __future__ import annotations

import json
import os
from typing import Any

import httpx
import xml.etree.ElementTree as ET


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
