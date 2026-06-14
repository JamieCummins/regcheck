from __future__ import annotations

import asyncio
import base64
import gzip
import json
import time
from copy import deepcopy
from typing import Any

from ..core.storage import get_s3_config, s3_delete_key, s3_get_bytes

MANIFEST_KEY_TEMPLATE = "report:{task_id}:manifest"
SOURCE_RAW_KEY_TEMPLATE = "report:{task_id}:source:{source_id}:raw"
SOURCE_RENDER_KEY_TEMPLATE = "report:{task_id}:source:{source_id}:render"
S3_CLEANUP_ZSET = "report:s3_artifact_cleanup"


def manifest_key(task_id: str) -> str:
    return MANIFEST_KEY_TEMPLATE.format(task_id=task_id)


def _raw_key(task_id: str, source_id: str) -> str:
    return SOURCE_RAW_KEY_TEMPLATE.format(task_id=task_id, source_id=source_id)


def _render_key(task_id: str, source_id: str) -> str:
    return SOURCE_RENDER_KEY_TEMPLATE.format(task_id=task_id, source_id=source_id)


async def _redis_set_bytes(redis_client, key: str, data: bytes, ttl_seconds: int) -> int:
    compressed = gzip.compress(data)
    encoded = base64.b64encode(compressed).decode("ascii")
    await redis_client.set(key, encoded, ex=ttl_seconds)
    return len(compressed)


async def _redis_get_bytes(redis_client, key: str) -> bytes | None:
    value = await redis_client.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        value = base64.b64decode(value.encode("ascii"))
    try:
        return gzip.decompress(value)
    except OSError:
        return gzip.decompress(base64.b64decode(value))


async def _store_artifact_bytes(
    redis_client,
    *,
    redis_key: str,
    data: bytes,
    ttl_seconds: int,
) -> dict[str, Any]:
    """Persist report evidence artifacts in Redis.

    S3 may be configured for upload handoff between web and worker dynos, but
    completed report evidence artifacts live in Redis so the status and viewer
    paths share one storage backend.
    """
    compressed_bytes = await _redis_set_bytes(redis_client, redis_key, data, ttl_seconds)
    return {"storage": "redis", "key": redis_key, "compressed_bytes": compressed_bytes}


def _iter_manifest_artifacts(manifest: dict[str, Any]):
    sources = manifest.get("sources") or {}
    for source_id, source in sources.items():
        if not isinstance(source, dict):
            continue
        artifacts = source.get("_artifacts") or {}
        if not isinstance(artifacts, dict):
            continue
        for artifact_name, artifact in artifacts.items():
            if isinstance(artifact, dict):
                yield str(source_id), str(artifact_name), artifact


def manifest_artifact_stats(manifest: dict[str, Any]) -> dict[str, int]:
    artifact_count = 0
    artifact_bytes = 0
    for _source_id, _artifact_name, artifact in _iter_manifest_artifacts(manifest):
        artifact_count += 1
        try:
            artifact_bytes += int(artifact.get("compressed_bytes") or 0)
        except (TypeError, ValueError):
            pass
    return {
        "source_count": len(manifest.get("sources") or {}),
        "chunk_count": len(manifest.get("chunks") or {}),
        "artifact_count": artifact_count,
        "artifact_bytes": artifact_bytes,
    }


async def verify_manifest_artifacts(
    redis_client,
    *,
    task_id: str,
    manifest: dict[str, Any],
) -> dict[str, int]:
    if not await redis_client.exists(manifest_key(task_id)):
        raise RuntimeError("manifest key was not visible after save")

    for source_id, artifact_name, artifact in _iter_manifest_artifacts(manifest):
        storage = artifact.get("storage")
        key = artifact.get("key")
        if storage != "redis":
            raise RuntimeError(
                f"{source_id}.{artifact_name} uses unsupported report artifact storage: {storage or 'missing'}"
            )
        if not key:
            raise RuntimeError(f"{source_id}.{artifact_name} is missing a Redis artifact key")
        if not await redis_client.exists(key):
            raise RuntimeError(f"{source_id}.{artifact_name} Redis artifact key was not visible after save")

    return manifest_artifact_stats(manifest)


async def store_source_artifacts(
    redis_client,
    *,
    task_id: str,
    source: dict[str, Any],
    raw_bytes: bytes | None,
    raw_content_type: str | None,
    render_data: dict[str, Any],
    ttl_seconds: int,
) -> dict[str, Any]:
    """Persist source artifacts and return the source manifest entry with internal refs."""
    source_id = str(source.get("id") or "")
    if not source_id:
        raise ValueError("Source manifest entry requires an id")

    source_entry = dict(source)
    artifacts: dict[str, Any] = {}

    render_bytes = json.dumps(render_data, ensure_ascii=False).encode("utf-8")
    artifacts["render"] = await _store_artifact_bytes(
        redis_client,
        redis_key=_render_key(task_id, source_id),
        data=render_bytes,
        ttl_seconds=ttl_seconds,
    )

    if raw_bytes is not None:
        filename = str(source.get("raw_filename") or f"{source_id}.bin")
        raw_artifact = await _store_artifact_bytes(
            redis_client,
            redis_key=_raw_key(task_id, source_id),
            data=raw_bytes,
            ttl_seconds=ttl_seconds,
        )
        artifacts["raw"] = {
            **raw_artifact,
            "content_type": raw_content_type,
            "filename": filename,
        }
        source_entry["raw_available"] = True
    else:
        source_entry["raw_available"] = False

    source_entry["_artifacts"] = artifacts
    return source_entry


async def store_manifest(
    redis_client,
    *,
    task_id: str,
    manifest: dict[str, Any],
    ttl_seconds: int,
) -> None:
    await redis_client.set(
        manifest_key(task_id),
        json.dumps(manifest, ensure_ascii=False),
        ex=ttl_seconds,
    )


async def load_manifest(redis_client, task_id: str) -> dict[str, Any] | None:
    payload = await redis_client.get(manifest_key(task_id))
    if payload is None:
        return None
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    return json.loads(payload)


async def delete_report_artifacts(redis_client, task_id: str) -> None:
    """Delete a report's Redis evidence artifacts: the manifest and every
    per-source render/raw blob it references."""
    keys = [manifest_key(task_id)]
    try:
        manifest = await load_manifest(redis_client, task_id)
    except Exception:
        manifest = None
    if manifest:
        for source_id in (manifest.get("sources") or {}):
            keys.append(_render_key(task_id, str(source_id)))
            keys.append(_raw_key(task_id, str(source_id)))
    try:
        await redis_client.delete(*keys)
    except Exception:  # pragma: no cover - best effort cleanup
        pass


async def load_artifact_bytes(redis_client, artifact: dict[str, Any] | None) -> bytes | None:
    if not artifact:
        return None
    storage = artifact.get("storage")
    key = artifact.get("key")
    if not key:
        return None
    if storage == "redis":
        return await _redis_get_bytes(redis_client, key)
    if storage == "s3":
        cfg = get_s3_config()
        if cfg is None:
            return None

        def _get() -> bytes:
            return s3_get_bytes(cfg, key=key)

        return await asyncio.to_thread(_get)
    return None


def public_manifest(manifest: dict[str, Any], task_id: str) -> dict[str, Any]:
    """Return a browser-safe copy of the manifest without storage internals."""
    payload = deepcopy(manifest)
    for source_id, source in (payload.get("sources") or {}).items():
        if not isinstance(source, dict):
            continue
        source.pop("_artifacts", None)
        source["render_data_url"] = f"/report/{task_id}/sources/{source_id}/render-data"
        if source.get("raw_available"):
            source["raw_url"] = f"/report/{task_id}/sources/{source_id}/raw"
        if source.get("kind") == "pdf":
            source["page_url_template"] = f"/report/{task_id}/sources/{source_id}/pages/{{page_number}}.png"
    return payload


async def cleanup_expired_s3_artifacts(redis_client, *, limit: int = 100) -> int:
    cfg = get_s3_config()
    if cfg is None:
        return 0
    now = time.time()
    try:
        keys = await redis_client.zrangebyscore(S3_CLEANUP_ZSET, 0, now, start=0, num=limit)
    except Exception:
        return 0
    decoded_keys = [
        key.decode("utf-8") if isinstance(key, bytes) else str(key)
        for key in keys
    ]
    if not decoded_keys:
        return 0

    def _delete_all() -> None:
        for key in decoded_keys:
            try:
                s3_delete_key(cfg, key=key)
            except Exception:
                continue

    await asyncio.to_thread(_delete_all)
    try:
        await redis_client.zrem(S3_CLEANUP_ZSET, *decoded_keys)
    except Exception:
        pass
    return len(decoded_keys)
