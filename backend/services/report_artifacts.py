from __future__ import annotations

import asyncio
import base64
import gzip
import json
import time
from copy import deepcopy
from typing import Any

from ..core.storage import (
    get_persist_s3_config,
    get_s3_config,
    s3_config_for_bucket,
    s3_copy_object,
    s3_delete_key,
    s3_get_bytes,
    s3_put_bytes,
)

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


def _s3_artifact_key(redis_key: str) -> str:
    # Mirror the Redis key under the shared regcheck/ prefix as an S3 path.
    return "regcheck/" + redis_key.replace(":", "/")


async def _store_artifact_bytes(
    redis_client,
    *,
    redis_key: str,
    data: bytes,
    ttl_seconds: int | None,
    content_type: str | None = None,
) -> dict[str, Any]:
    """Persist a report evidence artifact.

    Large binary artifacts (original documents, rendered PDFs, render data) go to
    S3 — object storage that's durable, cheap, and out of Redis RAM — whenever a
    bucket is configured; otherwise (local dev) they fall back to Redis
    (gzip+base64).

    Routing follows the report's retention: persisted reports (``ttl_seconds is
    None``, i.e. signed-in) go to the persistent bucket and are kept; anonymous
    reports (finite ttl) go to the temp bucket, which auto-expires via an S3
    lifecycle rule (the cleanup schedule below is a belt-and-suspenders backstop).
    The entry records its ``storage``+``bucket``+``key`` so reads/deletes target
    the right bucket and pre-existing Redis artifacts keep working through a cutover.
    """
    cfg = get_persist_s3_config() if ttl_seconds is None else get_s3_config()
    if cfg is not None:
        s3_key = _s3_artifact_key(redis_key)

        def _put() -> None:
            s3_put_bytes(cfg, key=s3_key, data=data, content_type=content_type)

        await asyncio.to_thread(_put)
        if ttl_seconds is not None:
            try:
                await redis_client.zadd(S3_CLEANUP_ZSET, {s3_key: time.time() + ttl_seconds})
            except Exception:  # pragma: no cover - best-effort scheduling
                pass
        return {"storage": "s3", "bucket": cfg.bucket, "key": s3_key, "compressed_bytes": len(data)}

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
        if not key:
            raise RuntimeError(f"{source_id}.{artifact_name} is missing an artifact key")
        if storage == "redis":
            if not await redis_client.exists(key):
                raise RuntimeError(f"{source_id}.{artifact_name} Redis artifact key was not visible after save")
        elif storage == "s3":
            # s3_put_bytes is synchronous and raises on failure; new objects are
            # read-after-write consistent, so a returned put means it's present.
            continue
        else:
            raise RuntimeError(
                f"{source_id}.{artifact_name} uses unsupported report artifact storage: {storage or 'missing'}"
            )

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
        content_type="application/json",
    )

    if raw_bytes is not None:
        filename = str(source.get("raw_filename") or f"{source_id}.bin")
        raw_artifact = await _store_artifact_bytes(
            redis_client,
            redis_key=_raw_key(task_id, source_id),
            data=raw_bytes,
            ttl_seconds=ttl_seconds,
            content_type=raw_content_type,
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
    """Delete a report's evidence artifacts: the manifest and every per-source
    render/raw blob it references, across both Redis and S3."""
    try:
        manifest = await load_manifest(redis_client, task_id)
    except Exception:
        manifest = None

    redis_keys = [manifest_key(task_id)]
    s3_targets: list[tuple[str | None, str]] = []  # (bucket, key)
    if manifest:
        for _source_id, _artifact_name, artifact in _iter_manifest_artifacts(manifest):
            if artifact.get("storage") == "s3" and artifact.get("key"):
                s3_targets.append((artifact.get("bucket"), str(artifact["key"])))
        # Also clear any legacy/Redis-stored per-source blobs.
        for source_id in (manifest.get("sources") or {}):
            redis_keys.append(_render_key(task_id, str(source_id)))
            redis_keys.append(_raw_key(task_id, str(source_id)))

    try:
        await redis_client.delete(*redis_keys)
    except Exception:  # pragma: no cover - best effort cleanup
        pass

    if s3_targets:
        def _delete_all() -> None:
            for bucket, key in s3_targets:
                cfg = s3_config_for_bucket(bucket) if bucket else get_s3_config()
                if cfg is None:
                    continue
                try:
                    s3_delete_key(cfg, key=key)
                except Exception:
                    continue
        await asyncio.to_thread(_delete_all)
        try:  # drop any temp-bucket keys from the expiry schedule too
            await redis_client.zrem(S3_CLEANUP_ZSET, *[key for _bucket, key in s3_targets])
        except Exception:  # pragma: no cover - best effort
            pass


async def migrate_artifacts_to_persist(redis_client, task_id: str) -> int:
    """Move a report's evidence to the persistent S3 bucket + drop its expiry, so a
    claimed (formerly anonymous) report's source documents don't auto-delete.

    Server-side-copies each temp-bucket artifact to the persist bucket, rewrites the
    manifest's bucket refs, unschedules cleanup, and persists Redis-stored artifacts
    + the manifest. When S3 isn't configured it just persists the Redis keys. Best
    effort per artifact; returns the number of S3 objects moved.
    """
    manifest = await load_manifest(redis_client, task_id)
    if not manifest:
        return 0

    persist_cfg = get_persist_s3_config()
    cleanup_keys: list[str] = []
    redis_persist_keys: list[str] = []
    s3_moves: list[tuple[Any, str, dict]] = []  # (src_cfg, key, artifact dict)

    for _source_id, _name, artifact in _iter_manifest_artifacts(manifest):
        key = artifact.get("key")
        if not key:
            continue
        storage = artifact.get("storage")
        if storage == "redis":
            redis_persist_keys.append(str(key))
        elif storage == "s3":
            bucket = artifact.get("bucket")
            if persist_cfg is not None and bucket == persist_cfg.bucket:
                cleanup_keys.append(str(key))  # already persistent; just unschedule
                continue
            src_cfg = s3_config_for_bucket(bucket) if bucket else get_s3_config()
            if persist_cfg is None or src_cfg is None:
                cleanup_keys.append(str(key))  # nowhere to move it; drop from schedule
                continue
            s3_moves.append((src_cfg, str(key), artifact))

    if s3_moves:
        def _copy_all() -> None:
            for src_cfg, key, _artifact in s3_moves:
                try:
                    s3_copy_object(src_cfg, persist_cfg, key=key)
                except Exception:
                    continue
        await asyncio.to_thread(_copy_all)
        for _src, key, artifact in s3_moves:
            artifact["bucket"] = persist_cfg.bucket  # mutates the manifest in place
            cleanup_keys.append(key)

    for key in redis_persist_keys:
        try:
            await redis_client.persist(key)
        except Exception:  # pragma: no cover - best effort
            pass
    # Re-store the manifest with updated bucket refs and no expiry.
    try:
        await store_manifest(redis_client, task_id=task_id, manifest=manifest, ttl_seconds=None)
    except Exception:  # pragma: no cover - best effort
        pass
    if cleanup_keys:
        try:
            await redis_client.zrem(S3_CLEANUP_ZSET, *cleanup_keys)
        except Exception:  # pragma: no cover - best effort
            pass
    return len(s3_moves)


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
        # Read from whichever bucket the artifact was written to (temp vs
        # persist); fall back to the default bucket for entries without one.
        bucket = artifact.get("bucket")
        cfg = s3_config_for_bucket(bucket) if bucket else get_s3_config()
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
