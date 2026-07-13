from __future__ import annotations

import asyncio
import base64
import gzip
import json
import logging
import os
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from backend.core.config import get_settings
from backend.core.redis import create_redis_client
from backend.core.storage import get_s3_config, s3_delete_key, s3_download_to_path
from backend.services.comparisons import (
    animals_trial_comparison,
    clinical_trial_comparison,
    general_preregistration_comparison,
    run_with_concurrency_limit,
)
from backend.services.osf import fetch_osf_preregistration
from backend.services import reports as reports_service
from backend.services.report_artifacts import cleanup_expired_s3_artifacts

logger = logging.getLogger(__name__)


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int((os.environ.get(name) or "").strip()))
    except (TypeError, ValueError):
        return default


# A job that keeps crashing the worker *process* (a hard crash/OOM, not a caught
# error — those are marked FAILURE and acked) would otherwise be re-queued
# forever by crash recovery, blocking the queue. Cap the recovery retries and
# dead-letter beyond that.
_MAX_JOB_ATTEMPTS = _int_env("MAX_JOB_ATTEMPTS", 3)

# Backpressure: cap concurrently in-flight jobs so the dequeue loop can't drain
# the whole queue into 'comparison:processing'/memory faster than it runs them.
# Matches the comparison concurrency limit so dequeue tracks actual capacity.
_MAX_INFLIGHT_JOBS = _int_env("MAX_CONCURRENT_COMPARISON_TASKS", 8)


LEGACY_PROCESSING_KEY = "comparison:processing"
_HEARTBEAT_PREFIX = "worker:heartbeat:"
_PROCESSING_PREFIX = "comparison:processing:"
# Heartbeat lives this long; refreshed every loop iteration (≤ the 5s BRPOPLPUSH
# timeout), so a worker that dies stops being "alive" within one TTL. A peer only
# reclaims another worker's in-flight jobs once that worker's heartbeat has lapsed,
# which prevents deploy-overlap job theft (both dynos briefly alive).
WORKER_HEARTBEAT_TTL = _int_env("WORKER_HEARTBEAT_TTL", 30)


def _heartbeat_key(worker_id: str) -> str:
    return f"{_HEARTBEAT_PREFIX}{worker_id}"


def _processing_key(worker_id: str) -> str:
    return f"{_PROCESSING_PREFIX}{worker_id}"


async def _recover_stalled_jobs(
    redis_client,
    *,
    max_attempts: int,
    task_ttl_seconds: int,
    processing_key: str = LEGACY_PROCESSING_KEY,
) -> tuple[int, int]:
    """Reclaim jobs left in ``processing_key`` by a crashed/departed worker.

    Each reclaimed job's attempt count is incremented. Jobs that have exceeded
    ``max_attempts`` (i.e. they keep crashing the worker) are moved to
    ``comparison:deadletter`` and marked FAILURE instead of being re-queued
    forever. Returns ``(requeued, dead_lettered)``.
    """
    stalled = await redis_client.lrange(processing_key, 0, -1)
    if not stalled:
        return (0, 0)

    requeue: list[str] = []
    dead = 0
    for raw in stalled:
        try:
            job = json.loads(raw)
        except Exception:
            # Undecodable payload — cannot retry it meaningfully.
            await redis_client.rpush("comparison:deadletter", raw)
            dead += 1
            continue
        attempts = int(job.get("attempts", 0) or 0) + 1
        job["attempts"] = attempts
        if attempts > max_attempts:
            task_id = job.get("task_id")
            if task_id:
                try:
                    await redis_client.hset(
                        task_id,
                        mapping={
                            "state": "FAILURE",
                            "status": "Worker error: job exceeded retry limit",
                        },
                    )
                    await redis_client.expire(task_id, task_ttl_seconds)
                except Exception:  # pragma: no cover - best-effort status
                    logger.warning(
                        "Failed to mark dead-lettered job FAILURE",
                        extra={"task_id": task_id},
                    )
            await redis_client.rpush("comparison:deadletter", json.dumps(job))
            dead += 1
        else:
            requeue.append(json.dumps(job))

    if requeue:
        await redis_client.rpush("comparison:queue", *requeue)
    await redis_client.delete(processing_key)
    return (len(requeue), dead)


async def _recover_orphaned_processing(
    redis_client, *, my_worker_id: str, max_attempts: int, task_ttl_seconds: int
) -> tuple[int, int]:
    """Reclaim in-flight jobs belonging to workers that are no longer alive.

    Runs at boot and periodically. A per-worker processing list is reclaimed only
    when its owning worker's heartbeat has lapsed — so a peer that is still alive
    (e.g. the old dyno during a deploy overlap) keeps its jobs. The legacy shared
    ``comparison:processing`` list (pre-lease protocol) is always reclaimable.
    """
    total_requeued = 0
    total_dead = 0

    # Legacy shared list from the pre-lease protocol: always orphaned.
    r, d = await _recover_stalled_jobs(
        redis_client,
        max_attempts=max_attempts,
        task_ttl_seconds=task_ttl_seconds,
        processing_key=LEGACY_PROCESSING_KEY,
    )
    total_requeued += r
    total_dead += d

    try:
        keys = await redis_client.keys(f"{_PROCESSING_PREFIX}*")
    except Exception:  # pragma: no cover - defensive
        keys = []
    for key in keys or []:
        key = key.decode("utf-8") if isinstance(key, (bytes, bytearray)) else key
        worker_id = key[len(_PROCESSING_PREFIX):]
        if not worker_id or worker_id == my_worker_id:
            continue  # my own live list
        try:
            alive = await redis_client.exists(_heartbeat_key(worker_id))
        except Exception:  # pragma: no cover - defensive
            alive = 1  # unknown → assume alive (don't steal)
        if alive:
            continue
        r, d = await _recover_stalled_jobs(
            redis_client,
            max_attempts=max_attempts,
            task_ttl_seconds=task_ttl_seconds,
            processing_key=key,
        )
        total_requeued += r
        total_dead += d
    return (total_requeued, total_dead)


async def _restore_upload(redis_client, path: str, redis_key: str | None) -> None:
    """Recreate an uploaded file on the worker dyno if it doesn't exist locally."""
    if not path or Path(path).exists():
        return
    if not redis_key:
        raise FileNotFoundError(f"Missing redis key to restore upload for {path}")
    encoded = await redis_client.get(redis_key)
    if not encoded:
        raise FileNotFoundError(f"Upload payload not found in redis for key {redis_key}")
    try:
        compressed = base64.b64decode(encoded)
        raw = gzip.decompress(compressed)
    except Exception as exc:
        raise ValueError(f"Failed to decode stored upload for key {redis_key}: {exc}") from exc
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)


async def _ensure_s3_upload(cfg, path: str, key: str | None) -> None:
    if not path or Path(path).exists():
        return
    if not key:
        raise FileNotFoundError(f"Missing S3 key to restore upload for {path}")

    def _download() -> None:
        s3_download_to_path(cfg, key=key, path=path)

    await asyncio.to_thread(_download)


async def _cleanup_s3_keys(cfg, keys: dict[str, Any] | None) -> None:
    if cfg is None or not keys:
        return

    def _delete_all() -> None:
        for value in keys.values():
            if isinstance(value, str) and value.strip():
                try:
                    s3_delete_key(cfg, key=value.strip())
                except Exception:
                    continue

    await asyncio.to_thread(_delete_all)


async def _dispatch_job(job: dict[str, Any], redis_client) -> None:
    comparison_type = job.get("comparison_type")
    task_id = job.get("task_id")
    upload_keys = job.get("upload_keys") or {}
    s3_keys = job.get("s3_keys") or {}
    s3_cfg = get_s3_config()
    settings = get_settings()

    # Deletion tombstone: if the report was deleted after this job was queued,
    # don't run it — otherwise the LLM/evidence writes would recreate Redis data
    # for a deleted report (which the viewer gate could then serve).
    if task_id and await reports_service.is_deleted(redis_client, task_id):
        logger.info("Skipping queued job for deleted report", extra={"task_id": task_id})
        return

    async def _runner() -> None:
        # Restore uploads if missing on worker dyno.
        # Prefer S3 when configured, otherwise fall back to legacy Redis blobs.
        paper_path = job.get("paper_path", "") or ""
        prereg_path = job.get("prereg_path", "") or ""
        csv_path = job.get("registration_csv_path", "") or ""
        try:
            if s3_cfg is not None:
                await _ensure_s3_upload(s3_cfg, paper_path, s3_keys.get("paper"))
                await _ensure_s3_upload(s3_cfg, prereg_path, s3_keys.get("prereg"))
                await _ensure_s3_upload(s3_cfg, csv_path, s3_keys.get("csv"))
            else:
                await _restore_upload(redis_client, paper_path, upload_keys.get("paper"))
                await _restore_upload(redis_client, prereg_path, upload_keys.get("prereg"))
                await _restore_upload(redis_client, csv_path, upload_keys.get("csv"))
        except Exception as exc:
            logger.error("Failed to restore uploads for job", exc_info=exc, extra={"task_id": task_id})
            raise

        if comparison_type == "clinical_trials":
            await clinical_trial_comparison(
                job.get("registration_id", ""),
                job.get("paper_path", ""),
                job.get("paper_ext", ""),
                job.get("client", "openai"),
                task_id,
                redis_client,
                parser_choice=job.get("parser_choice", "grobid"),
                reasoning_effort=job.get("reasoning_effort"),
                selected_dimensions=job.get("selected_dimensions"),
                append_previous_output=job.get("append_previous_output", False),
            )
        elif comparison_type in ("general_preregistration", "registered_report"):
            prereg_path = job.get("prereg_path", "") or ""
            prereg_ext = job.get("prereg_ext", "") or ""
            osf_url = job.get("osf_url")
            if osf_url:
                # Resolve the OSF link to a local prereg file (registration → text,
                # or hosted file → download). Sync requests, off the event loop.
                prereg_path, prereg_ext = await asyncio.to_thread(
                    fetch_osf_preregistration, osf_url, dest_dir=settings.upload_dir
                )
            paper_path_g = job.get("paper_path", "") or ""
            # Log the resolved inputs so a read/parse failure can be traced to the
            # exact file + extension (esp. for OSF, where the prereg is fetched here).
            logger.info(
                "general_preregistration inputs resolved",
                extra={
                    "task_id": task_id,
                    "osf_url": osf_url,
                    "prereg_ext": prereg_ext,
                    "prereg_exists": bool(prereg_path) and Path(prereg_path).exists(),
                    "prereg_size": (Path(prereg_path).stat().st_size if prereg_path and Path(prereg_path).exists() else None),
                    "paper_ext": job.get("paper_ext"),
                    "paper_exists": bool(paper_path_g) and Path(paper_path_g).exists(),
                    "paper_size": (Path(paper_path_g).stat().st_size if paper_path_g and Path(paper_path_g).exists() else None),
                    "multiple_experiments": job.get("multiple_experiments"),
                },
            )
            await general_preregistration_comparison(
                prereg_path,
                prereg_ext,
                job.get("paper_path", ""),
                job.get("paper_ext", ""),
                job.get("client", "openai"),
                job.get("parser_choice", "grobid"),
                task_id,
                redis_client,
                job.get("selected_dimensions"),
                append_previous_output=job.get("append_previous_output", False),
                reasoning_effort=job.get("reasoning_effort"),
                multiple_experiments=job.get("multiple_experiments"),
                experiment_number=job.get("experiment_number"),
                experiment_text=job.get("experiment_text"),
                comparison_context=("registered_report" if comparison_type == "registered_report" else "preregistration"),
            )
        elif comparison_type == "animals_trials":
            await animals_trial_comparison(
                job.get("registration_id", ""),
                job.get("paper_path", ""),
                job.get("paper_ext", ""),
                job.get("client", "openai"),
                registration_csv_path=job.get("registration_csv_path"),
                parser_choice=job.get("parser_choice", "grobid"),
                task_id=task_id,
                redis_client=redis_client,
                selected_dimensions=job.get("selected_dimensions"),
                append_previous_output=job.get("append_previous_output", False),
                reasoning_effort=job.get("reasoning_effort"),
            )
        else:
            logger.warning("Unknown comparison_type in job", extra={"job": job})

    try:
        await run_with_concurrency_limit(_runner)
        # Deletion may have arrived WHILE the job ran; if so, remove everything the
        # run just wrote so a deleted report can't come back to life.
        if task_id and await reports_service.is_deleted(redis_client, task_id):
            logger.info("Report deleted mid-run; cleaning up job output", extra={"task_id": task_id})
            await reports_service.delete_report_everywhere(redis_client, None, task_id, None)
            return
    except Exception as exc:
        logger.error("Job failed", exc_info=exc, extra={"task_id": task_id})
        # Surface the exception type + originating code location in the status so a
        # failure is diagnosable from the report page even when the worker's stdout
        # logs aren't captured by the platform's log drain.
        loc = ""
        tb = traceback.extract_tb(exc.__traceback__)
        app_frames = [f for f in tb if "/backend/" in f.filename.replace("\\", "/")]
        if app_frames:
            last = app_frames[-1]
            loc = f" [{Path(last.filename).name}:{last.lineno} in {last.name}]"
        status_msg = f"Worker error: {type(exc).__name__}: {exc}{loc}"[:480]
        if task_id:
            try:
                await redis_client.hset(
                    task_id,
                    mapping={
                        "state": "FAILURE",
                        "status": status_msg,
                    },
                )
                await redis_client.expire(task_id, settings.task_ttl_seconds)
            except Exception:  # pragma: no cover - best-effort failure status
                logger.warning("Failed to update task status after error", exc_info=exc, extra={"task_id": task_id})
    finally:
        # Durable uploads (S3/Redis) are intentionally RETAINED so a report can be
        # regenerated from its original files; they expire with the report window
        # (Redis TTL) and are removed when the report is deleted. Only the ephemeral
        # local temp copies are cleaned here to keep worker disk pressure down.
        for path_key in ("paper_path", "prereg_path", "registration_csv_path"):
            value = job.get(path_key)
            if value:
                try:
                    Path(value).unlink(missing_ok=True)
                except Exception:
                    continue


def _runtime_release() -> str:
    """Describe the code this dyno is running, for the boot log — so "is the worker
    on the latest deploy?" is answerable at a glance (the recurring failure mode).
    The release/commit vars need ``heroku labs:enable runtime-dyno-metadata``."""
    rel = (os.environ.get("HEROKU_RELEASE_VERSION") or "").strip()
    commit = (os.environ.get("HEROKU_SLUG_COMMIT") or "").strip()
    created = (os.environ.get("HEROKU_RELEASE_CREATED_AT") or "").strip()
    dyno = (os.environ.get("DYNO") or "").strip()
    parts = []
    if rel:
        parts.append(f"release={rel}")
    if commit:
        parts.append(f"commit={commit[:8]}")
    if created:
        parts.append(f"released_at={created}")
    if dyno:
        parts.append(f"dyno={dyno}")
    if not (rel or commit):
        parts.append("release=unknown (enable `heroku labs:enable runtime-dyno-metadata`)")
    return " | ".join(parts)


async def worker_loop() -> None:
    settings = get_settings()
    # Worker uses blocking Redis commands (BRPOPLPUSH). Give reads a bit more
    # headroom than the command timeout to avoid spurious socket read timeouts.
    redis_client = create_redis_client(settings.redis_url, socket_timeout=15)
    worker_id = uuid.uuid4().hex
    processing_key = _processing_key(worker_id)
    heartbeat_key = _heartbeat_key(worker_id)
    logger.info(
        "Worker started [%s] id=%s; waiting for jobs on 'comparison:queue'",
        _runtime_release(),
        worker_id,
    )

    # On deploy/scale events, Redis may not be immediately reachable. Warm up the
    # connection with retries so the worker doesn't crash-loop or spam errors.
    for attempt in range(10):
        try:
            await redis_client.ping()
            break
        except Exception as exc:  # pragma: no cover - environment dependent
            delay = min(10, 0.5 * (attempt + 1))
            logger.warning(
                "Redis ping failed; retrying",
                extra={"attempt": attempt + 1, "delay_seconds": delay, "dynd": os.environ.get("DYNO")},
                exc_info=exc,
            )
            await asyncio.sleep(delay)

    async def _refresh_heartbeat() -> None:
        try:
            await redis_client.set(heartbeat_key, str(time.time()), ex=WORKER_HEARTBEAT_TTL)
        except Exception:  # pragma: no cover - best-effort liveness
            logger.warning("Failed to refresh worker heartbeat", exc_info=True)

    async def _recover(reason: str) -> None:
        try:
            requeued, dead = await _recover_orphaned_processing(
                redis_client,
                my_worker_id=worker_id,
                max_attempts=_MAX_JOB_ATTEMPTS,
                task_ttl_seconds=settings.task_ttl_seconds,
            )
            if requeued or dead:
                logger.info(
                    "Recovered orphaned jobs (%s): requeued=%d dead_lettered=%d", reason, requeued, dead
                )
        except Exception as exc:  # pragma: no cover - defensive recovery
            logger.warning("Failed to recover orphaned jobs", exc_info=exc)

    # Claim the heartbeat before recovery so a peer can't mistake us for dead, then
    # reclaim jobs orphaned by workers that ARE dead (crash/restart/old deploy).
    await _refresh_heartbeat()
    await _recover("boot")

    inflight = asyncio.Semaphore(_MAX_INFLIGHT_JOBS)

    while True:
        now = time.time()
        await _refresh_heartbeat()
        last_cleanup = getattr(worker_loop, "_last_report_cleanup", 0.0)
        if now - last_cleanup >= 60:
            setattr(worker_loop, "_last_report_cleanup", now)
            try:
                await cleanup_expired_s3_artifacts(redis_client)
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                logger.warning("Report artifact cleanup failed", exc_info=exc)
        # Periodically reclaim jobs from workers whose heartbeat has since lapsed
        # (e.g. the old dyno finally exiting after a deploy overlap).
        last_recover = getattr(worker_loop, "_last_orphan_recovery", 0.0)
        if now - last_recover >= WORKER_HEARTBEAT_TTL:
            setattr(worker_loop, "_last_orphan_recovery", now)
            await _recover("periodic")

        # Block here (not on the queue) when at capacity, so jobs stay queued
        # rather than being drained into 'processing'/memory.
        await inflight.acquire()
        try:
            raw_job = await redis_client.brpoplpush("comparison:queue", processing_key, timeout=5)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Worker BRPOP failed; retrying", exc_info=exc)
            inflight.release()
            await asyncio.sleep(5)
            continue
        if raw_job is None:
            inflight.release()
            continue

        try:
            job = json.loads(raw_job)
        except Exception as exc:
            logger.error("Failed to decode job payload", exc_info=exc, extra={"raw": raw_job})
            try:
                await redis_client.lrem(processing_key, 1, raw_job)
            except Exception:
                logger.warning("Failed to remove undecodable job from processing queue", exc_info=exc)
            inflight.release()
            continue

        # Fire-and-forget; the in-flight slot is released when the job finishes.
        async def _run_and_ack(payload: dict[str, Any], raw_payload: str) -> None:
            try:
                await _dispatch_job(payload, redis_client)
            finally:
                try:
                    await redis_client.lrem(processing_key, 1, raw_payload)
                except Exception as exc:  # pragma: no cover - best-effort ack cleanup
                    logger.warning("Failed to remove job from processing queue", exc_info=exc)
                inflight.release()

        asyncio.create_task(_run_and_ack(job, raw_job))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(worker_loop())


if __name__ == "__main__":  # pragma: no cover - entrypoint
    main()
