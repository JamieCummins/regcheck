from __future__ import annotations

import asyncio
import base64
import gzip
import json
import logging
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

logger = logging.getLogger(__name__)


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
        elif comparison_type == "general_preregistration":
            await general_preregistration_comparison(
                job.get("prereg_path", ""),
                job.get("prereg_ext", ""),
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
    except Exception as exc:
        logger.error("Job failed", exc_info=exc, extra={"task_id": task_id})
        if task_id:
            try:
                await redis_client.hset(
                    task_id,
                    mapping={
                        "state": "FAILURE",
                        "status": f"Worker error: {exc}",
                    },
                )
                await redis_client.expire(task_id, settings.task_ttl_seconds)
            except Exception:  # pragma: no cover - best-effort failure status
                logger.warning("Failed to update task status after error", exc_info=exc, extra={"task_id": task_id})
    finally:
        # Always clean up uploaded artifacts stored in S3 for this task.
        if s3_cfg is not None:
            await _cleanup_s3_keys(s3_cfg, s3_keys)
        # Remove temporary local files to reduce disk pressure.
        for path_key in ("paper_path", "prereg_path", "registration_csv_path"):
            value = job.get(path_key)
            if value:
                try:
                    Path(value).unlink(missing_ok=True)
                except Exception:
                    continue


async def worker_loop() -> None:
    settings = get_settings()
    redis_client = create_redis_client(settings.redis_url)
    logger.info("Worker started; waiting for jobs on 'comparison:queue'")

    # Recover any jobs left in the processing queue from a previous crash/restart.
    try:
        stalled = await redis_client.lrange("comparison:processing", 0, -1)
        if stalled:
            await redis_client.rpush("comparison:queue", *stalled)
            await redis_client.delete("comparison:processing")
            logger.info("Recovered %d stalled job(s) from processing queue", len(stalled))
    except Exception as exc:  # pragma: no cover - defensive recovery
        logger.warning("Failed to recover stalled jobs", exc_info=exc)

    while True:
        try:
            raw_job = await redis_client.brpoplpush("comparison:queue", "comparison:processing", timeout=5)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Worker BRPOP failed; retrying", exc_info=exc)
            await asyncio.sleep(5)
            continue
        if raw_job is None:
            continue

        try:
            job = json.loads(raw_job)
        except Exception as exc:
            logger.error("Failed to decode job payload", exc_info=exc, extra={"raw": raw_job})
            try:
                await redis_client.lrem("comparison:processing", 1, raw_job)
            except Exception:
                logger.warning("Failed to remove undecodable job from processing queue", exc_info=exc)
            continue

        # Fire-and-forget under semaphore
        async def _run_and_ack(payload: dict[str, Any], raw_payload: str) -> None:
            try:
                await _dispatch_job(payload, redis_client)
            finally:
                try:
                    await redis_client.lrem("comparison:processing", 1, raw_payload)
                except Exception as exc:  # pragma: no cover - best-effort ack cleanup
                    logger.warning("Failed to remove job from processing queue", exc_info=exc)

        asyncio.create_task(_run_and_ack(job, raw_job))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(worker_loop())


if __name__ == "__main__":  # pragma: no cover - entrypoint
    main()
