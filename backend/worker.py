from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Coroutine

from backend.core.config import get_settings
from backend.core.redis import create_redis_client
from backend.services.comparisons import (
    animals_trial_comparison,
    clinical_trial_comparison,
    general_preregistration_comparison,
    run_with_concurrency_limit,
)

logger = logging.getLogger(__name__)


async def _dispatch_job(job: dict[str, Any], redis_client) -> None:
    comparison_type = job.get("comparison_type")
    task_id = job.get("task_id")

    async def _runner() -> None:
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

    await run_with_concurrency_limit(_runner)


async def worker_loop() -> None:
    settings = get_settings()
    redis_client = create_redis_client(settings.redis_url)
    logger.info("Worker started; waiting for jobs on 'comparison:queue'")

    while True:
        try:
            result = await redis_client.brpop("comparison:queue", timeout=5)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Worker BRPOP failed; retrying", exc_info=exc)
            await asyncio.sleep(5)
            continue
        if result is None:
            continue

        _, raw_job = result
        try:
            job = json.loads(raw_job)
        except Exception as exc:
            logger.error("Failed to decode job payload", exc_info=exc, extra={"raw": raw_job})
            continue

        # Fire-and-forget under semaphore
        asyncio.create_task(_dispatch_job(job, redis_client))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(worker_loop())


if __name__ == "__main__":  # pragma: no cover - entrypoint
    main()
