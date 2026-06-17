import asyncio
import json
import os

# backend.worker imports services.comparisons, which instantiates a Groq client
# at module load; satisfy it so this module imports in isolation too.
os.environ.setdefault("GROQ_API_KEY", "test")

from backend.worker import _recover_stalled_jobs


class FakeRedis:
    """Minimal async stand-in for the worker's Redis usage in recovery."""

    def __init__(self) -> None:
        self.lists: dict[str, list[str]] = {}
        self.hashes: dict[str, dict] = {}
        self.expires: dict[str, int] = {}

    async def lrange(self, key, start, end):
        items = self.lists.get(key, [])
        if end == -1:
            return list(items[start:])
        return list(items[start : end + 1])

    async def rpush(self, key, *values):
        self.lists.setdefault(key, []).extend(values)
        return len(self.lists[key])

    async def delete(self, key):
        self.lists.pop(key, None)
        self.hashes.pop(key, None)

    async def hset(self, key, mapping=None, **kwargs):
        bucket = self.hashes.setdefault(key, {})
        if mapping:
            bucket.update(mapping)
        bucket.update(kwargs)

    async def expire(self, key, ttl):
        self.expires[key] = ttl


def test_recover_requeues_jobs_under_limit():
    redis = FakeRedis()
    redis.lists["comparison:processing"] = [
        json.dumps({"task_id": "a"}),
        json.dumps({"task_id": "b", "attempts": 1}),
    ]

    requeued, dead = asyncio.run(
        _recover_stalled_jobs(redis, max_attempts=3, task_ttl_seconds=100)
    )

    assert (requeued, dead) == (2, 0)
    assert "comparison:processing" not in redis.lists  # cleared
    queued = [json.loads(x) for x in redis.lists["comparison:queue"]]
    # Attempt counts incremented on reclaim.
    assert {(j["task_id"], j["attempts"]) for j in queued} == {("a", 1), ("b", 2)}
    assert "comparison:deadletter" not in redis.lists


def test_recover_dead_letters_poison_message():
    redis = FakeRedis()
    # Already at the limit; the next reclaim pushes it over.
    redis.lists["comparison:processing"] = [json.dumps({"task_id": "boom", "attempts": 3})]

    requeued, dead = asyncio.run(
        _recover_stalled_jobs(redis, max_attempts=3, task_ttl_seconds=100)
    )

    assert (requeued, dead) == (0, 1)
    assert "comparison:queue" not in redis.lists  # not retried
    dl = [json.loads(x) for x in redis.lists["comparison:deadletter"]]
    assert dl[0]["task_id"] == "boom" and dl[0]["attempts"] == 4
    # Task marked FAILURE with a TTL so the status surfaces to the user.
    assert redis.hashes["boom"]["state"] == "FAILURE"
    assert redis.expires["boom"] == 100


def test_recover_dead_letters_undecodable_payload():
    redis = FakeRedis()
    redis.lists["comparison:processing"] = ["{not valid json"]

    requeued, dead = asyncio.run(
        _recover_stalled_jobs(redis, max_attempts=3, task_ttl_seconds=100)
    )

    assert (requeued, dead) == (0, 1)
    assert redis.lists["comparison:deadletter"] == ["{not valid json"]


def test_recover_noop_when_processing_empty():
    redis = FakeRedis()
    assert asyncio.run(
        _recover_stalled_jobs(redis, max_attempts=3, task_ttl_seconds=100)
    ) == (0, 0)
