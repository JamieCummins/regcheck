import asyncio
import json

import backend.worker as worker_mod
from backend.worker import _recover_stalled_jobs


class FakeRedis:
    """Minimal async stand-in for the worker's Redis usage in recovery."""

    def __init__(self) -> None:
        self.lists: dict[str, list[str]] = {}
        self.hashes: dict[str, dict] = {}
        self.expires: dict[str, int] = {}
        self.strings: dict[str, str] = {}

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

    async def keys(self, pattern):
        prefix = pattern[:-1] if pattern.endswith("*") else pattern
        return [k for k in self.lists if k.startswith(prefix)]

    async def exists(self, key):
        return 1 if (key in self.strings or key in self.hashes or key in self.lists) else 0

    async def set(self, key, value, ex=None, nx=False):
        if nx and key in self.strings:
            return None
        self.strings[key] = value
        return True


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


# --- _dispatch_job routing/cleanup integration ------------------------------

def _make_job(tmp_path, **over):
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"%PDF-1.4 stub")
    job = {"task_id": "t1", "paper_path": str(paper), "paper_ext": ".pdf", "client": "openai"}
    job.update(over)
    return job, paper


def _no_s3(monkeypatch):
    # Keep the upload-restore path entirely local (no AWS calls).
    monkeypatch.setattr(worker_mod, "get_s3_config", lambda: None)


def test_dispatch_routes_general_preregistration(tmp_path, monkeypatch):
    _no_s3(monkeypatch)
    calls = {}

    async def stub(*args, **kwargs):
        calls["general"] = True

    monkeypatch.setattr(worker_mod, "general_preregistration_comparison", stub)
    prereg = tmp_path / "prereg.txt"
    prereg.write_text("p")
    job, paper = _make_job(
        tmp_path, comparison_type="general_preregistration",
        prereg_path=str(prereg), prereg_ext=".txt",
    )

    asyncio.run(worker_mod._dispatch_job(job, FakeRedis()))

    assert calls.get("general")
    # Local temp files are unlinked in the finally regardless of outcome.
    assert not paper.exists() and not prereg.exists()


def test_dispatch_routes_clinical_trials(tmp_path, monkeypatch):
    _no_s3(monkeypatch)
    calls = {}

    async def stub(*args, **kwargs):
        calls["clinical"] = True

    monkeypatch.setattr(worker_mod, "clinical_trial_comparison", stub)
    job, paper = _make_job(tmp_path, comparison_type="clinical_trials", registration_id="NCT1")

    asyncio.run(worker_mod._dispatch_job(job, FakeRedis()))

    assert calls.get("clinical")
    assert not paper.exists()


def test_dispatch_routes_animals_trials(tmp_path, monkeypatch):
    _no_s3(monkeypatch)
    calls = {}

    async def stub(*args, **kwargs):
        calls["animals"] = True

    monkeypatch.setattr(worker_mod, "animals_trial_comparison", stub)
    job, paper = _make_job(tmp_path, comparison_type="animals_trials", registration_id="X")

    asyncio.run(worker_mod._dispatch_job(job, FakeRedis()))

    assert calls.get("animals")


def test_dispatch_failure_sets_failure_status_and_still_cleans_up(tmp_path, monkeypatch):
    _no_s3(monkeypatch)

    async def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(worker_mod, "general_preregistration_comparison", boom)
    prereg = tmp_path / "prereg.txt"
    prereg.write_text("p")
    job, paper = _make_job(
        tmp_path, comparison_type="general_preregistration",
        prereg_path=str(prereg), prereg_ext=".txt",
    )
    redis = FakeRedis()

    asyncio.run(worker_mod._dispatch_job(job, redis))

    assert redis.hashes["t1"]["state"] == "FAILURE"
    assert "kaboom" in redis.hashes["t1"]["status"]
    # Cleanup runs even on failure.
    assert not paper.exists() and not prereg.exists()


def test_dispatch_unknown_type_is_noop(tmp_path, monkeypatch):
    _no_s3(monkeypatch)
    job, paper = _make_job(tmp_path, comparison_type="bogus")
    # Should neither raise nor mark FAILURE; just logs and cleans up.
    redis = FakeRedis()
    asyncio.run(worker_mod._dispatch_job(job, redis))
    assert "t1" not in redis.hashes
    assert not paper.exists()


def test_orphan_recovery_reclaims_dead_worker_but_not_live_peer():
    from backend.worker import _recover_orphaned_processing, _heartbeat_key, _processing_key

    redis = FakeRedis()
    live, dead, me = "wLIVE", "wDEAD", "wME"
    redis.lists[_processing_key(live)] = [json.dumps({"task_id": "a"})]
    redis.lists[_processing_key(dead)] = [json.dumps({"task_id": "b"})]
    redis.lists[_processing_key(me)] = [json.dumps({"task_id": "c"})]  # my own — never touch
    redis.strings[_heartbeat_key(live)] = "1"  # live peer has a heartbeat
    # dead worker: no heartbeat key

    requeued, dead_n = asyncio.run(_recover_orphaned_processing(
        redis, my_worker_id=me, max_attempts=3, task_ttl_seconds=100
    ))
    assert requeued == 1 and dead_n == 0
    # Only the dead worker's job was requeued; live peer + my own list untouched.
    assert redis.lists.get("comparison:queue") == [json.dumps({"task_id": "b", "attempts": 1})]
    assert _processing_key(dead) not in redis.lists       # cleared
    assert _processing_key(live) in redis.lists           # left alone
    assert _processing_key(me) in redis.lists             # left alone


def test_orphan_recovery_reclaims_legacy_shared_list():
    from backend.worker import _recover_orphaned_processing, LEGACY_PROCESSING_KEY

    redis = FakeRedis()
    redis.lists[LEGACY_PROCESSING_KEY] = [json.dumps({"task_id": "legacy"})]
    requeued, _ = asyncio.run(_recover_orphaned_processing(
        redis, my_worker_id="wME", max_attempts=3, task_ttl_seconds=100
    ))
    assert requeued == 1
    assert redis.lists.get("comparison:queue") == [json.dumps({"task_id": "legacy", "attempts": 1})]
    assert LEGACY_PROCESSING_KEY not in redis.lists
