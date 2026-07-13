from __future__ import annotations

import json

import pytest

from backend.core.storage import S3Config
from backend.db import models
from backend.db.session import create_engine_from_url, create_sessionmaker, init_models
from backend.services import report_artifacts as ra
from backend.services import reports as reports_service


class FakeRedis:
    def __init__(self):
        self.h: dict[str, dict] = {}
        self.kv: dict[str, str] = {}
        self.ttls: dict[str, object] = {}
        self.persisted: set[str] = set()
        self.zsets: dict[str, dict] = {}

    async def hgetall(self, k):
        return dict(self.h.get(k, {}))

    async def hset(self, k, mapping=None):
        self.h.setdefault(k, {}).update(mapping or {})

    async def persist(self, k):
        self.persisted.add(k)
        self.ttls[k] = None
        return True

    async def get(self, k):
        return self.kv.get(k)

    async def set(self, k, v, ex=None):
        self.kv[k] = v
        self.ttls[k] = ex

    async def zadd(self, name, mapping):
        self.zsets.setdefault(name, {}).update(mapping)

    async def zrem(self, name, *members):
        z = self.zsets.get(name)
        if z:
            for m in members:
                z.pop(m, None)

    async def delete(self, *ks):
        for k in ks:
            self.h.pop(k, None)
            self.kv.pop(k, None)


@pytest.mark.asyncio
async def test_claim_anonymous_report_takes_ownership_and_persists(tmp_path):
    engine = create_engine_from_url(f"sqlite+aiosqlite:///{tmp_path / 'claim.db'}")
    await init_models(engine)
    Session = create_sessionmaker(engine)
    redis = FakeRedis()
    # An anonymous report: redis hash with no owner, on a 7-day TTL, public.
    redis.h["task-1"] = {
        "state": "SUCCESS",
        "title": "My report",
        "comparison_type": "general_preregistration",
        "owner_id": "",
        "retention": "604800",
        "visibility": "public",
    }
    redis.ttls["task-1"] = 604800

    async with Session() as db:
        db.add(models.User(id="user-1", email="a@b.c", display_name="A"))
        await db.commit()
        claimed = await reports_service.claim_anonymous_reports(
            redis, db, owner_id="user-1", task_ids=["task-1", "expired-or-missing"]
        )
        await db.commit()

        assert claimed == ["task-1"]
        # Redis hash flipped to owned + persistent.
        assert redis.h["task-1"]["owner_id"] == "user-1"
        assert redis.h["task-1"]["retention"] == "persist"
        assert redis.h["task-1"]["visibility"] == "private"  # default-private on claim
        assert "task-1" in redis.persisted
        # Durable ownership row created, carrying the title/type.
        row = await reports_service.get_report_row(db, "task-1")
        assert row is not None
        assert row.owner_id == "user-1"
        assert row.visibility == "private"
        assert row.title == "My report"

    await engine.dispose()


@pytest.mark.asyncio
async def test_claim_skips_report_already_owned(tmp_path):
    engine = create_engine_from_url(f"sqlite+aiosqlite:///{tmp_path / 'claim2.db'}")
    await init_models(engine)
    Session = create_sessionmaker(engine)
    redis = FakeRedis()
    redis.h["task-x"] = {"owner_id": "someone-else", "state": "SUCCESS"}

    async with Session() as db:
        db.add(models.User(id="user-2", email="b@c.d"))
        await db.commit()
        claimed = await reports_service.claim_anonymous_reports(
            redis, db, owner_id="user-2", task_ids=["task-x"]
        )
        assert claimed == []
        assert redis.h["task-x"]["owner_id"] == "someone-else"  # untouched

    await engine.dispose()


@pytest.mark.asyncio
async def test_migrate_evidence_temp_to_persist(monkeypatch):
    redis = FakeRedis()
    manifest = {
        "version": 1,
        "task_id": "t",
        "sources": {
            "paper": {
                "id": "paper",
                "_artifacts": {
                    "raw": {"storage": "s3", "bucket": "regcheck-temp-files", "key": "regcheck/report/t/source/paper/raw"},
                    "render": {"storage": "s3", "bucket": "regcheck-temp-files", "key": "regcheck/report/t/source/paper/render"},
                },
            }
        },
    }
    redis.kv[ra.manifest_key("t")] = json.dumps(manifest)
    redis.ttls[ra.manifest_key("t")] = 604800
    redis.zsets[ra.S3_CLEANUP_ZSET] = {
        "regcheck/report/t/source/paper/raw": 1.0,
        "regcheck/report/t/source/paper/render": 1.0,
    }

    copies: list[tuple[str, str, str]] = []
    monkeypatch.setattr(ra, "get_persist_s3_config", lambda: S3Config(bucket="regcheck-persist", region="r"))
    monkeypatch.setattr(ra, "get_s3_config", lambda: S3Config(bucket="regcheck-temp-files", region="r"))
    monkeypatch.setattr(ra, "s3_config_for_bucket", lambda b: S3Config(bucket=b, region="r"))
    monkeypatch.setattr(ra, "s3_copy_object", lambda src, dst, *, key: copies.append((src.bucket, dst.bucket, key)))

    moved = await ra.migrate_artifacts_to_persist(redis, "t")

    assert moved == 2
    assert {c[2] for c in copies} == {
        "regcheck/report/t/source/paper/raw",
        "regcheck/report/t/source/paper/render",
    }
    assert all(dst == "regcheck-persist" for _src, dst, _key in copies)
    # Manifest re-stored with persist-bucket refs and no expiry.
    updated = json.loads(redis.kv[ra.manifest_key("t")])
    arts = updated["sources"]["paper"]["_artifacts"]
    assert arts["raw"]["bucket"] == "regcheck-persist"
    assert arts["render"]["bucket"] == "regcheck-persist"
    assert redis.ttls[ra.manifest_key("t")] is None
    # Cleanup schedule cleared for the moved keys.
    assert not redis.zsets.get(ra.S3_CLEANUP_ZSET)
