from __future__ import annotations

import pytest

from backend.db import models
from backend.db.session import create_engine_from_url, create_sessionmaker, init_models
from backend.services import reports as reports_service


class FakeRedis:
    def __init__(self):
        self.hashes: dict[str, dict] = {}
        self.values: dict[str, str] = {}
        self.deleted: list[str] = []
        self.sremoved: list[str] = []

    async def hset(self, key, mapping=None):
        self.hashes.setdefault(key, {}).update(mapping or {})

    async def get(self, key):
        return self.values.get(key)

    async def delete(self, *keys):
        for k in keys:
            self.deleted.append(k)
            self.values.pop(k, None)
            self.hashes.pop(k, None)

    async def srem(self, key, *members):
        self.sremoved.extend(members)


def test_generate_default_title():
    assert reports_service.generate_default_title(paper_filename="smith_et_al_2024.pdf") == "smith et al 2024"
    assert reports_service.generate_default_title(paper_filename=None, registration_id="NCT01234567") == "NCT01234567"
    auto = reports_service.generate_default_title(comparison_type="general_preregistration")
    assert auto.startswith("Comparison ")


def test_normalize_visibility():
    assert reports_service.normalize_visibility("public") == "public"
    assert reports_service.normalize_visibility("PRIVATE") == "private"
    assert reports_service.normalize_visibility("nonsense") == "private"
    assert reports_service.normalize_visibility(None) == "private"


@pytest.mark.asyncio
async def test_create_list_and_public_filter(tmp_path):
    engine = create_engine_from_url(f"sqlite+aiosqlite:///{tmp_path / 'r.db'}")
    try:
        await init_models(engine)
        Session = create_sessionmaker(engine)
        async with Session() as s:
            user = models.User(email="o@uni.edu", display_name="Owner", handle="owner")
            s.add(user)
            await s.flush()
            uid = user.id
            await reports_service.create_report_row(
                s, task_id="t-pub", owner_id=uid, visibility="public", title="Public one",
                comparison_type="general_preregistration",
            )
            await reports_service.create_report_row(
                s, task_id="t-priv", owner_id=uid, visibility="private", title="Private one",
                comparison_type="clinical_trials",
            )
            await s.commit()

        async with Session() as s:
            mine = await reports_service.list_reports_for_owner(s, uid)
            assert {r.task_id for r in mine} == {"t-pub", "t-priv"}
            public = await reports_service.list_public_reports_for_owner(s, uid)
            assert [r.task_id for r in public] == ["t-pub"]
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_rename_and_visibility_mirror_to_redis(tmp_path):
    engine = create_engine_from_url(f"sqlite+aiosqlite:///{tmp_path / 'r2.db'}")
    redis = FakeRedis()
    try:
        await init_models(engine)
        Session = create_sessionmaker(engine)
        async with Session() as s:
            user = models.User(email="o@uni.edu")
            s.add(user)
            await s.flush()
            await reports_service.create_report_row(
                s, task_id="t1", owner_id=user.id, visibility="private", title="Old",
                comparison_type=None,
            )
            await s.commit()

        async with Session() as s:
            report = await reports_service.get_report_row(s, "t1")
            new_title = await reports_service.rename_report(redis, s, report, "  New Name  ")
            assert new_title == "New Name"
            assert report.title == "New Name"
            assert redis.hashes["t1"]["title"] == "New Name"

            vis = await reports_service.set_report_visibility(redis, s, report, "public")
            assert vis == "public" and report.visibility == "public"
            assert redis.hashes["t1"]["visibility"] == "public"
            # Invalid visibility falls back to private.
            assert await reports_service.set_report_visibility(redis, s, report, "weird") == "private"
            await s.commit()
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_delete_report_everywhere(tmp_path):
    engine = create_engine_from_url(f"sqlite+aiosqlite:///{tmp_path / 'r3.db'}")
    redis = FakeRedis()
    try:
        await init_models(engine)
        Session = create_sessionmaker(engine)
        async with Session() as s:
            user = models.User(email="o@uni.edu")
            s.add(user)
            await s.flush()
            await reports_service.create_report_row(
                s, task_id="gone", owner_id=user.id, visibility="public", title="x",
                comparison_type=None,
            )
            await s.commit()

        async with Session() as s:
            report = await reports_service.get_report_row(s, "gone")
            await reports_service.delete_report_everywhere(redis, s, "gone", report)
            await s.commit()

        async with Session() as s:
            assert await reports_service.get_report_row(s, "gone") is None
        # Redis content + survey were targeted for deletion.
        assert "gone" in redis.deleted
        assert "survey:gone" in redis.deleted
        assert "report:gone:manifest" in redis.deleted
    finally:
        await engine.dispose()
