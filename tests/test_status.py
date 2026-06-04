from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routes import status
from backend.services.report_artifacts import manifest_key


class FakeRedis:
    def __init__(self):
        self.hashes = {}
        self.values = {}

    async def hgetall(self, key):
        return self.hashes.get(key, {})

    async def exists(self, key):
        return 1 if key in self.values else 0


def _client(fake_redis: FakeRedis) -> TestClient:
    app = FastAPI()
    app.state.redis = fake_redis
    app.include_router(status.router)
    return TestClient(app)


def test_task_status_marks_success_without_manifest_as_missing():
    redis = FakeRedis()
    redis.hashes["task-1"] = {
        "state": "SUCCESS",
        "status": "Report complete",
        "result_json": json.dumps({"items": []}),
    }

    response = _client(redis).get("/task_status/task-1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["evidence_available"] is False
    assert payload["evidence_status"] == "missing"


def test_task_status_overrides_stale_pending_evidence_for_success():
    redis = FakeRedis()
    redis.hashes["task-1"] = {
        "state": "SUCCESS",
        "status": "Report complete",
        "result_json": json.dumps({"items": []}),
        "evidence_status": "pending",
    }

    response = _client(redis).get("/task_status/task-1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["evidence_available"] is False
    assert payload["evidence_status"] == "missing"
    assert "Evidence artifacts were not found" in payload["evidence_error"]


def test_task_status_marks_existing_manifest_as_ready():
    redis = FakeRedis()
    redis.hashes["task-1"] = {
        "state": "SUCCESS",
        "status": "Report complete",
        "result_json": json.dumps({"items": []}),
    }
    redis.values[manifest_key("task-1")] = "{}"

    response = _client(redis).get("/task_status/task-1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["evidence_available"] is True
    assert payload["evidence_status"] == "ready"
