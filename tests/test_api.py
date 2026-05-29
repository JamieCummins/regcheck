from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.core.config import Settings
from backend.routes import api
from backend.services.dimensions import default_dimensions_for


class FakeRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict] = {}
        self.lists: dict[str, list[str]] = {}
        self.values: dict[str, str] = {}
        self.expirations: dict[str, int] = {}

    async def llen(self, key: str) -> int:
        return len(self.lists.get(key, []))

    async def hset(self, key: str, mapping: dict | None = None, **kwargs) -> None:
        self.hashes.setdefault(key, {})
        if mapping:
            self.hashes[key].update(mapping)
        if kwargs:
            self.hashes[key].update(kwargs)

    async def expire(self, key: str, seconds: int) -> None:
        self.expirations[key] = seconds

    async def rpush(self, key: str, *values: str) -> None:
        self.lists.setdefault(key, []).extend(values)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.values[key] = value
        if ex is not None:
            self.expirations[key] = ex

    async def hgetall(self, key: str) -> dict:
        return self.hashes.get(key, {})


def _make_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, api_token: str | None = "secret"):
    monkeypatch.delenv("S3_BUCKET", raising=False)
    redis = FakeRedis()
    app = FastAPI()
    app.state.redis = redis
    app.state.settings = Settings(
        redis_url="redis://test/0",
        session_secret="test-session",
        api_token=api_token,
        task_ttl_seconds=86400,
        max_queue_length=200,
        static_dir=str(tmp_path / "static"),
        templates_dir=str(tmp_path / "templates"),
        upload_dir=str(tmp_path / "uploads"),
    )
    app.include_router(api.router)
    return TestClient(app), redis


def _auth_headers(token: str = "secret") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _paper_file():
    return ("paper.pdf", b"%PDF-1.4\npaper", "application/pdf")


def _registration_file():
    return ("registration.pdf", b"%PDF-1.4\nregistration", "application/pdf")


def _queued_job(redis: FakeRedis) -> dict:
    return json.loads(redis.lists["comparison:queue"][0])


def test_create_comparison_requires_configured_api_token(tmp_path, monkeypatch):
    client, _redis = _make_client(tmp_path, monkeypatch, api_token=None)

    response = client.post(
        "/api/v1/comparisons",
        data={"registration_id": "NCT01234567"},
        files={"paper": _paper_file()},
        headers=_auth_headers(),
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "API_AUTH_NOT_CONFIGURED"


def test_create_comparison_requires_auth(tmp_path, monkeypatch):
    client, _redis = _make_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/v1/comparisons",
        data={"registration_id": "NCT01234567"},
        files={"paper": _paper_file()},
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "MISSING_API_AUTH"


def test_create_comparison_rejects_invalid_auth(tmp_path, monkeypatch):
    client, _redis = _make_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/v1/comparisons",
        data={"registration_id": "NCT01234567"},
        files={"paper": _paper_file()},
        headers=_auth_headers("wrong"),
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "INVALID_API_AUTH"


def test_create_comparison_accepts_x_api_key(tmp_path, monkeypatch):
    client, _redis = _make_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/v1/comparisons",
        data={"registration_id": "NCT01234567"},
        files={"paper": _paper_file()},
        headers={"X-API-Key": "secret"},
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["state"] == "queued"
    assert payload["status_url"] == f"/api/v1/comparisons/{payload['task_id']}"


def test_create_comparison_requires_paper(tmp_path, monkeypatch):
    client, _redis = _make_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/v1/comparisons",
        data={"registration_id": "NCT01234567"},
        headers=_auth_headers(),
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "MISSING_PAPER"


def test_create_comparison_requires_one_registration_input(tmp_path, monkeypatch):
    client, _redis = _make_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/v1/comparisons",
        files={"paper": _paper_file()},
        headers=_auth_headers(),
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "MISSING_REGISTRATION_INPUT"


def test_create_comparison_rejects_ambiguous_registration_input(tmp_path, monkeypatch):
    client, _redis = _make_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/v1/comparisons",
        data={"registration_id": "NCT01234567"},
        files=[
            ("paper", _paper_file()),
            ("registration_file", _registration_file()),
        ],
        headers=_auth_headers(),
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "AMBIGUOUS_REGISTRATION_INPUT"


def test_create_comparison_rejects_invalid_registration_id(tmp_path, monkeypatch):
    client, _redis = _make_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/v1/comparisons",
        data={"registration_id": "not-a-trial-id"},
        files={"paper": _paper_file()},
        headers=_auth_headers(),
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REGISTRATION_ID"


def test_create_clinical_comparison_queues_default_dimensions(tmp_path, monkeypatch):
    client, redis = _make_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/v1/comparisons",
        data={"registration_id": "https://clinicaltrials.gov/study/NCT01234567"},
        files={"paper": _paper_file()},
        headers=_auth_headers(),
    )

    assert response.status_code == 202
    task_id = response.json()["task_id"]
    expected_dimensions = default_dimensions_for("clinical_trials")
    expected_names = [item["dimension"] for item in expected_dimensions]
    assert json.loads(redis.hashes[task_id]["dimensions"]) == expected_names
    assert redis.hashes[task_id]["total_dimensions"] == len(expected_dimensions)

    job = _queued_job(redis)
    assert job["comparison_type"] == "clinical_trials"
    assert job["registration_id"] == "NCT01234567"
    assert job["parser_choice"] == "grobid"
    assert job["client"] == "openai"
    assert job["reasoning_effort"] == "medium"
    assert job["append_previous_output"] is True
    assert job["selected_dimensions"] == expected_dimensions


def test_create_file_comparison_queues_general_defaults(tmp_path, monkeypatch):
    client, redis = _make_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/v1/comparisons",
        files=[
            ("paper", _paper_file()),
            ("registration_file", _registration_file()),
        ],
        headers=_auth_headers(),
    )

    assert response.status_code == 202
    task_id = response.json()["task_id"]
    expected_dimensions = default_dimensions_for("general_preregistration")
    expected_names = [item["dimension"] for item in expected_dimensions]
    assert json.loads(redis.hashes[task_id]["dimensions"]) == expected_names

    job = _queued_job(redis)
    assert job["comparison_type"] == "general_preregistration"
    assert job["selected_dimensions"] == expected_dimensions
    assert job["append_previous_output"] is True


def test_create_comparison_uses_supplied_dimensions(tmp_path, monkeypatch):
    client, redis = _make_client(tmp_path, monkeypatch)
    dimensions = [
        {"dimension": "Primary Outcome(s)", "definition": "Primary endpoint definition."},
        {"dimension": "Sample Size", "definition": "Total and per-arm sample size."},
    ]

    response = client.post(
        "/api/v1/comparisons",
        data={
            "registration_id": "NCT01234567",
            "dimensions": json.dumps(dimensions),
            "append_previous_output": "no",
        },
        files={"paper": _paper_file()},
        headers=_auth_headers(),
    )

    assert response.status_code == 202
    job = _queued_job(redis)
    assert job["selected_dimensions"] == dimensions
    assert job["append_previous_output"] is False


def test_create_comparison_rejects_invalid_dimensions(tmp_path, monkeypatch):
    client, _redis = _make_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/v1/comparisons",
        data={"registration_id": "NCT01234567", "dimensions": "not-json"},
        files={"paper": _paper_file()},
        headers=_auth_headers(),
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_DIMENSIONS"


@pytest.mark.parametrize(
    ("internal_state", "api_state"),
    [
        ("PENDING", "queued"),
        ("IN_PROGRESS", "in_progress"),
        ("SUCCESS", "success"),
        ("FAILURE", "failure"),
    ],
)
def test_get_comparison_normalizes_task_state(tmp_path, monkeypatch, internal_state, api_state):
    client, redis = _make_client(tmp_path, monkeypatch)
    redis.hashes["abc-123"] = {
        "state": internal_state,
        "status": "Processed 3/8: Primary Outcome(s)",
        "processed_dimensions": "3",
        "total_dimensions": "8",
        "result_json": json.dumps({"items": []}),
    }

    response = client.get("/api/v1/comparisons/abc-123", headers=_auth_headers())

    assert response.status_code == 200
    assert response.json() == {
        "task_id": "abc-123",
        "state": api_state,
        "status": "Processed 3/8: Primary Outcome(s)",
        "processed_dimensions": 3,
        "total_dimensions": 8,
        "result": {"items": []},
    }


def test_get_comparison_returns_404_for_unknown_task(tmp_path, monkeypatch):
    client, _redis = _make_client(tmp_path, monkeypatch)

    response = client.get("/api/v1/comparisons/missing-task", headers=_auth_headers())

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "TASK_NOT_FOUND"
