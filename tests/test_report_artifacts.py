from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routes import report
from backend.services.report_artifacts import (
    load_artifact_bytes,
    load_manifest,
    public_manifest,
    store_manifest,
    store_source_artifacts,
    verify_manifest_artifacts,
)


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.expiries = {}

    async def set(self, key, value, ex=None):
        self.values[key] = value
        self.expiries[key] = ex

    async def get(self, key):
        return self.values.get(key)

    async def exists(self, key):
        return 1 if key in self.values else 0


async def _store_pdf_report(fake_redis, tmp_path):
    import fitz

    pdf_path = tmp_path / "paper.pdf"
    doc = fitz.open()
    page = doc.new_page(width=220, height=160)
    page.insert_text((24, 60), "Evidence text")
    doc.save(pdf_path)
    doc.close()

    source = await store_source_artifacts(
        fake_redis,
        task_id="task-1",
        source={
            "id": "paper",
            "label": "Paper",
            "kind": "pdf",
            "render_mode": "pdf",
            "page_count": 1,
            "raw_filename": "paper.pdf",
        },
        raw_bytes=pdf_path.read_bytes(),
        raw_content_type="application/pdf",
        render_data={"kind": "pdf", "text": "Evidence text"},
        ttl_seconds=3600,
    )
    manifest = {
        "version": 1,
        "task_id": "task-1",
        "sources": {"paper": source},
        "chunks": {},
    }
    await store_manifest(fake_redis, task_id="task-1", manifest=manifest, ttl_seconds=3600)
    return manifest


def test_report_artifact_storage_redis_round_trip(event_loop, monkeypatch):
    monkeypatch.delenv("S3_BUCKET", raising=False)
    fake_redis = FakeRedis()

    source = event_loop.run_until_complete(
        store_source_artifacts(
            fake_redis,
            task_id="task-1",
            source={"id": "registration", "label": "Registration", "kind": "text"},
            raw_bytes=b"raw",
            raw_content_type="text/plain",
            render_data={"kind": "text", "text": "raw"},
            ttl_seconds=120,
        )
    )
    manifest = {"sources": {"registration": source}, "chunks": {}}
    event_loop.run_until_complete(store_manifest(fake_redis, task_id="task-1", manifest=manifest, ttl_seconds=120))

    loaded = event_loop.run_until_complete(load_manifest(fake_redis, "task-1"))
    raw = event_loop.run_until_complete(load_artifact_bytes(fake_redis, source["_artifacts"]["raw"]))
    public = public_manifest(loaded, "task-1")

    assert source["_artifacts"]["render"]["compressed_bytes"] > 0
    assert source["_artifacts"]["raw"]["compressed_bytes"] > 0
    assert raw == b"raw"
    assert "_artifacts" not in public["sources"]["registration"]
    assert public["sources"]["registration"]["raw_url"] == "/report/task-1/sources/registration/raw"


def test_report_artifact_storage_uses_redis_even_when_s3_is_configured(event_loop, monkeypatch):
    monkeypatch.setattr("backend.services.report_artifacts.get_s3_config", lambda: object())
    fake_redis = FakeRedis()

    source = event_loop.run_until_complete(
        store_source_artifacts(
            fake_redis,
            task_id="task-1",
            source={"id": "registration", "label": "Registration", "kind": "text"},
            raw_bytes=b"raw",
            raw_content_type="text/plain",
            render_data={"kind": "text", "text": "raw"},
            ttl_seconds=120,
        )
    )

    assert source["_artifacts"]["render"]["storage"] == "redis"
    assert source["_artifacts"]["raw"]["storage"] == "redis"
    raw = event_loop.run_until_complete(load_artifact_bytes(fake_redis, source["_artifacts"]["raw"]))
    render = event_loop.run_until_complete(load_artifact_bytes(fake_redis, source["_artifacts"]["render"]))
    assert raw == b"raw"
    assert b'"text": "raw"' in render


def test_report_artifact_verification_fails_when_redis_artifact_is_missing(event_loop):
    fake_redis = FakeRedis()

    source = event_loop.run_until_complete(
        store_source_artifacts(
            fake_redis,
            task_id="task-1",
            source={"id": "registration", "label": "Registration", "kind": "text"},
            raw_bytes=b"raw",
            raw_content_type="text/plain",
            render_data={"kind": "text", "text": "raw"},
            ttl_seconds=120,
        )
    )
    manifest = {"sources": {"registration": source}, "chunks": {}}
    event_loop.run_until_complete(store_manifest(fake_redis, task_id="task-1", manifest=manifest, ttl_seconds=120))
    del fake_redis.values[source["_artifacts"]["raw"]["key"]]

    with pytest.raises(RuntimeError, match="registration.raw Redis artifact key"):
        event_loop.run_until_complete(
            verify_manifest_artifacts(fake_redis, task_id="task-1", manifest=manifest)
        )


def test_report_routes_return_manifest_raw_render_and_page(event_loop, tmp_path, monkeypatch):
    monkeypatch.delenv("S3_BUCKET", raising=False)
    fake_redis = FakeRedis()
    event_loop.run_until_complete(_store_pdf_report(fake_redis, tmp_path))

    app = FastAPI()
    app.state.redis = fake_redis
    app.include_router(report.router)
    client = TestClient(app)

    manifest_response = client.get("/report/task-1/manifest")
    assert manifest_response.status_code == 200
    assert manifest_response.headers["cache-control"].startswith("private")
    assert "_artifacts" not in manifest_response.json()["sources"]["paper"]

    render_response = client.get("/report/task-1/sources/paper/render-data")
    assert render_response.status_code == 200
    assert render_response.json()["text"] == "Evidence text"

    raw_response = client.get("/report/task-1/sources/paper/raw")
    assert raw_response.status_code == 200
    assert raw_response.headers["content-type"].startswith("application/pdf")

    page_response = client.get("/report/task-1/sources/paper/pages/1.png")
    assert page_response.status_code == 200
    assert page_response.headers["content-type"].startswith("image/png")
    assert page_response.content.startswith(b"\x89PNG")


def test_report_routes_missing_manifest_returns_404():
    app = FastAPI()
    app.state.redis = FakeRedis()
    app.include_router(report.router)
    client = TestClient(app)

    response = client.get("/report/missing/manifest")

    assert response.status_code == 404
