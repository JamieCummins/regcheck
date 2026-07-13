from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routes import comparisons


def _client(tmp_path: Path) -> TestClient:
    app = FastAPI()
    app.state.settings = SimpleNamespace(upload_dir=str(tmp_path))
    app.include_router(comparisons.router)
    return TestClient(app)


def _long_text() -> str:
    return ("This preregistration specifies hypotheses, sample size, and analysis plan. " * 12).strip()


def test_preflight_upload_with_text_is_not_thin(tmp_path):
    client = _client(tmp_path)
    resp = client.post(
        "/preflight/registration",
        data={"prereg_source": "upload"},
        files={"preregistration": ("prereg.txt", _long_text().encode("utf-8"), "text/plain")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["thin"] is False
    assert body["chars"] >= body["threshold"]
    assert body["source"] == "upload"


def test_preflight_upload_near_empty_is_thin(tmp_path):
    client = _client(tmp_path)
    resp = client.post(
        "/preflight/registration",
        data={"prereg_source": "upload"},
        files={"preregistration": ("stub.txt", b"see attached", "text/plain")},
    )
    body = resp.json()
    assert body["ok"] is True
    assert body["thin"] is True
    assert body["chars"] < body["threshold"]


def test_preflight_no_input_is_not_actionable(tmp_path):
    client = _client(tmp_path)
    resp = client.post("/preflight/registration", data={"prereg_source": "upload"})
    body = resp.json()
    assert body["ok"] is False
    assert body["reason"] == "no_input"


def test_preflight_unsupported_upload_type(tmp_path):
    client = _client(tmp_path)
    resp = client.post(
        "/preflight/registration",
        data={"prereg_source": "upload"},
        files={"preregistration": ("notes.rtf", b"x" * 50, "application/rtf")},
    )
    body = resp.json()
    assert body["ok"] is False
    assert body["reason"] == "unsupported_type"


def test_preflight_osf_thin_registry_stub(tmp_path, monkeypatch):
    # An OSF *registry* link whose flattened metadata is near-empty (the real
    # content is an attached file) must be flagged thin.
    stub = tmp_path / "osf_stub.txt"
    stub.write_text("Title only", encoding="utf-8")

    def fake_fetch(url, dest_dir):
        return str(stub), ".txt"

    monkeypatch.setattr(comparisons, "fetch_osf_preregistration", fake_fetch)
    client = _client(tmp_path)
    resp = client.post(
        "/preflight/registration",
        data={"prereg_source": "osf", "osf_url": "https://osf.io/abc12/"},
    )
    body = resp.json()
    assert body["ok"] is True
    assert body["thin"] is True
    assert body["source"] == "osf"


def test_preflight_osf_resolution_failure_is_soft(tmp_path, monkeypatch):
    def boom(url, dest_dir):
        raise RuntimeError("404 from OSF")

    monkeypatch.setattr(comparisons, "fetch_osf_preregistration", boom)
    client = _client(tmp_path)
    resp = client.post(
        "/preflight/registration",
        data={"prereg_source": "osf", "osf_url": "https://osf.io/missing/"},
    )
    body = resp.json()
    assert body["ok"] is False
    assert body["reason"] == "osf_unresolved"
