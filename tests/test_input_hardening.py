"""Phase-2 input/config hardening regression tests.

Covers: OSF download streaming + size cap, aggregate multi-file upload caps,
dimension payload bounds, TLS-verification defaults (Redis + Postgres), and the
production DATABASE_URL boot guard."""
from __future__ import annotations

import asyncio
import io
import os
import ssl
from pathlib import Path

import pytest
from fastapi import HTTPException, UploadFile

from backend.core import config as config_mod
from backend.core import redis as redis_mod
from backend.db import session as db_session
from backend.routes import comparisons as comp
from backend.services import osf


def _run(coro):
    # Private loop; don't touch the global current loop (legacy get_event_loop tests).
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _upload(name: str, content: str) -> UploadFile:
    data = content.encode("utf-8")
    return UploadFile(file=io.BytesIO(data), filename=name, size=len(data))


# ---------------------------------------------------------------- OSF download


class _FakeResp:
    def __init__(self, chunks: list[bytes], headers: dict[str, str] | None = None):
        self._chunks = chunks
        self.headers = headers or {}
        self.closed = False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield from self._chunks

    def close(self):
        self.closed = True


_FILE_DATA = {"attributes": {"name": "prereg.pdf"}, "links": {"download": "https://x/d"}}


def test_osf_download_rejects_declared_oversize(monkeypatch, tmp_path):
    resp = _FakeResp([b"x"], headers={"content-length": "999", "content-type": "application/pdf"})
    monkeypatch.setattr(osf, "_MAX_DOWNLOAD_BYTES", 10)
    monkeypatch.setattr(osf, "_get_with_retry", lambda *a, **k: resp)
    with pytest.raises(ValueError, match="larger than"):
        osf._download_file(_FILE_DATA, tmp_path, "abc12")
    assert resp.closed
    assert list(tmp_path.iterdir()) == []  # nothing written


def test_osf_download_enforces_cap_while_streaming(monkeypatch, tmp_path):
    # No Content-Length header: the cap must still hold on the streamed body.
    resp = _FakeResp([b"x" * 8, b"x" * 8], headers={"content-type": "application/pdf"})
    monkeypatch.setattr(osf, "_MAX_DOWNLOAD_BYTES", 10)
    monkeypatch.setattr(osf, "_get_with_retry", lambda *a, **k: resp)
    with pytest.raises(ValueError, match="larger than"):
        osf._download_file(_FILE_DATA, tmp_path, "abc12")
    assert resp.closed
    assert list(tmp_path.iterdir()) == []  # partial file removed


def test_osf_download_streams_within_cap(monkeypatch, tmp_path):
    resp = _FakeResp([b"%PDF-", b"body"], headers={"content-type": "application/pdf"})
    monkeypatch.setattr(osf, "_MAX_DOWNLOAD_BYTES", 100)
    monkeypatch.setattr(osf, "_get_with_retry", lambda *a, **k: resp)
    path, ext = osf._download_file(_FILE_DATA, tmp_path, "abc12")
    assert ext == ".pdf"
    with open(path, "rb") as fh:
        assert fh.read() == b"%PDF-body"
    assert resp.closed


# ------------------------------------------------------- aggregate upload caps


def test_coalesce_rejects_too_many_files(monkeypatch, tmp_path):
    monkeypatch.setattr(comp, "MAX_UPLOAD_FILES", 2)
    files = [_upload(f"f{i}.txt", "x") for i in range(3)]
    with pytest.raises(HTTPException) as ei:
        _run(comp._coalesce_uploads(files, upload_dir=tmp_path, kind="paper"))
    assert ei.value.status_code == 413
    assert "at most 2" in ei.value.detail


def test_coalesce_rejects_combined_bytes_over_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(comp, "MAX_COMBINED_UPLOAD_BYTES", 4)
    files = [_upload("a.txt", "abc"), _upload("b.txt", "def")]  # 6 bytes combined
    with pytest.raises(HTTPException) as ei:
        _run(comp._coalesce_uploads(files, upload_dir=tmp_path, kind="paper"))
    assert ei.value.status_code == 413
    # temp files are cleaned up on failure
    assert [p for p in tmp_path.iterdir() if p.is_file()] == []


def test_coalesce_within_caps_still_combines(tmp_path):
    files = [_upload("a.txt", "First."), _upload("b.txt", "Second.")]
    out = _run(comp._coalesce_uploads(files, upload_dir=tmp_path, kind="paper"))
    assert out.filename == "paper-combined.txt"


# ------------------------------------------------------------- dimension caps


def _dims_payload(n: int, name: str = "Sample size", definition: str = "def") -> str:
    import json

    return json.dumps([{"dimension": f"{name} {i}", "definition": definition} for i in range(n)])


def test_dimensions_count_cap(monkeypatch):
    monkeypatch.setattr(comp, "MAX_DIMENSIONS", 3)
    assert len(comp._parse_dimensions(_dims_payload(3))) == 3
    with pytest.raises(HTTPException) as ei:
        comp._parse_dimensions(_dims_payload(4))
    assert ei.value.status_code == 400


def test_dimension_name_and_definition_length_caps():
    import json

    long_name = json.dumps([{"dimension": "x" * (comp.MAX_DIMENSION_NAME_CHARS + 1), "definition": ""}])
    with pytest.raises(HTTPException, match="names are limited"):
        comp._parse_dimensions(long_name)

    long_def = json.dumps(
        [{"dimension": "ok", "definition": "x" * (comp.MAX_DIMENSION_DEFINITION_CHARS + 1)}]
    )
    with pytest.raises(HTTPException, match="definitions are limited"):
        comp._parse_dimensions(long_def)


def test_dimensions_payload_size_cap(monkeypatch):
    monkeypatch.setattr(comp, "_MAX_DIMENSIONS_PAYLOAD_CHARS", 50)
    with pytest.raises(HTTPException, match="too large"):
        comp._parse_dimensions(_dims_payload(10))


# ------------------------------------------------------------- TLS defaults


def test_rediss_verifies_certificates_by_default(monkeypatch):
    captured = {}

    def fake_from_url(url, **kw):
        captured["kw"] = kw
        return object()

    monkeypatch.setattr(redis_mod.aioredis, "from_url", fake_from_url)
    monkeypatch.delenv("REDIS_TLS_INSECURE", raising=False)
    redis_mod.create_redis_client("rediss://host:6379/0")
    assert "ssl_cert_reqs" not in captured["kw"]  # library default = verified TLS


def test_rediss_insecure_requires_explicit_opt_out(monkeypatch):
    captured = {}

    def fake_from_url(url, **kw):
        captured["kw"] = kw
        return object()

    monkeypatch.setattr(redis_mod.aioredis, "from_url", fake_from_url)
    monkeypatch.setenv("REDIS_TLS_INSECURE", "1")
    redis_mod.create_redis_client("rediss://host:6379/0")
    assert captured["kw"]["ssl_cert_reqs"] is None


def test_remote_postgres_verifies_by_default(monkeypatch):
    url = "postgresql+asyncpg://u:p@db.example.com:5432/app"
    monkeypatch.delenv("DATABASE_SSL", raising=False)
    ctx = db_session._connect_args(url)["ssl"]
    assert ctx.check_hostname is True
    assert ctx.verify_mode == ssl.CERT_REQUIRED


def test_postgres_relaxed_opt_out(monkeypatch):
    url = "postgresql+asyncpg://u:p@db.example.com:5432/app"
    monkeypatch.setenv("DATABASE_SSL", "relaxed")
    ctx = db_session._connect_args(url)["ssl"]
    assert ctx.check_hostname is False
    assert ctx.verify_mode == ssl.CERT_NONE


def test_postgres_disable_and_local(monkeypatch):
    monkeypatch.setenv("DATABASE_SSL", "disable")
    assert db_session._connect_args("postgresql+asyncpg://u:p@db.example.com/app") == {}
    monkeypatch.delenv("DATABASE_SSL", raising=False)
    assert db_session._connect_args("postgresql+asyncpg://u:p@localhost/app") == {}


# ------------------------------------------------ production DATABASE_URL guard


def test_production_refuses_sqlite_fallback(monkeypatch):
    monkeypatch.setenv("DYNO", "web.1")
    monkeypatch.setenv("REDIS_URL", "rediss://example.com:6379/0")
    monkeypatch.setenv("SESSION_SECRET", "s" * 32)
    # Blank both so resolve_database_url() falls back to SQLite even if .env sets them.
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("HEROKU_POSTGRESQL_URL", "")
    config_mod.get_settings.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="DATABASE_URL"):
            config_mod.get_settings()
    finally:
        config_mod.get_settings.cache_clear()


def test_production_allows_explicit_sqlite_database_url(monkeypatch, tmp_path):
    # Only the SILENT fallback is refused; an explicitly configured sqlite URL
    # is a deliberate deployment choice and must boot.
    monkeypatch.setenv("DYNO", "web.1")
    monkeypatch.setenv("REDIS_URL", "rediss://example.com:6379/0")
    monkeypatch.setenv("SESSION_SECRET", "s" * 32)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/explicit.db")
    config_mod.get_settings.cache_clear()
    try:
        settings = config_mod.get_settings()
        assert settings.database_url.startswith("sqlite+aiosqlite")
    finally:
        config_mod.get_settings.cache_clear()


def test_db_metadata_importable_without_database(tmp_path):
    # Alembic's env.py imports backend.db.base on the release dyno; that import
    # must not drag in the app (whose settings guard would raise before
    # migrations even start). Regression for the v1.0.0 staging deploy failure.
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k not in {"DATABASE_URL", "HEROKU_POSTGRESQL_URL"}}
    env["DYNO"] = "web.1"
    proc = subprocess.run(
        [sys.executable, "-c", "import backend.db.base, backend.db.models, backend.db.session"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert proc.returncode == 0, proc.stderr
