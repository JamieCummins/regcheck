from __future__ import annotations

import base64
import json
import os
import tempfile

import itsdangerous
from starlette.testclient import TestClient


class FakeRedis:
    def __init__(self):
        self.h: dict[str, dict] = {}
        self.queue: list[str] = []
        self.keys: set[str] = set()

    async def ping(self):
        return True

    async def hgetall(self, k):
        return self.h.get(k, {})

    async def hget(self, k, field):
        return (self.h.get(k) or {}).get(field)

    async def hset(self, k, mapping=None):
        self.h.setdefault(k, {}).update(mapping or {})

    async def exists(self, k):
        return 1 if (k in self.h or k in self.keys) else 0

    async def rpush(self, k, value):
        self.queue.append(value)

    async def delete(self, *ks):
        for k in ks:
            self.h.pop(k, None)
            self.keys.discard(k)

    async def expire(self, *a):
        pass

    async def persist(self, *a):
        pass

    async def aclose(self):
        pass


def _build_app():
    tmp = tempfile.mkdtemp()
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{tmp}/r.db"
    os.environ["SESSION_SECRET"] = "regen-secret"
    for k in ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "ORCID_CLIENT_ID", "ORCID_CLIENT_SECRET"):
        os.environ.pop(k, None)
    import backend.core.config as cfg

    cfg.get_settings.cache_clear()
    from backend import create_app

    app = create_app()
    app.state.redis = FakeRedis()
    return app


def _session_cookie(data: dict) -> str:
    signer = itsdangerous.TimestampSigner("regen-secret")
    return signer.sign(base64.b64encode(json.dumps(data).encode())).decode()


def _job(tid: str) -> dict:
    return {
        "comparison_type": "general_preregistration",
        "task_id": tid,
        "client": "openai",
        "upload_keys": {"paper": f"upload:{tid}:paper", "prereg": None, "csv": None},
        "s3_keys": {},
    }


def test_regenerate_requeues_for_session_owner():
    app = _build_app()
    with TestClient(app) as client:
        tid = "task-regen"
        app.state.redis.h[tid] = {
            "state": "SUCCESS",
            "regen_job": json.dumps(_job(tid)),
            "evidence_status": "missing",
        }
        app.state.redis.keys.add(f"upload:{tid}:paper")  # original upload still present
        client.cookies.set("session", _session_cookie({"owned_reports": [tid]}))

        resp = client.post(f"/reports/{tid}/regenerate")
        assert resp.status_code == 200, resp.text
        assert resp.json()["ok"] is True
        # the job was re-queued verbatim and the task was reset to PENDING
        assert app.state.redis.queue and json.loads(app.state.redis.queue[-1])["task_id"] == tid
        assert app.state.redis.h[tid]["state"] == "PENDING"
        assert app.state.redis.h[tid]["evidence_status"] == "pending"


def test_regenerate_forbidden_for_non_manager():
    app = _build_app()
    with TestClient(app) as client:
        tid = "task-x"
        app.state.redis.h[tid] = {"state": "SUCCESS", "regen_job": json.dumps(_job(tid))}
        resp = client.post(f"/reports/{tid}/regenerate")  # no session ownership
        assert resp.status_code == 403
        assert app.state.redis.queue == []


def test_regenerate_conflict_when_no_stored_job():
    app = _build_app()
    with TestClient(app) as client:
        tid = "task-y"
        app.state.redis.h[tid] = {"state": "SUCCESS"}  # no regen_job
        client.cookies.set("session", _session_cookie({"owned_reports": [tid]}))
        resp = client.post(f"/reports/{tid}/regenerate")
        assert resp.status_code == 409


def test_regenerate_conflict_when_uploads_expired():
    app = _build_app()
    with TestClient(app) as client:
        tid = "task-z"
        app.state.redis.h[tid] = {"state": "SUCCESS", "regen_job": json.dumps(_job(tid))}
        # the paper upload key is NOT present → treated as expired
        client.cookies.set("session", _session_cookie({"owned_reports": [tid]}))
        resp = client.post(f"/reports/{tid}/regenerate")
        assert resp.status_code == 409
        assert app.state.redis.queue == []
