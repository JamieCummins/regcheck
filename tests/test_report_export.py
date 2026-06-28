from __future__ import annotations

import asyncio
import json
import os
import tempfile

from backend.db import models
from backend.services import reports as reports_service
from backend.services import report_artifacts as ra
from backend.services.evidence import build_text_evidence_source


class FakeRedis:
    def __init__(self):
        self.h: dict[str, dict] = {}
        self.kv: dict[str, object] = {}

    async def ping(self):
        return True

    async def hgetall(self, k):
        return self.h.get(k, {})

    async def hset(self, k, mapping=None):
        self.h.setdefault(k, {}).update(mapping or {})

    async def exists(self, k):
        return 1 if (k in self.h or k in self.kv) else 0

    async def get(self, k):
        return self.kv.get(k)

    async def set(self, k, v, ex=None):
        self.kv[k] = v

    async def delete(self, *ks):
        for k in ks:
            self.h.pop(k, None)
            self.kv.pop(k, None)

    async def srem(self, k, *m):
        pass


def _build_app(tmp):
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{tmp}/exp.db"
    os.environ["SESSION_SECRET"] = "exp-secret"
    for k in ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "ORCID_CLIENT_ID", "ORCID_CLIENT_SECRET"):
        os.environ.pop(k, None)
    import backend.core.config as cfg
    cfg.get_settings.cache_clear()
    from backend import create_app
    app = create_app()
    app.state.redis = FakeRedis()
    return app


def test_export_html_returns_self_contained_report():
    from starlette.testclient import TestClient

    tmp = tempfile.mkdtemp()
    app = _build_app(tmp)
    with TestClient(app) as client:
        async def seed():
            Session = app.state.db_sessionmaker
            async with Session() as s:
                owner = models.User(email="o@uni.edu")
                s.add(owner)
                await s.flush()
                await reports_service.create_report_row(
                    s, task_id="pp", owner_id=owner.id, visibility="public",
                    title="PP", comparison_type="general_preregistration",
                )
                await s.commit()
            # Persist one evidence source (artifacts + manifest) the way the worker does.
            payload = build_text_evidence_source(
                source_id="prereg", label="Preregistration",
                text="The planned sample size is 120 participants.",
                chunk_prefix="PREREG", max_chunk_tokens=100,
            )
            source_entry = await ra.store_source_artifacts(
                app.state.redis, task_id="pp", source=payload["source"],
                raw_bytes=None, raw_content_type=None,
                render_data=payload["render_data"], ttl_seconds=3600,
            )
            manifest = {
                "version": 1, "task_id": "pp", "comparison_type": "general_preregistration",
                "sources": {"prereg": source_entry}, "chunks": payload["chunks"],
            }
            await ra.store_manifest(app.state.redis, task_id="pp", manifest=manifest, ttl_seconds=3600)

        asyncio.get_event_loop().run_until_complete(seed())

        items = [{
            "dimension": "Sample size", "deviation_judgement": "yes",
            "registration_content_quotes": "[PREREG_0001] The planned sample size is 120 participants.",
            "paper_content_quotes": "", "registration_content_summary": "",
            "paper_content_summary": "", "deviation_information": "Sample size changed.",
        }]
        app.state.redis.h["pp"] = {
            "state": "SUCCESS", "result_json": json.dumps({"items": items}),
            "visibility": "public", "report_name": "My Export",
        }

        r = client.get("/report/pp/export.html")
        assert r.status_code == 200, r.text[:400]
        assert "text/html" in r.headers["content-type"]
        assert "attachment" in r.headers.get("content-disposition", "")
        body = r.text
        assert "__REGCHECK_BUNDLE__" in body          # bundle inlined
        assert "Sample size" in body                  # the comparison item is present
        assert "My Export" in body                    # title taken from report_name
        assert "/report/pp/sources" not in body       # self-contained: no server-backed URLs

        # 404 when the report has no stored evidence manifest.
        app.state.redis.h["nomanifest"] = {"state": "SUCCESS", "result_json": json.dumps({"items": []}), "visibility": "public"}
        assert client.get("/report/nomanifest/export.html").status_code == 404
