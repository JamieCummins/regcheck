from __future__ import annotations

import contextlib
import logging
import os

from fastapi import FastAPI, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import text as sa_text
from starlette.middleware.sessions import SessionMiddleware

from .core.auth_context import CurrentUserMiddleware
from .core.config import get_settings
from .core.logging import configure_logging
from .core.security_headers import FrameBustMiddleware, SecurityHeadersMiddleware
from .core.oauth import build_oauth
from .core.redis import create_redis_client
from .db.session import create_engine_from_url, create_sessionmaker, init_models
from .routes import auth, comparisons, pages, report, status
from .routes import api as api_routes
from .routes import reports as reports_routes
from .routes import survey

# Single source of truth for the application release version: OpenAPI, /docs,
# and the CHANGELOG all reference this value.
APP_VERSION = "1.0.0"

# Single source of truth for static-asset cache-busting. Bump this on any
# CSS/JS change so browsers refetch; it is exposed to every template as the
# `static_version` global (templates reference it as `?v={{ static_version }}`).
STATIC_VERSION = "20260714-demo-refresh"


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging()
    logger = logging.getLogger(__name__)
    if not (os.environ.get("SESSION_SECRET") or "").strip():
        logger.warning("SESSION_SECRET not set; using an ephemeral session secret.")

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: warm the Redis connection so the first request isn't slow, and
        # for local SQLite create the schema (production schema is managed by
        # Alembic in the release phase).
        try:
            await app.state.redis.ping()
        except Exception as exc:  # pragma: no cover - best-effort warmup
            logger.warning("Redis warmup ping failed; first request may be slower", exc_info=exc)
        if settings.database_url.startswith("sqlite"):
            try:
                await init_models(app.state.db_engine)
            except Exception as exc:  # pragma: no cover - best-effort dev convenience
                logger.warning("SQLite schema init failed", exc_info=exc)
        yield
        # Shutdown: dispose the engine and close Redis.
        engine = getattr(app.state, "db_engine", None)
        if engine is not None:
            await engine.dispose()
        redis_client = getattr(app.state, "redis", None)
        if redis_client is not None:
            try:
                await redis_client.aclose()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass

    # Default /docs loads Swagger UI from a CDN, which the CSP (script-src 'self')
    # blocks — so docs are disabled here and re-served below from self-hosted
    # assets under /static/vendor/swagger-ui/.
    app = FastAPI(
        title="RegCheck",
        version=APP_VERSION,
        description=(
            "Compare study registrations with published papers. The keyed public "
            "API lives under /api/v1 (Authorization: Bearer rc_live_…)."
        ),
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )
    # Middleware: the LAST added is outermost. CurrentUserMiddleware reads
    # request.session, so it must run *inside* SessionMiddleware — i.e. added
    # before it here so SessionMiddleware ends up the outer layer.
    # SecurityHeadersMiddleware is added last so it is outermost and stamps
    # headers onto every response, including errors.
    app.add_middleware(CurrentUserMiddleware)
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.session_secret,
        https_only=settings.is_production,
        same_site="lax",
        max_age=14 * 24 * 60 * 60,
    )
    # CSP enforces by default; set CSP_REPORT_ONLY=1 to validate a tightened policy
    # on staging (logs violations without blocking) before enforcing.
    csp_report_only = (os.environ.get("CSP_REPORT_ONLY") or "").strip().lower() in {"1", "true", "yes", "on"}
    app.add_middleware(
        SecurityHeadersMiddleware, hsts=settings.is_production, csp_report_only=csp_report_only
    )
    # Outermost (added after SecurityHeaders): rescues cross-site iframe embeds
    # — chiefly browsers still rendering the cached pre-2026 registrar masking
    # page — by busting to the canonical site instead of a CSP-blanked frame.
    app.add_middleware(
        FrameBustMiddleware,
        canonical_base=(os.environ.get("CANONICAL_BASE_URL") or "https://regcheck.app").strip(),
    )
    # Per-IP rate limiting on cost-bearing submission endpoints is applied per
    # route via the comparison_rate_limit dependency (see core.rate_limit),
    # gated to production so it is a no-op locally and in tests.
    app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")
    # NB: uploaded source files are intentionally NOT served over HTTP. They are
    # handed to the worker via S3/Redis and rendered through gated report
    # artifacts; a public mount would bypass report visibility controls.

    app.state.settings = settings
    app.state.redis = create_redis_client(settings.redis_url)
    app.state.templates = Jinja2Templates(directory=settings.templates_dir)
    app.state.oauth = build_oauth(settings)
    app.state.db_engine = create_engine_from_url(settings.database_url)
    app.state.db_sessionmaker = create_sessionmaker(app.state.db_engine)

    # Expose the current user + static asset version to all templates.
    app.state.templates.env.globals["current_user"] = (
        lambda request: getattr(request.state, "user", None)
    )
    app.state.templates.env.globals["static_version"] = STATIC_VERSION
    app.state.templates.env.globals["brand_name"] = settings.brand_name
    # The quality tool lives on the separate PreCheck site; the Tools menu links out.
    app.state.templates.env.globals["precheck_url"] = (
        os.environ.get("PRECHECK_URL") or "https://precheck.app"
    ).strip()

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # Safety net so an unexpected error returns a clean 500 (and is logged
        # with a stack trace) instead of leaking internals. HTTPException and
        # validation errors keep their own FastAPI handlers.
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})

    @app.get("/docs", include_in_schema=False)
    async def swagger_docs():
        """Interactive API docs from self-hosted Swagger UI assets (the stock CDN
        build violates the site CSP). Assets are vendored under static/vendor/."""
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title="RegCheck API",
            swagger_js_url=f"/static/vendor/swagger-ui/swagger-ui-bundle.js?v={STATIC_VERSION}",
            swagger_css_url=f"/static/vendor/swagger-ui/swagger-ui.css?v={STATIC_VERSION}",
            swagger_favicon_url="/static/assets/favicon.ico",
        )

    @app.get("/health", include_in_schema=False)
    async def health() -> dict[str, str]:
        """Liveness probe for uptime monitors. Dependency-free by design so it
        stays green while Redis/DB hiccup (readiness is a separate concern)."""
        return {"status": "ok"}

    @app.get("/ready", include_in_schema=False)
    async def ready() -> JSONResponse:
        """Readiness probe: Redis, DB, and a live worker heartbeat (the worker
        refreshes worker:heartbeat:<id> with a short TTL, so key existence means
        fresh). Returns 503 with per-check status when any dependency is down.
        Check detail is a bare error class name — full errors go to the log, not
        to an unauthenticated endpoint."""
        checks: dict[str, str] = {}

        try:
            await app.state.redis.ping()
            checks["redis"] = "ok"
        except Exception as exc:
            logger.warning("Readiness: redis check failed", exc_info=exc)
            checks["redis"] = f"error: {type(exc).__name__}"

        try:
            async with app.state.db_sessionmaker() as db:
                await db.execute(sa_text("SELECT 1"))
            checks["database"] = "ok"
        except Exception as exc:
            logger.warning("Readiness: database check failed", exc_info=exc)
            checks["database"] = f"error: {type(exc).__name__}"

        if checks["redis"] == "ok":
            try:
                beats = await app.state.redis.keys("worker:heartbeat:*")
                checks["worker"] = f"ok ({len(beats)} live)" if beats else "error: no live worker heartbeat"
            except Exception as exc:
                logger.warning("Readiness: worker heartbeat check failed", exc_info=exc)
                checks["worker"] = f"error: {type(exc).__name__}"
        else:
            checks["worker"] = "unknown: redis unavailable"

        ok = all(v.startswith("ok") for v in checks.values())
        return JSONResponse(
            {"status": "ready" if ok else "unavailable", "checks": checks},
            status_code=200 if ok else 503,
        )

    # Only the keyed public API belongs in the OpenAPI spec; the web UI routes
    # are session-driven pages, not API surface.
    app.include_router(pages.router, include_in_schema=False)
    app.include_router(auth.router, include_in_schema=False)
    app.include_router(reports_routes.router, include_in_schema=False)
    app.include_router(api_routes.router)
    app.include_router(comparisons.router, include_in_schema=False)
    app.include_router(survey.router, include_in_schema=False)
    app.include_router(report.router, include_in_schema=False)
    app.include_router(status.router, include_in_schema=False)

    return app
