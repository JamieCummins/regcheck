from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from .core.auth_context import CurrentUserMiddleware
from .core.config import get_settings
from .core.logging import configure_logging
from .core.security_headers import SecurityHeadersMiddleware
from .core.oauth import build_oauth
from .core.redis import create_redis_client
from .db.session import create_engine_from_url, create_sessionmaker, init_models
from .routes import auth, comparisons, pages, report, status
from .routes import api as api_routes
from .routes import reports as reports_routes
from .routes import survey


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging()
    logger = logging.getLogger(__name__)
    if not (os.environ.get("SESSION_SECRET") or "").strip():
        logger.warning("SESSION_SECRET not set; using an ephemeral session secret.")

    app = FastAPI()
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
    app.add_middleware(SecurityHeadersMiddleware, hsts=settings.is_production)
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

    # Expose the current user to all templates (navbar account menu, etc.).
    app.state.templates.env.globals["current_user"] = (
        lambda request: getattr(request.state, "user", None)
    )

    @app.on_event("startup")
    async def warm_redis_connection() -> None:
        try:
            await app.state.redis.ping()
        except Exception as exc:  # pragma: no cover - best-effort warmup
            logger.warning("Redis warmup ping failed; first request may be slower", exc_info=exc)

    @app.on_event("startup")
    async def prepare_database() -> None:
        # Production schema is managed by Alembic (release phase). For local
        # SQLite development, create tables on startup so the app is usable
        # without running migrations.
        if settings.database_url.startswith("sqlite"):
            try:
                await init_models(app.state.db_engine)
            except Exception as exc:  # pragma: no cover - best-effort dev convenience
                logger.warning("SQLite schema init failed", exc_info=exc)

    @app.on_event("shutdown")
    async def dispose_resources() -> None:
        engine = getattr(app.state, "db_engine", None)
        if engine is not None:
            await engine.dispose()
        redis_client = getattr(app.state, "redis", None)
        if redis_client is not None:
            try:
                await redis_client.aclose()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass

    @app.get("/health", include_in_schema=False)
    async def health() -> dict[str, str]:
        """Liveness probe for uptime monitors. Dependency-free by design so it
        stays green while Redis/DB hiccup (readiness is a separate concern)."""
        return {"status": "ok"}

    app.include_router(pages.router)
    app.include_router(auth.router)
    app.include_router(reports_routes.router)
    app.include_router(api_routes.router)
    app.include_router(comparisons.router)
    app.include_router(survey.router)
    app.include_router(report.router)
    app.include_router(status.router)

    return app
