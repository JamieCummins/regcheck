from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from .core.config import get_settings
from .core.logging import configure_logging
from .core.redis import create_redis_client
from .routes import comparisons, pages, status
from .routes import survey


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging()
    logger = logging.getLogger(__name__)
    if not (os.environ.get("SESSION_SECRET") or "").strip():
        logger.warning("SESSION_SECRET not set; using an ephemeral session secret.")

    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key=settings.session_secret)
    app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")
    app.mount("/uploads", StaticFiles(directory=settings.upload_dir), name="uploads")

    app.state.settings = settings
    app.state.redis = create_redis_client(settings.redis_url)
    app.state.templates = Jinja2Templates(directory=settings.templates_dir)

    @app.on_event("startup")
    async def warm_redis_connection() -> None:
        try:
            await app.state.redis.ping()
        except Exception as exc:  # pragma: no cover - best-effort warmup
            logger.warning("Redis warmup ping failed; first request may be slower", exc_info=exc)

    app.include_router(pages.router)
    app.include_router(comparisons.router)
    app.include_router(survey.router)
    app.include_router(status.router)

    return app
