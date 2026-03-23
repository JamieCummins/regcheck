from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from backend.core.database import create_db_and_tables, create_engine, seed

from .core.config import get_settings
from .core.logging import configure_logging
from .core.redis import create_redis_client, try_redis_ping
from .routes import comparisons, pages, status
from .routes import survey
from .routes import user
from .routes import results


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging()
    logger = logging.getLogger(__name__)
    if not (os.environ.get("SESSION_SECRET") or "").strip():
        logger.warning("SESSION_SECRET not set; using an ephemeral session secret.")

    redis = create_redis_client(settings.redis_url)
    create_engine(settings.db_url)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # On startup
        create_db_and_tables()
        seed()  # TODO: Remove later
        await try_redis_ping(redis, logger)

        yield

        # On shutdown
        ...

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(SessionMiddleware, secret_key=settings.session_secret)
    app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")
    app.mount("/uploads", StaticFiles(directory=settings.upload_dir), name="uploads")

    app.state.settings = settings
    app.state.redis = redis
    app.state.templates = Jinja2Templates(directory=settings.templates_dir)

    app.include_router(pages.router)
    app.include_router(comparisons.router)
    app.include_router(survey.router)
    app.include_router(status.router)
    app.include_router(user.router)
    app.include_router(results.router)

    return app
