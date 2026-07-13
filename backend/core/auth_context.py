from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp

from ..services.users import get_user


class CurrentUserMiddleware(BaseHTTPMiddleware):
    """Populate ``request.state.user`` from the signed session on every request.

    Must run *inside* SessionMiddleware (added before it in create_app) so
    ``request.session`` is available. Templates read the user via the
    ``current_user`` Jinja global, which falls back to None when unset.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        request.state.user = None
        try:
            user_id = request.session.get("user_id")
        except Exception:  # pragma: no cover - session not available
            user_id = None
        if user_id:
            sessionmaker = getattr(request.app.state, "db_sessionmaker", None)
            if sessionmaker is not None:
                try:
                    async with sessionmaker() as db:
                        request.state.user = await get_user(db, user_id)
                except Exception:  # pragma: no cover - defensive; never block a request on auth
                    request.state.user = None
        return await call_next(request)
