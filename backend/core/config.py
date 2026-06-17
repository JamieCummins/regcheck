from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


@dataclass(slots=True)
class Settings:
    """Application configuration values loaded from the environment."""

    redis_url: str
    database_url: str
    session_secret: str
    task_ttl_seconds: int
    anonymous_task_ttl_seconds: int
    max_queue_length: int
    static_dir: str
    templates_dir: str
    upload_dir: str
    google_client_id: str
    google_client_secret: str
    orcid_client_id: str
    orcid_client_secret: str
    orcid_sandbox: bool
    oauth_redirect_base_url: str
    external_parser_url: str
    external_parser_api_key: str
    is_production: bool

    def ensure_directories(self) -> None:
        """Ensure that directories required by the application exist."""
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)

    @property
    def external_parser_enabled(self) -> bool:
        """The optional external structured parser is offered only when its service URL is set."""
        return bool(self.external_parser_url)

    @property
    def google_oauth_enabled(self) -> bool:
        return bool(self.google_client_id and self.google_client_secret)

    @property
    def orcid_oauth_enabled(self) -> bool:
        return bool(self.orcid_client_id and self.orcid_client_secret)

    @property
    def auth_enabled(self) -> bool:
        return self.google_oauth_enabled or self.orcid_oauth_enabled


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings."""
    load_dotenv()

    base_dir = Path(__file__).resolve().parents[2]
    require_session_secret = (os.environ.get("REQUIRE_SESSION_SECRET") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    redis_url = (
        os.environ.get("REDIS_TLS_URL")
        or os.environ.get("REDIS_URL")
        or os.environ.get("HEROKU_REDIS_OLIVE_TLS_URL")
        or os.environ.get("HEROKU_REDIS_OLIVE_URL")
        or os.environ.get("REDISCLOUD_URL")
        or os.environ.get("REDISGREEN_URL")
        or "redis://localhost:6379/0"
    )
    if os.environ.get("DYNO") and redis_url.startswith("redis://localhost"):
        raise RuntimeError("REDIS_URL/REDIS_TLS_URL must be set for production deployments.")
    session_secret_env = (os.environ.get("SESSION_SECRET") or "").strip()
    if (os.environ.get("DYNO") or require_session_secret) and not session_secret_env:
        raise RuntimeError("SESSION_SECRET must be set for production deployments.")
    session_secret = session_secret_env or secrets.token_urlsafe(32)

    def _int_env(name: str, default: int, minimum: int = 1) -> int:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            parsed = int(str(raw).strip())
            return max(minimum, parsed)
        except (TypeError, ValueError):
            return default

    task_ttl_seconds = _int_env("TASK_TTL_SECONDS", 3 * 24 * 60 * 60, minimum=60)
    # Anonymous (not signed-in) reports auto-delete after this window.
    anonymous_task_ttl_seconds = _int_env(
        "ANONYMOUS_TASK_TTL_SECONDS", 7 * 24 * 60 * 60, minimum=60
    )
    max_queue_length = _int_env("MAX_QUEUE_LENGTH", 200, minimum=1)
    # Production-like deployment: served over HTTPS, so the session cookie is
    # marked Secure and HSTS is emitted. Detected from Heroku's DYNO marker, with
    # an explicit override for non-Heroku HTTPS hosts.
    is_production = bool((os.environ.get("DYNO") or "").strip()) or (
        (os.environ.get("FORCE_SECURE_COOKIES") or "").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    static_dir = os.environ.get("STATIC_DIR", str(base_dir / "static"))
    templates_dir = os.environ.get("TEMPLATES_DIR", str(base_dir / "templates"))
    upload_dir = os.environ.get("UPLOAD_DIR", str(base_dir / "uploads"))

    def _str_env(name: str) -> str:
        return (os.environ.get(name) or "").strip()

    def _bool_env(name: str) -> bool:
        return _str_env(name).lower() in {"1", "true", "yes", "on"}

    # Resolved lazily by the db layer; stored here so the value is logged/visible
    # in one place. Imported here to avoid a hard dependency at module import.
    from ..db.session import resolve_database_url

    database_url = resolve_database_url()

    settings = Settings(
        redis_url=redis_url,
        database_url=database_url,
        session_secret=session_secret,
        task_ttl_seconds=task_ttl_seconds,
        anonymous_task_ttl_seconds=anonymous_task_ttl_seconds,
        max_queue_length=max_queue_length,
        static_dir=static_dir,
        templates_dir=templates_dir,
        upload_dir=upload_dir,
        google_client_id=_str_env("GOOGLE_CLIENT_ID"),
        google_client_secret=_str_env("GOOGLE_CLIENT_SECRET"),
        orcid_client_id=_str_env("ORCID_CLIENT_ID"),
        orcid_client_secret=_str_env("ORCID_CLIENT_SECRET"),
        orcid_sandbox=_bool_env("ORCID_SANDBOX"),
        oauth_redirect_base_url=_str_env("OAUTH_REDIRECT_BASE_URL"),
        external_parser_url=_str_env("EXTERNAL_PARSER_URL"),
        external_parser_api_key=_str_env("EXTERNAL_PARSER_API_KEY"),
        is_production=is_production,
    )
    settings.ensure_directories()
    return settings
