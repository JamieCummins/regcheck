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
    session_secret: str
    static_dir: str
    templates_dir: str
    upload_dir: str

    def ensure_directories(self) -> None:
        """Ensure that directories required by the application exist."""
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings."""
    load_dotenv()

    base_dir = Path(__file__).resolve().parents[2]
    redis_url = (
        os.environ.get("REDIS_URL")
        or os.environ.get("HEROKU_REDIS_OLIVE_URL")
        or "redis://localhost:6379/0"
    )
    session_secret = (os.environ.get("SESSION_SECRET") or "").strip() or secrets.token_urlsafe(32)

    static_dir = os.environ.get("STATIC_DIR", str(base_dir / "static"))
    templates_dir = os.environ.get("TEMPLATES_DIR", str(base_dir / "templates"))
    upload_dir = os.environ.get("UPLOAD_DIR", str(base_dir / "uploads"))

    settings = Settings(
        redis_url=redis_url,
        session_secret=session_secret,
        static_dir=static_dir,
        templates_dir=templates_dir,
        upload_dir=upload_dir,
    )
    settings.ensure_directories()
    return settings
