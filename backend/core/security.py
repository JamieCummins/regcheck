from __future__ import annotations

import hashlib
import secrets

# RegCheck API keys look like: rc_live_<43 url-safe base64 chars>.
API_KEY_PREFIX = "rc_live_"
# Number of leading characters (including the prefix) stored in clear for display.
API_KEY_DISPLAY_PREFIX_LEN = len(API_KEY_PREFIX) + 6


def generate_api_key() -> str:
    """Generate a new opaque API key (shown to the user exactly once)."""
    return f"{API_KEY_PREFIX}{secrets.token_urlsafe(32)}"


def hash_api_key(key: str) -> str:
    """Return the SHA-256 hex digest used for storage and lookup.

    API keys are high-entropy random tokens, so a fast hash is appropriate
    (and required for constant-time DB lookup by hash).
    """
    return hashlib.sha256((key or "").strip().encode("utf-8")).hexdigest()


def api_key_display_prefix(key: str) -> str:
    """A short, non-secret fragment of the key for UI display/identification."""
    return (key or "")[:API_KEY_DISPLAY_PREFIX_LEN]


def looks_like_api_key(value: str | None) -> bool:
    return bool(value) and value.strip().startswith(API_KEY_PREFIX)
