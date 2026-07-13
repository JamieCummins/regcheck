from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from authlib.integrations.starlette_client import OAuth

from .config import Settings

GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"


@dataclass(frozen=True)
class ProviderInfo:
    key: str
    label: str


def build_oauth(settings: Settings) -> OAuth:
    """Build an Authlib OAuth registry, registering only configured providers."""
    oauth = OAuth()

    if settings.google_oauth_enabled:
        oauth.register(
            name="google",
            client_id=settings.google_client_id,
            client_secret=settings.google_client_secret,
            server_metadata_url=GOOGLE_DISCOVERY_URL,
            client_kwargs={"scope": "openid email profile"},
        )

    if settings.orcid_oauth_enabled:
        base = "https://sandbox.orcid.org" if settings.orcid_sandbox else "https://orcid.org"
        oauth.register(
            name="orcid",
            client_id=settings.orcid_client_id,
            client_secret=settings.orcid_client_secret,
            authorize_url=f"{base}/oauth/authorize",
            access_token_url=f"{base}/oauth/token",
            client_kwargs={"scope": "/authenticate"},
        )

    return oauth


def available_providers(settings: Settings) -> list[ProviderInfo]:
    providers: list[ProviderInfo] = []
    if settings.google_oauth_enabled:
        providers.append(ProviderInfo("google", "Google"))
    if settings.orcid_oauth_enabled:
        providers.append(ProviderInfo("orcid", "ORCID"))
    return providers


@dataclass(frozen=True)
class OAuthIdentityInfo:
    provider: str
    subject: str
    email: str | None
    email_verified: bool
    display_name: str | None


def extract_identity(provider: str, token: dict[str, Any]) -> OAuthIdentityInfo:
    """Normalize an Authlib token/userinfo into a provider-agnostic identity.

    - Google (OIDC): identity comes from the parsed `userinfo` (id_token claims).
    - ORCID (/authenticate): the iD and name are returned at the top level of the
      token; ORCID does not return an email under this scope.
    """
    if provider == "google":
        userinfo = token.get("userinfo") or {}
        subject = str(userinfo.get("sub") or "").strip()
        email = (userinfo.get("email") or None)
        return OAuthIdentityInfo(
            provider="google",
            subject=subject,
            email=email,
            email_verified=bool(userinfo.get("email_verified")),
            display_name=(userinfo.get("name") or None),
        )

    if provider == "orcid":
        subject = str(token.get("orcid") or "").strip()
        name = token.get("name") or None
        return OAuthIdentityInfo(
            provider="orcid",
            subject=subject,
            email=None,
            email_verified=False,
            display_name=name,
        )

    raise ValueError(f"Unsupported OAuth provider: {provider}")
