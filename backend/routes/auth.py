from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.oauth import available_providers, extract_identity
from ..db.session import get_db
from ..services import reports as reports_service
from ..services.users import (
    create_api_key,
    get_user,
    list_api_keys,
    revoke_api_key,
    update_profile,
    upsert_oauth_user,
)
from .survey import SURVEY_QUESTIONS

router = APIRouter()
logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}


def _settings(request: Request):
    return request.app.state.settings


def _provider_keys(request: Request) -> set[str]:
    return {p.key for p in available_providers(_settings(request))}


def _callback_url(request: Request, provider: str) -> str:
    base = (_settings(request).oauth_redirect_base_url or "").rstrip("/")
    if base:
        return f"{base}/auth/{provider}/callback"
    return str(request.url_for("oauth_callback", provider=provider))


def _safe_next(raw: str | None) -> str:
    """Only allow same-site relative redirects to avoid open-redirects."""
    if raw and raw.startswith("/") and not raw.startswith("//"):
        return raw
    return "/"


@router.get("/login", name="login")
async def login(request: Request, next: str = "/"):
    if getattr(request.state, "user", None) is not None:
        return RedirectResponse(_safe_next(next), status_code=302)
    settings = _settings(request)
    return request.app.state.templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "providers": available_providers(settings),
            "auth_enabled": settings.auth_enabled,
            "next": _safe_next(next),
        },
    )


@router.get("/auth/{provider}", name="oauth_login")
async def oauth_login(request: Request, provider: str, next: str = "/"):
    if provider not in _provider_keys(request):
        raise HTTPException(status_code=404, detail="Unknown or disabled sign-in provider")
    request.session["oauth_next"] = _safe_next(next)
    client = request.app.state.oauth.create_client(provider)
    return await client.authorize_redirect(request, _callback_url(request, provider))


@router.get("/auth/{provider}/callback", name="oauth_callback")
async def oauth_callback(request: Request, provider: str, db: AsyncSession = Depends(get_db)):
    if provider not in _provider_keys(request):
        raise HTTPException(status_code=404, detail="Unknown or disabled sign-in provider")
    client = request.app.state.oauth.create_client(provider)
    try:
        token = await client.authorize_access_token(request)
    except Exception as exc:
        logger.warning("OAuth callback failed for %s", provider, exc_info=exc)
        return RedirectResponse(url=request.url_for("login"), status_code=303)

    identity = extract_identity(provider, token)
    if not identity.subject:
        logger.warning("OAuth %s returned no stable subject id", provider)
        return RedirectResponse(url=request.url_for("login"), status_code=303)

    # Link to the currently signed-in user if there is one (adding a provider).
    link_to_user = getattr(request.state, "user", None)
    user = await upsert_oauth_user(db, identity, link_to_user=link_to_user)
    await db.commit()

    request.session["user_id"] = user.id

    # Claim any anonymous reports this browser created before signing in, so they're
    # kept (owned + off the 7-day TTL) instead of auto-deleting. Best effort.
    owned = request.session.get("owned_reports") or []
    if owned:
        try:
            claimed = await reports_service.claim_anonymous_reports(
                request.app.state.redis, db, owner_id=user.id, task_ids=list(owned)
            )
            await db.commit()
            if claimed:
                claimed_set = set(claimed)
                request.session["owned_reports"] = [t for t in owned if t not in claimed_set]
        except Exception:  # pragma: no cover - never block sign-in on claim failure
            logger.warning("Failed to claim anonymous reports on login", exc_info=True)

    next_url = _safe_next(request.session.pop("oauth_next", "/"))
    return RedirectResponse(url=next_url, status_code=303)


@router.post("/logout", name="logout")
async def logout(request: Request):
    request.session.pop("user_id", None)
    return RedirectResponse(url="/", status_code=303)


@router.get("/profile", name="profile")
async def profile(request: Request, db: AsyncSession = Depends(get_db)):
    state_user = getattr(request.state, "user", None)
    if state_user is None:
        return RedirectResponse(url=f"{request.url_for('login')}?next=/profile", status_code=302)
    user = await get_user(db, state_user.id)
    keys = await list_api_keys(db, user.id)
    # A freshly created key's plaintext is shown exactly once, then cleared.
    new_key = request.session.pop("new_api_key", None)
    return request.app.state.templates.TemplateResponse(
        "profile.html",
        {
            "request": request,
            "user": user,
            "questions": SURVEY_QUESTIONS,
            "saved": request.query_params.get("saved") == "1",
            "api_keys": keys,
            "new_api_key": new_key,
        },
    )


@router.post("/profile")
async def profile_update(
    request: Request,
    db: AsyncSession = Depends(get_db),
    display_name: str | None = Form(None),
    handle: str | None = Form(None),
    is_public_profile: str | None = Form(None),
    use_case: str | None = Form(None),
    academic_position: str | None = Form(None),
    research_field: str | None = Form(None),
):
    state_user = getattr(request.state, "user", None)
    if state_user is None:
        return RedirectResponse(url=request.url_for("login"), status_code=302)
    user = await get_user(db, state_user.id)
    await update_profile(
        db,
        user,
        display_name=display_name,
        handle=handle,
        is_public_profile=(is_public_profile or "").strip().lower() in _TRUTHY,
        use_case=use_case,
        academic_position=academic_position,
        research_field=research_field,
    )
    await db.commit()
    return RedirectResponse(url=f"{request.url_for('profile')}?saved=1", status_code=303)


@router.post("/profile/api-keys", name="api_key_create")
async def api_key_create(
    request: Request,
    db: AsyncSession = Depends(get_db),
    name: str | None = Form(None),
):
    state_user = getattr(request.state, "user", None)
    if state_user is None:
        return RedirectResponse(url=request.url_for("login"), status_code=302)
    user = await get_user(db, state_user.id)
    _key, raw = await create_api_key(db, user, name)
    await db.commit()
    # Stash the plaintext for a one-time display on the profile page.
    request.session["new_api_key"] = raw
    return RedirectResponse(url=request.url_for("profile"), status_code=303)


@router.post("/profile/api-keys/{key_id}/revoke", name="api_key_revoke")
async def api_key_revoke(request: Request, key_id: str, db: AsyncSession = Depends(get_db)):
    state_user = getattr(request.state, "user", None)
    if state_user is None:
        return RedirectResponse(url=request.url_for("login"), status_code=302)
    await revoke_api_key(db, state_user.id, key_id)
    await db.commit()
    return RedirectResponse(url=request.url_for("profile"), status_code=303)
