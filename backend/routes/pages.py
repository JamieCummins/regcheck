from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from ..services.dimensions import discipline_sets_for_ui, registration_quality_set_for_ui

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse(request, "index.html")


@router.get("/compare", response_class=HTMLResponse, name="compare")
async def compare(request: Request):
    templates = request.app.state.templates
    # The discipline dimension presets are defined once in the backend and injected
    # here so the wizard, API, and CLI all resolve the same dimensions/definitions.
    return templates.TemplateResponse(
        request,
        "general_preregistration.html",
        {"discipline_sets_json": json.dumps(discipline_sets_for_ui())},
    )


@router.get("/evaluate_registration", response_class=HTMLResponse, name="evaluate_registration")
async def evaluate_registration(request: Request):
    templates = request.app.state.templates
    # Same wizard machinery as /compare, parametrized for the single-document
    # quality flow; the only "discipline" is the registration-quality criteria set.
    return templates.TemplateResponse(
        request,
        "registration_quality.html",
        {"discipline_sets_json": json.dumps(registration_quality_set_for_ui())},
    )


@router.get("/clinical_trials", response_class=HTMLResponse, name="clinical_trials")
async def clinical_trials_get(request: Request):
    return RedirectResponse(url=request.url_for("compare"), status_code=302)


@router.get(
    "/general_preregistration",
    response_class=HTMLResponse,
    name="general_preregistration",
)
async def general_preregistration_get(request: Request):
    return RedirectResponse(url=request.url_for("compare"), status_code=302)


@router.get("/contact", response_class=HTMLResponse)
async def contact(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse(request, "contact.html")


@router.get("/demo", name="demo")
async def demo(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse(request, "demo.html")


# Tools that are planned but not yet built. The nav links point here so a click
# explains the status instead of doing nothing.
_COMING_SOON = {
    "code-paper": {
        "title": "Code–Paper Comparison",
        "blurb": "Compare a study's analysis code against what its paper reports, to check that the published results match the code that produced them.",
    },
}


@router.get("/coming-soon/{feature}", response_class=HTMLResponse, name="coming_soon")
async def coming_soon(request: Request, feature: str):
    templates = request.app.state.templates
    info = _COMING_SOON.get(
        feature,
        {"title": "This tool", "blurb": "This RegCheck tool is in development."},
    )
    return templates.TemplateResponse(
        request,
        "coming_soon.html",
        {"feature_title": info["title"], "feature_blurb": info["blurb"]},
    )


@router.get("/team", name="team")
async def team(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse(request, "team.html")


@router.get("/jobs", name="jobs")
async def jobs(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse(request, "jobs.html")


@router.get("/privacy", response_class=HTMLResponse, name="privacy")
async def privacy(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse(request, "privacy.html")


@router.get("/faq", response_class=HTMLResponse, name="faq")
async def faq(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse(request, "faq.html")


@router.get("/api", response_class=HTMLResponse, name="api_docs")
async def api_docs(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse(request, "api.html")
