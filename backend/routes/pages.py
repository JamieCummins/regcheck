from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/compare", response_class=HTMLResponse, name="compare")
async def compare(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse("general_preregistration.html", {"request": request})


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
    return templates.TemplateResponse("contact.html", {"request": request})


@router.get("/demo", name="demo")
async def demo(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse("demo.html", {"request": request})


# Tools that are planned but not yet built. The nav links point here so a click
# explains the status instead of doing nothing.
_COMING_SOON = {
    "code-paper": {
        "title": "Code–Paper Comparison",
        "blurb": "Compare a study's analysis code against what its paper reports, to check that the published results match the code that produced them.",
    },
    "evaluate-registration": {
        "title": "Evaluate Registration Quality",
        "blurb": "Assess how complete and specific a preregistration is, so you can strengthen it before a study runs — or review it more efficiently.",
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
        "coming_soon.html",
        {"request": request, "feature_title": info["title"], "feature_blurb": info["blurb"]},
    )


@router.get("/team", name="team")
async def team(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse("team.html", {"request": request})


@router.get("/jobs", name="jobs")
async def jobs(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse("jobs.html", {"request": request})


@router.get("/privacy", response_class=HTMLResponse, name="privacy")
async def privacy(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse("privacy.html", {"request": request})


@router.get("/faq", response_class=HTMLResponse, name="faq")
async def faq(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse("faq.html", {"request": request})


@router.get("/api", response_class=HTMLResponse, name="api_docs")
async def api_docs(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse("api.html", {"request": request})
