"""Render a RegCheck comparison as a self-contained, offline HTML report.

The web report viewer (``static/js/report.js`` + ``static/css/report.css``) can be
driven entirely from a static ``{items, manifest, render_data}`` bundle — the same
shape the ``/demo`` page loads. This module inlines that bundle plus the viewer's
CSS/JS into a single ``.html`` file that opens directly in a browser (``file://``),
giving the full Overview + Evidence experience with quote tracing, no server needed.

Because the bundle's source entries carry no page-image URLs, the viewer falls back
to text rendering and locates quotes by string match — so the file is fully
self-contained (no page-image routes to fetch).
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
_STATIC = _ROOT / "static"

# Bootstrap utilities the viewer relies on (report.js toggles `d-none` to switch
# views). We inline just these rather than all of Bootstrap, which the report view
# does not otherwise need.
_UTILITY_CSS = """
.d-none { display: none !important; }
.visually-hidden { position: absolute !important; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }
"""


def _read(rel: str) -> str:
    return (_STATIC / rel).read_text(encoding="utf-8")


def _script_safe_json(data: Any) -> str:
    # Escape so the JSON can't terminate the surrounding <script> early.
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


def build_standalone_report_html(
    *,
    title: str,
    items: list[dict[str, Any]],
    manifest: dict[str, Any] | None,
    render_data: dict[str, Any] | None,
    meta: str | None = None,
) -> str:
    """Return a complete HTML document string for a self-contained report."""
    bundle = {
        "items": items or [],
        "manifest": manifest or None,
        "render_data": render_data or {},
    }
    custom_css = _read("css/custom-styles.css")
    report_css = _read("css/report.css")
    quote_locator_js = _read("js/quote-locator.js")
    report_js = _read("js/report.js")
    safe_title = html.escape(title or "RegCheck Report")
    meta_html = (
        f'<div class="report-page footer" style="margin-top:auto;font-size:.72rem;color:var(--rc-muted);'
        f'padding-top:1rem;border-top:1px solid var(--rc-border)">{html.escape(meta)}</div>'
        if meta
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{safe_title}</title>
<style>
{custom_css}
{report_css}
{_UTILITY_CSS}
</style>
</head>
<body class="text-bg-dark report-page page-shell">
  <div class="report-shell">
    <header class="report-header">
      <div class="report-topbar">
        <div class="report-topbar__lead">
          <div class="report-view-toggle d-none" id="report-view-toggle" role="group" aria-label="Report view">
            <button type="button" class="report-view-btn is-active" data-view="decision">Overview</button>
            <button type="button" class="report-view-btn" data-view="documents">Evidence</button>
          </div>
          <h1 id="results-page-tagline" class="report-title">{safe_title}</h1>
          <span class="report-status-pill" id="report-status-pill" aria-live="polite">
            <span class="report-status-pill__dot" id="report-status-dot"></span>
            <span id="report-status-text">Report</span>
          </span>
        </div>
        <div class="report-actions">
          <button type="button" class="report-action-btn report-action-btn--icon report-action-btn--ghost report-focus-btn" id="report-focus-btn" aria-label="Full screen" title="Full screen (more room for documents)" aria-pressed="false">
            <svg class="report-action-btn__icon report-focus-btn__expand" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M8 3H5a2 2 0 0 0-2 2v3"/><path d="M21 8V5a2 2 0 0 0-2-2h-3"/><path d="M3 16v3a2 2 0 0 0 2 2h3"/><path d="M16 21h3a2 2 0 0 0 2-2v-3"/></svg>
            <svg class="report-action-btn__icon report-focus-btn__compress" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M8 3v3a2 2 0 0 1-2 2H3"/><path d="M21 8h-3a2 2 0 0 1-2-2V3"/><path d="M3 16h3a2 2 0 0 1 2 2v3"/><path d="M16 21v-3a2 2 0 0 1 2-2h3"/></svg>
          </button>
        </div>
      </div>
    </header>

    <main id="report-app" class="report-workspace" data-task-id="local">
      <div id="view-decision" class="report-view report-view--decision">
        <aside class="report-dimension-rail" aria-label="Dimensions">
          <div class="rail-heading"><span>Dimensions</span><span id="dimension-count" class="rail-count">0</span></div>
          <div id="dimension-list" class="dimension-list"></div>
        </aside>
        <section class="report-detail" aria-label="Report detail">
          <div id="report-empty-state" class="report-empty-state">
            <span class="report-empty-state__spinner" aria-hidden="true"></span>
            <p>Preparing your report&hellip;</p>
          </div>
          <article id="dimension-detail" class="dimension-detail d-none"></article>
        </section>
        <aside class="report-quotes" aria-label="Summaries">
          <div id="quotes-pane" class="quotes-pane"></div>
        </aside>
      </div>
      <section id="view-documents" class="report-view report-view--documents d-none" aria-label="Source documents"></section>
    </main>
    {meta_html}
  </div>

  <div id="report-toast" class="report-toast" role="status" aria-live="polite"></div>

  <script>window.__REGCHECK_BUNDLE__ = {_script_safe_json(bundle)};</script>
  <script>
{quote_locator_js}
  </script>
  <script>
{report_js}
  </script>
</body>
</html>
"""


def write_report_html(
    path: str,
    *,
    title: str,
    items: list[dict[str, Any]],
    manifest: dict[str, Any] | None,
    render_data: dict[str, Any] | None,
    meta: str | None = None,
) -> str:
    """Render and write the report to ``path``; returns the path written."""
    document = build_standalone_report_html(
        title=title, items=items, manifest=manifest, render_data=render_data, meta=meta
    )
    Path(path).write_text(document, encoding="utf-8")
    return path
