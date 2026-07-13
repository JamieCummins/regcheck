import json
import re

from backend.services import comparisons
from backend.services.report_html import build_standalone_report_html, write_report_html


def _sample_payloads():
    return [
        {
            "source": {"id": "registration", "label": "Preregistration", "kind": "text", "render_mode": "text"},
            "chunks": {"PREREG_0001": {"id": "PREREG_0001", "source_id": "registration", "text": "We will recruit 100."}},
            "render_data": {"kind": "text", "text": "We will recruit 100 participants."},
        },
        {
            "source": {"id": "paper", "label": "Paper", "kind": "pdf", "render_mode": "pdf"},
            "chunks": {"PAPER_0001": {"id": "PAPER_0001", "source_id": "paper", "text": "We recruited 92."}},
            "render_data": {"kind": "pdf", "text": "We recruited 92 participants.", "pages": []},
        },
    ]


def test_assemble_inline_bundle_shape_and_no_urls():
    manifest, render_data = comparisons._assemble_inline_bundle(
        None, "general_preregistration", _sample_payloads()
    )
    # Sources + chunks merged, keyed by id.
    assert set(manifest["sources"]) == {"registration", "paper"}
    assert set(manifest["chunks"]) == {"PREREG_0001", "PAPER_0001"}
    assert render_data["registration"]["text"].startswith("We will recruit")
    # URL-free so the offline viewer falls back to text rendering + quote tracing.
    for src in manifest["sources"].values():
        assert "page_url_template" not in src
        assert "render_data_url" not in src


def test_build_standalone_report_html_is_self_contained():
    items = [{"dimension": "Sample size", "deviation_judgement": "yes"}]
    manifest, render_data = comparisons._assemble_inline_bundle(
        None, "general_preregistration", _sample_payloads()
    )
    html = build_standalone_report_html(
        title="RegCheck · demo",
        items=items,
        manifest=manifest,
        render_data=render_data,
        meta="Model: gpustack",
    )
    # Inlines the viewer + bundle, references no external origins, no server fetch.
    assert "__REGCHECK_BUNDLE__" in html
    assert "report-view-toggle" in html and "view-documents" in html  # both panels present
    # No external resource fetches (Google Fonts @import is stripped); the only
    # remaining URL is the inert SVG XML namespace, which is never fetched.
    assert "fonts.googleapis.com" not in html
    assert "@import" not in html
    # The bundle round-trips and carries the items + chunks.
    blob = re.search(r"__REGCHECK_BUNDLE__ = (\{.*?\});</script>", html, re.S).group(1)
    bundle = json.loads(blob.replace("<\\/", "</"))
    assert bundle["items"] == items
    assert "PREREG_0001" in bundle["manifest"]["chunks"]


def test_write_report_html_round_trips(tmp_path):
    out = tmp_path / "report.html"
    manifest, render_data = comparisons._assemble_inline_bundle(
        None, "general_preregistration", _sample_payloads()
    )
    write_report_html(
        str(out),
        title="t",
        items=[{"dimension": "X", "deviation_judgement": "no"}],
        manifest=manifest,
        render_data=render_data,
    )
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    assert "__REGCHECK_BUNDLE__" in text
