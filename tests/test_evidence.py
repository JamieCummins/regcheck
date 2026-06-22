from __future__ import annotations

from pathlib import Path

import pytest

from backend.services.evidence import (
    build_file_evidence_source,
    build_json_evidence_source,
    build_pdf_evidence_source,
)


def test_build_pdf_evidence_source_locates_text_rectangles(tmp_path):
    import fitz

    pdf_path = tmp_path / "paper.pdf"
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((36, 72), "Planned enrollment is 120 participants.")
    doc.save(pdf_path)
    doc.close()

    payload = build_pdf_evidence_source(
        source_id="paper",
        label="Paper",
        pdf_path=str(pdf_path),
        chunk_prefix="PAPER",
        max_chunk_tokens=100,
    )

    chunk = payload["chunks"]["PAPER_0001"]
    pdf_locations = [location for location in chunk["locations"] if location["kind"] == "pdf"]
    assert payload["source"]["kind"] == "pdf"
    assert payload["source"]["page_count"] == 1
    assert pdf_locations
    assert pdf_locations[0]["page"] == 1
    assert pdf_locations[0]["rects"]


def test_build_json_evidence_source_tracks_text_spans():
    payload = build_json_evidence_source(
        source_id="registration",
        label="ClinicalTrials.gov Registration",
        data={"status": {"enrollment": "120", "startDate": "2024-01-01"}},
        chunk_prefix="PREREG",
        max_chunk_tokens=100,
    )

    chunk = payload["chunks"]["PREREG_0001"]
    location = chunk["locations"][0]

    assert payload["source"]["kind"] == "json"
    assert payload["render_data"]["rows"]
    assert location["kind"] == "json"
    assert payload["text"][location["start"] : location["end"]] == chunk["text"]


def test_pdf_rects_cover_multiline_and_hyphenated_text(tmp_path):
    """Word-sequence matching must locate chunks that wrap lines and break on
    hyphens — cases where page.search_for() fails or covers only a prefix."""
    import fitz

    pdf_path = tmp_path / "wrapped.pdf"
    doc = fitz.open()
    page = doc.new_page(width=300, height=300)
    # Simulate a hyphenated line wrap followed by more lines.
    page.insert_text((36, 72), "The planned enrollment is one hun-")
    page.insert_text((36, 90), "dred and twenty participants total,")
    page.insert_text((36, 108), "recruited online via Prolific.")
    doc.save(pdf_path)
    doc.close()

    payload = build_pdf_evidence_source(
        source_id="paper",
        label="Paper",
        pdf_path=str(pdf_path),
        chunk_prefix="PAPER",
        max_chunk_tokens=200,
    )

    chunk = payload["chunks"]["PAPER_0001"]
    pdf_locations = [loc for loc in chunk["locations"] if loc["kind"] == "pdf"]
    assert pdf_locations
    rects = pdf_locations[0]["rects"]
    # Full coverage: one merged rect per text line, not a truncated prefix.
    assert len(rects) >= 3
    tops = sorted({round(rect["y0"]) for rect in rects})
    assert len(tops) >= 3


def test_pdf_source_pages_carry_char_offsets(tmp_path):
    """Manifest pages must expose start/end offsets so the viewer can map a
    text span to its page when rect highlighting is unavailable."""
    import fitz

    pdf_path = tmp_path / "two-pages.pdf"
    doc = fitz.open()
    for label in ("First page body text.", "Second page body text."):
        page = doc.new_page(width=300, height=200)
        page.insert_text((36, 72), label)
    doc.save(pdf_path)
    doc.close()

    payload = build_pdf_evidence_source(
        source_id="paper",
        label="Paper",
        pdf_path=str(pdf_path),
        chunk_prefix="PAPER",
        max_chunk_tokens=100,
    )

    pages = payload["source"]["pages"]
    assert len(pages) == 2
    text = payload["text"]
    for page_info in pages:
        assert page_info["end"] > page_info["start"]
    assert "First page" in text[pages[0]["start"] : pages[0]["end"]]
    assert "Second page" in text[pages[1]["start"] : pages[1]["end"]]
    assert payload["render_data"]["pages"] == pages


REAL_PAPER_PDF = Path(__file__).resolve().parent.parent / "test_materials" / "paper_cumminsetal_2023.pdf"


@pytest.mark.skipif(not REAL_PAPER_PDF.exists(), reason="real paper fixture not present")
def test_pdf_rect_coverage_on_real_paper():
    """End-to-end coverage guard: every chunk of a real multi-page paper must
    get rect highlights (the old single-needle search managed only ~91%)."""
    payload = build_pdf_evidence_source(
        source_id="paper",
        label="Paper",
        pdf_path=str(REAL_PAPER_PDF),
        chunk_prefix="PAPER",
    )
    chunks = payload["chunks"]
    assert chunks
    with_rects = sum(
        1 for chunk in chunks.values()
        if any(loc["kind"] == "pdf" and loc.get("rects") for loc in chunk["locations"])
    )
    coverage = with_rects / len(chunks)
    assert coverage >= 0.98, f"rect coverage regressed: {with_rects}/{len(chunks)}"


def test_build_file_evidence_source_renders_docx_text_as_pdf(tmp_path):
    docx_path = tmp_path / "registration.docx"
    docx_path.write_bytes(b"placeholder")

    payload = build_file_evidence_source(
        source_id="registration",
        label="Registration",
        file_path=str(docx_path),
        file_ext=".docx",
        text="The planned sample size is 120 participants.",
        chunk_prefix="PREREG",
        max_chunk_tokens=100,
    )

    chunk = payload["chunks"]["PREREG_0001"]
    locations = [location for location in chunk["locations"] if location["kind"] == "pdf"]

    assert payload["source"]["kind"] == "pdf"
    assert payload["source"]["render_mode"] == "pdf"
    assert payload["raw_content_type"] == "application/pdf"
    assert payload["raw_bytes"].startswith(b"%PDF")
    assert locations
    assert locations[0]["rects"]


def test_text_pdf_render_paginates_without_dropping_content():
    """The renderer must not drop content: the old fixed-2800-char slice could
    overflow a page (text with many short lines) and silently clip + skip it."""
    fitz = pytest.importorskip("fitz")
    from backend.services.evidence import _build_text_pdf_render

    # Many short lines — the layout that defeated the fixed-char slicer.
    lines = [f"Line {i:03d}: the quick brown fox jumps over the lazy dog." for i in range(400)]
    pdf = _build_text_pdf_render("\n".join(lines), title="Preregistration")
    assert pdf

    doc = fitz.open(stream=pdf, filetype="pdf")
    rendered = "\n".join(doc[i].get_text() for i in range(len(doc)))
    # Every line survives (no dropped content)...
    for i in range(400):
        assert f"Line {i:03d}" in rendered, f"Line {i:03d} was dropped"
    # ...and no blank/near-blank pages.
    assert all(len(doc[i].get_text().strip()) > 30 for i in range(len(doc)))
