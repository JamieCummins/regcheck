from __future__ import annotations

from backend.services.evidence import build_json_evidence_source, build_pdf_evidence_source


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
