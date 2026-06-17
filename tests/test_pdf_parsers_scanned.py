import fitz
import pytest

from backend.services.pdf_parsers import extract_pdf_text, is_likely_scanned_pdf


def _make_scanned_pdf(path):
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    pm = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 50, 50), 0)
    rect = fitz.Rect(0, 0, 200, 200)
    page.insert_image(rect, stream=pm.tobytes("png"))
    doc.save(path)
    doc.close()


def _make_text_pdf(path, text: str = "hello"):
    doc = fitz.open()
    page = doc.new_page(width=300, height=300)
    page.insert_text((50, 50), text)
    doc.save(path)
    doc.close()


def test_is_likely_scanned_pdf_true_for_image_only_pdf(tmp_path):
    pdf_path = tmp_path / "scan.pdf"
    _make_scanned_pdf(str(pdf_path))
    assert is_likely_scanned_pdf(str(pdf_path)) is True


@pytest.mark.asyncio
async def test_extract_pdf_text_scanned_pdf_instructs_when_no_fallback(tmp_path, monkeypatch):
    pdf_path = tmp_path / "scan.pdf"
    _make_scanned_pdf(str(pdf_path))

    async def fake_grobid(_path: str) -> str:
        return '<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body></body></text></TEI>'

    monkeypatch.setenv("SCANNED_PDF_FALLBACK", "none")
    with pytest.raises(ValueError, match="appears to be scanned"):
        await extract_pdf_text(str(pdf_path), parser_choice="grobid", pdf_parser=fake_grobid)


@pytest.mark.asyncio
async def test_extract_pdf_text_scanned_pdf_falls_back_to_dpt2(tmp_path, monkeypatch):
    pdf_path = tmp_path / "scan.pdf"
    _make_scanned_pdf(str(pdf_path))

    async def fake_grobid(_path: str) -> str:
        return '<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body></body></text></TEI>'

    async def fake_dpt(_path: str):
        return {"text": "x" * 500}

    monkeypatch.setenv("SCANNED_PDF_FALLBACK", "dpt2")
    extracted, used = await extract_pdf_text(
        str(pdf_path),
        parser_choice="grobid",
        pdf_parser=fake_grobid,
        dpt_parser=fake_dpt,
    )
    assert "x" * 200 in extracted
    assert used == "dpt2_fallback"


@pytest.mark.asyncio
async def test_extract_pdf_text_grobid_error_falls_back_to_dpt2(tmp_path, monkeypatch):
    pdf_path = tmp_path / "scan.pdf"
    _make_scanned_pdf(str(pdf_path))

    async def fake_grobid_fail(_path: str) -> str:
        raise RuntimeError("grobid 500")

    async def fake_dpt(_path: str):
        return {"text": "ocr success"}

    monkeypatch.setenv("SCANNED_PDF_FALLBACK", "dpt2")
    extracted, used = await extract_pdf_text(
        str(pdf_path),
        parser_choice="grobid",
        pdf_parser=fake_grobid_fail,
        dpt_parser=fake_dpt,
    )
    assert extracted.startswith("ocr success")
    assert used == "dpt2_fallback"


@pytest.mark.asyncio
async def test_extract_pdf_text_grobid_error_then_dpt_then_pymupdf(tmp_path, monkeypatch):
    pdf_path = tmp_path / "text.pdf"
    _make_text_pdf(str(pdf_path), "hello fallback")

    async def fake_grobid_fail(_path: str) -> str:
        raise RuntimeError("grobid boom")

    async def fake_dpt_fail(_path: str):
        raise RuntimeError("dpt failed")

    monkeypatch.setenv("PDF_PARSER_FALLBACKS", "dpt2,pymupdf")

    extracted, used = await extract_pdf_text(
        str(pdf_path),
        parser_choice="grobid",
        pdf_parser=fake_grobid_fail,
        dpt_parser=fake_dpt_fail,
    )

    assert "hello fallback" in extracted
    assert used == "pymupdf_fallback"


@pytest.mark.asyncio
async def test_extract_pdf_text_legacy_dpt2_falls_through_to_pymupdf(tmp_path, monkeypatch):
    pdf_path = tmp_path / "legacy-text.pdf"
    _make_text_pdf(str(pdf_path), "hello legacy fallback")

    async def fake_grobid_fail(_path: str) -> str:
        raise RuntimeError("grobid boom")

    async def fake_dpt_fail(_path: str):
        raise RuntimeError("dpt 403")

    monkeypatch.delenv("PDF_PARSER_FALLBACKS", raising=False)
    monkeypatch.setenv("SCANNED_PDF_FALLBACK", "dpt2")

    extracted, used = await extract_pdf_text(
        str(pdf_path),
        parser_choice="grobid",
        pdf_parser=fake_grobid_fail,
        dpt_parser=fake_dpt_fail,
    )

    assert "hello legacy fallback" in extracted
    assert used == "pymupdf_fallback"


@pytest.mark.asyncio
async def test_extract_pdf_text_single_dpt2_chain_falls_through_to_pymupdf(tmp_path, monkeypatch):
    pdf_path = tmp_path / "explicit-text.pdf"
    _make_text_pdf(str(pdf_path), "hello explicit fallback")

    async def fake_grobid_fail(_path: str) -> str:
        raise RuntimeError("grobid boom")

    async def fake_dpt_fail(_path: str):
        raise RuntimeError("dpt 403")

    monkeypatch.setenv("PDF_PARSER_FALLBACKS", "dpt2")

    extracted, used = await extract_pdf_text(
        str(pdf_path),
        parser_choice="grobid",
        pdf_parser=fake_grobid_fail,
        dpt_parser=fake_dpt_fail,
    )

    assert "hello explicit fallback" in extracted
    assert used == "pymupdf_fallback"


@pytest.mark.asyncio
async def test_extract_pdf_text_dpt2_mode_falls_back_to_pymupdf(tmp_path, monkeypatch):
    pdf_path = tmp_path / "text2.pdf"
    _make_text_pdf(str(pdf_path), "hello dpt mode")

    async def fake_dpt_fail(_path: str):
        raise RuntimeError("dpt failed")

    monkeypatch.setenv("PDF_PARSER_FALLBACKS", "dpt2,pymupdf")

    extracted, used = await extract_pdf_text(
        str(pdf_path),
        parser_choice="dpt2",
        dpt_parser=fake_dpt_fail,
    )

    assert "hello dpt mode" in extracted
    assert used == "pymupdf_fallback"


@pytest.mark.asyncio
async def test_extract_pdf_text_pymupdf_primary(tmp_path):
    """PyMuPDF is selectable as a primary, in-process parser and preserves
    all selectable text (e.g. author notes)."""
    pdf = tmp_path / "paper.pdf"
    _make_text_pdf(str(pdf), "Author note: corresponding author jane@example.org")
    text, used = await extract_pdf_text(str(pdf), parser_choice="pymupdf")
    assert used == "pymupdf"
    assert "Author note" in text


@pytest.mark.asyncio
async def test_extract_pdf_text_rejects_unknown_parser(tmp_path):
    pdf = tmp_path / "paper.pdf"
    _make_text_pdf(str(pdf), "body")
    with pytest.raises(ValueError):
        await extract_pdf_text(str(pdf), parser_choice="nope")


def test_extract_external_text_reconstructs_sections_and_paragraphs():
    from backend.services.pdf_parsers import extract_external_text

    payload = {
        "section": [
            {"section_id": 1, "header": "Method"},
            {"section_id": 2, "header": "Results"},
        ],
        "text": [
            {"text_id": 1, "section_id": 1, "paragraph_id": 1, "text": "We recruited 200 people."},
            {"text_id": 2, "section_id": 1, "paragraph_id": 1, "text": "Data collection stopped at 200."},
            {"text_id": 3, "section_id": 2, "paragraph_id": 2, "text": "The effect was significant."},
        ],
    }
    out = extract_external_text(payload)
    assert "Method" in out and "Results" in out
    # sentences in the same paragraph join on one line; paragraphs/sections separate
    assert "We recruited 200 people. Data collection stopped at 200." in out
    assert "The effect was significant." in out
    assert extract_external_text({}) == ""


@pytest.mark.asyncio
async def test_extract_pdf_text_external_primary(tmp_path):
    pdf = tmp_path / "paper.pdf"
    _make_text_pdf(str(pdf), "ignored — external parser is injected")

    async def fake_external(_path: str):
        return {
            "section": [{"section_id": 1, "header": "Introduction"}],
            "text": [{"text_id": 1, "section_id": 1, "paragraph_id": 1, "text": "Hello from the parser."}],
        }

    text, used = await extract_pdf_text(str(pdf), parser_choice="external", external_parser=fake_external)
    assert used == "external"
    assert "Hello from the parser." in text


@pytest.mark.asyncio
async def test_extract_pdf_text_external_failure_falls_back(tmp_path, monkeypatch):
    pdf = tmp_path / "paper.pdf"
    _make_text_pdf(str(pdf), "Selectable body text for PyMuPDF fallback.")

    async def broken_external(_path: str):
        raise RuntimeError("Missing EXTERNAL_PARSER_URL")

    monkeypatch.setenv("PDF_PARSER_FALLBACKS", "pymupdf")
    text, used = await extract_pdf_text(str(pdf), parser_choice="external", external_parser=broken_external)
    assert used == "pymupdf_fallback"
    assert "Selectable body text" in text
