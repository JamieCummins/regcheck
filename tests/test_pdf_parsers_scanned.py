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
