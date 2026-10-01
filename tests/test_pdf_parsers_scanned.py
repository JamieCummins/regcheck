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


def test_extract_bibr_text_reconstructs_sections_and_paragraphs():
    from backend.services.pdf_parsers import extract_bibr_text

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
    out = extract_bibr_text(payload)
    assert "Method" in out and "Results" in out
    # sentences in the same paragraph join on one line; paragraphs/sections separate
    assert "We recruited 200 people. Data collection stopped at 200." in out
    assert "The effect was significant." in out
    assert extract_bibr_text({}) == ""


def test_extract_bibr_text_drops_references_keeps_notes_and_tables():
    from backend.services.pdf_parsers import extract_bibr_text

    payload = {
        "section": [
            {"section_id": 1, "header": "Method", "section_type": "method"},
            {"section_id": 2, "header": "References", "section_type": "references"},
        ],
        "text": [
            {"text_id": 1, "section_id": 1, "paragraph_id": 1, "text": "We recruited 200 people."},
            {"text_id": 2, "section_id": 2, "paragraph_id": 2, "text": "Smith, J. (2020). A cited work."},
            {"text_id": 3, "section_id": None, "paragraph_id": 3, "text": "3 We corrected a scoring error."},
        ],
        "table": [{"label": "1", "html": "<table><tr><td>Group</td><td>N</td></tr><tr><td>A</td><td>103</td></tr></table>"}],
    }
    out = extract_bibr_text(payload)
    assert "We recruited 200 people." in out
    assert "Smith, J." not in out and "References" not in out
    assert "3 We corrected a scoring error." in out
    assert "Table 1\nGroup | N\nA | 103" in out


def test_extract_bibr_text_keeps_headings_of_sections_without_own_text():
    from backend.services.pdf_parsers import extract_bibr_text

    payload = {
        "section": [
            {"section_id": 1, "header": "Method"},
            {"section_id": 2, "header": "Results"},  # parent: all its text is in 2.1
            {"section_id": 3, "header": "Primary analyses"},
        ],
        "text": [
            {"section_id": 1, "paragraph_id": 1, "text": "We did X."},
            {"section_id": 3, "paragraph_id": 2, "text": "Effect found."},
        ],
    }
    assert extract_bibr_text(payload) == "Method\n\nWe did X.\n\nResults\n\nPrimary analyses\n\nEffect found."


def test_extract_bibr_text_tolerates_malformed_table_html_and_inline_latex():
    from backend.services.pdf_parsers import extract_bibr_text

    payload = {
        "section": [{"section_id": 1, "header": "Results", "section_type": "results"}],
        "text": [{"section_id": 1, "paragraph_id": 1, "text": r"All p \(\geq\) .05 (\(hat{α}\) = .8)."}],
        "table": [
            {
                "label": "2",
                # unclosed <br>, entity, inline LaTeX superscript, literal "\n"
                "html": r"<table><tr><td>Switch Group\(^{a}\)<br>(n = 511)</td><td>A &amp; B\nC</td></tr></table>",
            }
        ],
    }
    out = extract_bibr_text(payload)
    assert "All p ≥ .05 (α = .8)." in out
    assert "Table 2\nSwitch Groupa (n = 511) | A & B C" in out
    assert "<td>" not in out and "\\(" not in out


@pytest.mark.asyncio
async def test_pdf2bibr_submits_polls_and_fetches_result(tmp_path, monkeypatch):
    import httpx

    from backend.services import pdf_parsers

    pdf = tmp_path / "paper.pdf"
    _make_text_pdf(str(pdf), "body")
    calls: list[tuple[str, str, str]] = []
    polls = iter(["running", "succeeded"])

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path, request.headers.get("authorization", "")))
        if request.method == "POST":
            return httpx.Response(202, json={"job_id": "j1", "status": "queued"})
        if request.url.path.endswith("/result"):
            return httpx.Response(200, json={"text": [{"text": "ok", "section_id": 1, "paragraph_id": 1}]})
        return httpx.Response(200, json={"job_id": "j1", "status": next(polls)})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        pdf_parsers.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )
    monkeypatch.setenv("BIBR_URL", "https://bibr.example/")
    monkeypatch.setenv("BIBR_API_KEY", "tok")
    monkeypatch.setenv("BIBR_POLL_SECONDS", "0")

    payload = await pdf_parsers.pdf2bibr(str(pdf))
    assert payload["text"][0]["text"] == "ok"
    assert [c[:2] for c in calls] == [
        ("POST", "/papers/jobs"),
        ("GET", "/papers/jobs/j1"),
        ("GET", "/papers/jobs/j1"),
        ("GET", "/papers/jobs/j1/result"),
    ]
    assert all(c[2] == "Bearer tok" for c in calls)


@pytest.mark.asyncio
async def test_pdf2bibr_failed_job_raises(tmp_path, monkeypatch):
    import httpx

    from backend.services import pdf_parsers

    pdf = tmp_path / "paper.pdf"
    _make_text_pdf(str(pdf), "body")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(202, json={"job_id": "j1", "status": "queued"})
        return httpx.Response(200, json={"job_id": "j1", "status": "failed", "error": "ocr crashed"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        pdf_parsers.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )
    monkeypatch.setenv("BIBR_URL", "https://bibr.example")
    monkeypatch.setenv("BIBR_POLL_SECONDS", "0")
    with pytest.raises(RuntimeError, match="ocr crashed"):
        await pdf_parsers.pdf2bibr(str(pdf))


@pytest.mark.asyncio
async def test_extract_pdf_text_accepts_legacy_external_name(tmp_path):
    pdf = tmp_path / "paper.pdf"
    _make_text_pdf(str(pdf), "ignored")

    async def fake_bibr(_path: str):
        return {"text": [{"text": "From bibr.", "section_id": 1, "paragraph_id": 1}]}

    text, used = await extract_pdf_text(str(pdf), parser_choice="external", bibr_parser=fake_bibr)
    assert used == "bibr" and "From bibr." in text


@pytest.mark.asyncio
async def test_extract_pdf_text_bibr_primary(tmp_path):
    pdf = tmp_path / "paper.pdf"
    _make_text_pdf(str(pdf), "ignored — bibr parser is injected")

    async def fake_external(_path: str):
        return {
            "section": [{"section_id": 1, "header": "Introduction"}],
            "text": [{"text_id": 1, "section_id": 1, "paragraph_id": 1, "text": "Hello from the parser."}],
        }

    text, used = await extract_pdf_text(str(pdf), parser_choice="bibr", bibr_parser=fake_external)
    assert used == "bibr"
    assert "Hello from the parser." in text


@pytest.mark.asyncio
async def test_extract_pdf_text_bibr_failure_falls_back(tmp_path, monkeypatch):
    pdf = tmp_path / "paper.pdf"
    _make_text_pdf(str(pdf), "Selectable body text for PyMuPDF fallback.")

    async def broken_external(_path: str):
        raise RuntimeError("Missing BIBR_URL")

    monkeypatch.setenv("PDF_PARSER_FALLBACKS", "pymupdf")
    text, used = await extract_pdf_text(str(pdf), parser_choice="bibr", bibr_parser=broken_external)
    assert used == "pymupdf_fallback"
    assert "Selectable body text" in text


# ── external escalation is opt-in (default chain is local-only) ────────────────


@pytest.mark.asyncio
async def test_default_chain_never_calls_external_parser(tmp_path, monkeypatch):
    # No fallback env config: a grobid failure must recover via LOCAL pymupdf and
    # the document must never be sent to the external DPT2 service.
    monkeypatch.delenv("PDF_PARSER_FALLBACKS", raising=False)
    monkeypatch.delenv("SCANNED_PDF_FALLBACK", raising=False)
    pdf_path = tmp_path / "text.pdf"
    _make_text_pdf(str(pdf_path), "local rescue")

    async def fake_grobid_fail(_path: str) -> str:
        raise RuntimeError("grobid down")

    dpt_calls = []

    async def dpt_spy(_path: str):
        dpt_calls.append(_path)
        return {"text": "should never run"}

    extracted, used = await extract_pdf_text(
        str(pdf_path), parser_choice="grobid", pdf_parser=fake_grobid_fail, dpt_parser=dpt_spy
    )
    assert "local rescue" in extracted
    assert used == "pymupdf_fallback"
    assert dpt_calls == []


@pytest.mark.asyncio
async def test_default_chain_scanned_pdf_instructs_instead_of_external_ocr(tmp_path, monkeypatch):
    # A scanned PDF with no configured fallback fails with guidance rather than
    # silently OCRing via the external service.
    monkeypatch.delenv("PDF_PARSER_FALLBACKS", raising=False)
    monkeypatch.delenv("SCANNED_PDF_FALLBACK", raising=False)
    pdf_path = tmp_path / "scan.pdf"
    _make_scanned_pdf(str(pdf_path))

    dpt_calls = []

    async def dpt_spy(_path: str):
        dpt_calls.append(_path)
        return {"text": "should never run"}

    with pytest.raises(ValueError, match="no usable text"):
        await extract_pdf_text(str(pdf_path), parser_choice="pymupdf", dpt_parser=dpt_spy)
    assert dpt_calls == []
