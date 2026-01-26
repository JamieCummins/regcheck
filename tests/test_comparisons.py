import os

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("GROQ_API_KEY", "test")
os.environ.setdefault("DEEPSEEK_API_KEY", "test")

from backend.services.comparisons import (  # noqa: E402
    ComparisonResult,
    clinical_trial_comparison,
    general_preregistration_comparison,
)

@pytest.mark.asyncio
async def test_general_preregistration_comparison(tmp_path):
    prereg = tmp_path / "prereg.txt"
    prereg.write_text("prereg")
    paper = tmp_path / "paper.pdf"
    paper.write_text("paper")

    async def fake_pdf_parser(path: str) -> str:
        return '<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body>paper body</body></text></TEI>'

    called = {}
    def fake_run(
        preregistration_input: str,
        extracted_paper_sections: str,
        client_choice: str,
        dimension_query: str,
        dimension_definition: str | None = None,
        **kwargs,
    ) -> ComparisonResult:
        called["dim"] = dimension_query
        called["definition"] = dimension_definition
        return ComparisonResult(items=[])

    res = await general_preregistration_comparison(
        str(prereg),
        ".txt",
        str(paper),
        ".pdf",
        "openai",
        "grobid",
        selected_dimensions=[{"dimension": "general", "definition": "custom def"}],
        pdf_parser=fake_pdf_parser,
        comparison_runner=fake_run,
    )
    assert called["dim"] == "general"
    assert called["definition"] == "custom def"
    assert isinstance(res, ComparisonResult)


def _make_scanned_pdf(path):
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    pm = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 50, 50), 0)
    rect = fitz.Rect(0, 0, 200, 200)
    page.insert_image(rect, stream=pm.tobytes("png"))
    doc.save(path)
    doc.close()


@pytest.mark.asyncio
async def test_general_preregistration_comparison_pdf_scanned_fallback(tmp_path, monkeypatch):
    prereg_pdf = tmp_path / "prereg.pdf"
    _make_scanned_pdf(str(prereg_pdf))
    paper = tmp_path / "paper.pdf"
    _make_scanned_pdf(str(paper))

    async def fake_grobid(path: str) -> str:
        if "prereg" in path:
            return '<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body></body></text></TEI>'
        return '<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body>paper body</body></text></TEI>'

    async def fake_dpt(path: str):
        return {"text": "registration body"}

    called = {}

    def fake_run(
        preregistration_input: str,
        extracted_paper_sections: str,
        client_choice: str,
        dimension_query: str,
        dimension_definition: str | None = None,
        **kwargs,
    ) -> ComparisonResult:
        called["prereg"] = preregistration_input
        return ComparisonResult(items=[])

    monkeypatch.setenv("SCANNED_PDF_FALLBACK", "dpt2")

    await general_preregistration_comparison(
        str(prereg_pdf),
        ".pdf",
        str(paper),
        ".pdf",
        "openai",
        "grobid",
        selected_dimensions=[{"dimension": "general", "definition": "custom def"}],
        pdf_parser=fake_grobid,
        dpt_parser=fake_dpt,
        comparison_runner=fake_run,
    )

    assert called["prereg"].startswith("registration body")

@pytest.mark.asyncio
async def test_clinical_trial_comparison(tmp_path):
    paper = tmp_path / "paper.pdf"
    paper.write_text("paper")

    async def fake_pdf_parser(path: str) -> str:
        return '<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body>paper body</body></text></TEI>'

    calls = []
    definitions = []
    def fake_run(
        preregistration_input: str,
        extracted_paper_sections: str,
        client_choice: str,
        dimension_query: str,
        dimension_definition: str | None = None,
        **kwargs,
    ) -> ComparisonResult:
        calls.append(dimension_query)
        definitions.append(dimension_definition)
        return ComparisonResult(items=[])

    selected_dims = [
        {"dimension": "Design: Planned sample size", "definition": "Custom definition"},
        {"dimension": "Ethics approval: number", "definition": "Approval number"},
    ]

    result = await clinical_trial_comparison(
        "NCT123",
        str(paper),
        ".pdf",
        "openai",
        selected_dimensions=selected_dims,
        nct_extractor=lambda t: "NCT0000",
        trial_fetcher=lambda n: {"Design": {"sub": "val"}},
        pdf_parser=fake_pdf_parser,
        comparison_runner=fake_run,
    )
    assert calls == [dim["dimension"] for dim in selected_dims]
    assert definitions == [dim["definition"] for dim in selected_dims]
    assert isinstance(result, ComparisonResult)
