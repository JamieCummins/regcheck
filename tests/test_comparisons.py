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
