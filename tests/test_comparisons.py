import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("GROQ_API_KEY", "test")
os.environ.setdefault("DEEPSEEK_API_KEY", "test")

import backend.services.comparisons as comparisons  # noqa: E402
from backend.services.comparisons import (  # noqa: E402
    ComparisonResult,
    clinical_trial_comparison,
    general_preregistration_comparison,
)


class FakeRedis:
    def __init__(self, ttl_value=3600):
        self.values = {}
        self.hashes = {}
        self.expiries = {}
        self.ttl_value = ttl_value

    async def set(self, key, value, ex=None):
        self.values[key] = value
        self.expiries[key] = ex

    async def get(self, key):
        return self.values.get(key)

    async def exists(self, key):
        return 1 if key in self.values else 0

    async def hset(self, key, mapping):
        self.hashes.setdefault(key, {}).update(mapping)

    async def ttl(self, key):
        return self.ttl_value

    async def expire(self, key, seconds):
        self.expiries[key] = seconds
        self.ttl_value = seconds
        return True

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


def test_groq_auth_error_is_not_retried(monkeypatch):
    class AuthError(Exception):
        status_code = 401

    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        raise AuthError("invalid_api_key")

    monkeypatch.setattr(
        comparisons,
        "groq_client",
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))),
    )

    with pytest.raises(RuntimeError, match="Groq authentication failed"):
        comparisons._groq_chat_completion(
            model="llama-test",
            messages=[{"role": "user", "content": "x"}],
            use_json_mode=True,
        )

    assert len(calls) == 1


def test_groq_response_format_error_retries_without_json_mode(monkeypatch):
    calls = []
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))]
    )

    def fake_create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("response_format is not supported")
        return response

    monkeypatch.setattr(
        comparisons,
        "groq_client",
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))),
    )

    result = comparisons._groq_chat_completion(
        model="llama-test",
        messages=[{"role": "user", "content": "x"}],
        use_json_mode=True,
    )

    assert result is response
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]


@pytest.mark.asyncio
async def test_store_evidence_manifest_sets_ready_diagnostics():
    redis = FakeRedis()

    manifest = await comparisons._store_evidence_manifest(
        redis_client=redis,
        task_id="task-1",
        comparison_type="general_preregistration",
        source_payloads=[
            {
                "source": {"id": "registration", "label": "Registration", "kind": "text"},
                "raw_bytes": b"raw",
                "raw_content_type": "text/plain",
                "render_data": {"kind": "text", "text": "raw"},
                "chunks": {"PREREG_0001": {"id": "PREREG_0001"}},
            }
        ],
        ttl_seconds=3600,
    )

    fields = redis.hashes["task-1"]
    assert fields["evidence_status"] == "ready"
    assert fields["evidence_error"] == ""
    assert fields["evidence_storage"] == "redis"
    assert fields["evidence_source_count"] == 1
    assert fields["evidence_chunk_count"] == 1
    assert fields["evidence_artifact_count"] == 2
    assert fields["evidence_artifact_bytes"] > 0
    assert manifest["sources"]["registration"]["_artifacts"]["raw"]["storage"] == "redis"


@pytest.mark.asyncio
async def test_evidence_success_fields_returns_error_when_artifact_is_missing():
    redis = FakeRedis()
    manifest = await comparisons._store_evidence_manifest(
        redis_client=redis,
        task_id="task-1",
        comparison_type="general_preregistration",
        source_payloads=[
            {
                "source": {"id": "registration", "label": "Registration", "kind": "text"},
                "raw_bytes": b"raw",
                "raw_content_type": "text/plain",
                "render_data": {"kind": "text", "text": "raw"},
            }
        ],
        ttl_seconds=3600,
    )
    del redis.values[manifest["sources"]["registration"]["_artifacts"]["raw"]["key"]]

    fields = await comparisons._evidence_success_fields(redis, "task-1", manifest)

    assert fields["evidence_status"] == "error"
    assert "Redis artifact key" in fields["evidence_error"]


@pytest.mark.asyncio
async def test_current_task_ttl_sets_task_expiry_when_missing(monkeypatch):
    monkeypatch.setenv("TASK_TTL_SECONDS", "1234")
    redis = FakeRedis(ttl_value=-1)

    ttl = await comparisons._current_task_ttl(redis, "task-1")

    assert ttl == 1234
    assert redis.expiries["task-1"] == 1234


@pytest.mark.parametrize(
    "defaults_name",
    ["CLINICAL_DEFAULT_DIMENSIONS", "PRECLINICAL_DEFAULT_DIMENSIONS"],
)
def test_default_dimension_sets_all_carry_definitions(defaults_name):
    """The canonical default sets are the single source of default dimensions;
    every entry must ship a non-empty definition (guards the old mismatch
    where 'Design: Planned sample size' silently resolved to an empty one)."""
    defaults = getattr(comparisons, defaults_name)
    assert defaults
    for item in defaults:
        assert item["dimension"].strip()
        assert item["definition"].strip(), f"missing definition for {item['dimension']!r}"


def test_clinical_defaults_are_the_curated_set():
    names = [item["dimension"] for item in comparisons.CLINICAL_DEFAULT_DIMENSIONS]
    assert names == [
        "Eligibility – inclusion criteria",
        "Eligibility – exclusion criteria",
        "Intervention/treatment and control/placebo",
        "Ethical approval – number",
        "Ethical approval – committee",
        "Ethical approval – date",
        "Sample size",
        "Date recruitment started",
        "Outcomes – primary",
        "Outcomes – secondary",
        "Method of randomisation and allocation",
    ]


def test_preclinical_defaults_are_the_curated_set():
    names = [item["dimension"] for item in comparisons.PRECLINICAL_DEFAULT_DIMENSIONS]
    assert names == [
        "Study type (exploratory vs. confirmatory)",
        "Total number of animals",
        "Number of animals per group",
        "Intervention and control",
        "Measures to reduce bias",
        "Primary outcomes",
        "Secondary outcomes",
        "Statistical analyses",
        "Hypotheses",
    ]


def test_resolve_dimensions_user_values_win_verbatim():
    """User-specified dimensions (UI or API) are used exactly as given —
    including empty definitions. No name-based fallback may fire."""
    user = [
        {"dimension": "Eligibility – inclusion criteria", "definition": ""},
        {"name": "Custom dimension", "definition": "  my definition  "},
        {"dimension": "   "},
        "not-a-dict",
    ]
    resolved = comparisons._resolve_dimensions(user, comparisons.CLINICAL_DEFAULT_DIMENSIONS)
    assert resolved == [
        {"dimension": "Eligibility – inclusion criteria", "definition": ""},
        {"dimension": "Custom dimension", "definition": "my definition"},
    ]


def test_resolve_dimensions_falls_back_to_explicit_defaults_only():
    resolved = comparisons._resolve_dimensions(None, comparisons.CLINICAL_DEFAULT_DIMENSIONS)
    assert resolved == comparisons.CLINICAL_DEFAULT_DIMENSIONS
    # Copies, not aliases: mutating a resolved item must not corrupt the canon.
    resolved[0]["definition"] = "mutated"
    assert comparisons.CLINICAL_DEFAULT_DIMENSIONS[0]["definition"] != "mutated"
    # And with no defaults supplied (general flow), empty selection stays empty.
    assert comparisons._resolve_dimensions(None) == []
    assert comparisons._resolve_dimensions([]) == []


def test_wizard_clinical_preset_matches_backend(tmp_path):
    """The wizard's 'Clinical / Medical' preset (static/js/wizard.js) must stay
    identical to CLINICAL_DEFAULT_DIMENSIONS — guards against the two clinical
    sources drifting apart."""
    import json
    import re
    from pathlib import Path

    wizard = Path(__file__).resolve().parent.parent / "static" / "js" / "wizard.js"
    src = wizard.read_text(encoding="utf-8")
    m = re.search(
        r'key:\s*"clinical",.*?dims:\s*(\[.*?\])\n\s{12}\},',
        src,
        re.DOTALL,
    )
    assert m, "clinical preset block not found in wizard.js"
    # Convert the JS array literal (unquoted keys) into JSON.
    js_array = m.group(1)
    js_array = re.sub(r'(\{|,)\s*(name|definition):', r'\1"\2":', js_array)
    preset = json.loads(js_array)
    backend = [
        {"name": d["dimension"], "definition": d["definition"]}
        for d in comparisons.CLINICAL_DEFAULT_DIMENSIONS
    ]
    assert preset == backend


def test_wizard_preclinical_preset_matches_backend():
    """The wizard's 'Preclinical / Animal' preset must stay identical to
    PRECLINICAL_DEFAULT_DIMENSIONS (the two are generated from one source)."""
    import json
    import re
    from pathlib import Path

    wizard = Path(__file__).resolve().parent.parent / "static" / "js" / "wizard.js"
    src = wizard.read_text(encoding="utf-8")
    m = re.search(r'key:\s*"preclinical",.*?dims:\s*(\[.*?\])\n\s{12}\}', src, re.DOTALL)
    assert m, "preclinical preset block not found in wizard.js"
    js_array = re.sub(r'(\{|,)\s*(name|definition):', r'\1"\2":', m.group(1))
    preset = json.loads(js_array)
    backend = [
        {"name": d["dimension"], "definition": d["definition"]}
        for d in comparisons.PRECLINICAL_DEFAULT_DIMENSIONS
    ]
    assert preset == backend
