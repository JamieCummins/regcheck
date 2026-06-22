import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("DEEPSEEK_API_KEY", "test")
os.environ.setdefault("CLAUDE_API_KEY", "test")

import backend.services.comparisons as comparisons  # noqa: E402
import backend.services.llm as llm  # noqa: E402
from backend.services.comparisons import (  # noqa: E402
    ComparisonResult,
    clinical_trial_comparison,
    general_preregistration_comparison,
)


def _run_coro(coro):
    """Run a coroutine on a private loop WITHOUT nulling the global current loop.
    (asyncio.run() sets the current loop to None on exit, which breaks the legacy
    tests that still call asyncio.get_event_loop() — test_sharing/test_api — when
    this module runs before them.)"""
    import asyncio

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


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

    async def hget(self, key, field):
        return self.hashes.get(key, {}).get(field)

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


def test_run_comparison_degrades_on_unparseable_response(monkeypatch):
    import hashlib

    import numpy as np

    from backend.services.embeddings import EmbeddingCorpus

    def _corpus(prefix):
        return EmbeddingCorpus(
            segments=["a segment of text"],
            embeddings=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            chunk_ids=[f"{prefix}_0001"],
            norms=np.array([1.0], dtype=np.float32),
            metadata=[{}],
        )

    prereg, paper = "p", "x"
    corpus_cache = {
        f"prereg:{hashlib.sha256(prereg.encode()).hexdigest()}": _corpus("PREREG"),
        f"paper:{hashlib.sha256(paper.encode()).hexdigest()}": _corpus("PAPER"),
    }
    # Avoid the embeddings API for the query; make the model return non-JSON prose.
    monkeypatch.setattr(comparisons, "get_embedding", lambda text, model=None: np.array([1.0, 0.0, 0.0], dtype=np.float32))
    monkeypatch.setattr(comparisons, "_claude_chat", lambda **kw: "I considered the dimension but produced no JSON object.")

    result = comparisons.run_comparison(prereg, paper, "claude", "Sample size", corpus_cache=corpus_cache)

    # One degraded item — the whole report is NOT aborted by one bad response.
    assert len(result.items) == 1
    item = result.items[0]
    assert item.dimension == "Sample size"
    assert item.deviation_judgement == "Insufficient evidence"
    assert item.deviation_information  # explains the parse failure


def test_prebuild_query_embeddings_batches_into_one_call(monkeypatch):
    import numpy as np

    calls = []

    def fake_embed(segments, model):
        calls.append(list(segments))
        return np.zeros((len(segments), 4), dtype=np.float32)

    monkeypatch.setattr(comparisons, "openai_embed_segments", fake_embed)
    dims = [
        {"dimension": "Sample size", "definition": "n"},
        {"dimension": "Outcomes", "definition": ""},
        {"name": "Blinding", "definition": "masking"},
        "junk",                 # non-dict -> skipped
        {"dimension": "   "},   # blank name -> skipped
    ]

    cache = _run_coro(comparisons._prebuild_query_embeddings(dims, embedding_model="m"))

    # ONE batched embedding call containing exactly the valid augmented queries.
    assert len(calls) == 1
    assert calls[0] == ["Sample size. n", "Outcomes", "Blinding. masking"]
    # Keys must match what run_comparison computes for the same query.
    assert comparisons._query_embedding_key("Sample size. n") in cache
    assert comparisons._query_embedding_key("Blinding. masking") in cache
    assert len(cache) == 3


def test_prebuild_query_embeddings_degrades_to_empty_on_failure(monkeypatch):
    def boom(segments, model):
        raise RuntimeError("embedding API down")

    monkeypatch.setattr(comparisons, "openai_embed_segments", boom)
    # On failure it returns {} so each dimension falls back to its own embedding.
    cache = _run_coro(
        comparisons._prebuild_query_embeddings(
            [{"dimension": "X", "definition": "y"}], embedding_model="m"
        )
    )
    assert cache == {}


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
async def test_current_task_ttl_persists_when_report_has_no_expiry(monkeypatch):
    # ttl == -1 means the task hash is persisted (no expiry) → evidence persists
    # too, and the resolver must NOT mutate the task hash's expiry.
    monkeypatch.setenv("TASK_TTL_SECONDS", "1234")
    redis = FakeRedis(ttl_value=-1)

    ttl = await comparisons._current_task_ttl(redis, "task-1")

    assert ttl is None
    assert "task-1" not in redis.expiries


@pytest.mark.asyncio
async def test_current_task_ttl_inherits_remaining_seconds():
    redis = FakeRedis(ttl_value=4242)
    assert await comparisons._current_task_ttl(redis, "task-1") == 4242


@pytest.mark.asyncio
async def test_current_task_ttl_honors_retention():
    """Anonymous runs carry an explicit seconds retention so their evidence
    artifacts inherit the same ~7-day expiry; signed-in runs persist."""
    redis = FakeRedis(ttl_value=999)
    redis.hashes["t-persist"] = {"retention": "persist"}
    assert await comparisons._current_task_ttl(redis, "t-persist") is None

    week = str(7 * 24 * 60 * 60)
    redis.hashes["t-anon"] = {"retention": week}
    assert await comparisons._current_task_ttl(redis, "t-anon") == int(week)


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


def test_split_system_for_anthropic_separates_system_and_conversation():
    system, convo = comparisons._split_system_for_anthropic(
        [
            {"role": "system", "content": "You are RegCheck."},
            {"role": "user", "content": "Compare these documents."},
        ]
    )
    assert system == "You are RegCheck."
    assert convo == [{"role": "user", "content": "Compare these documents."}]


def test_claude_response_text_concatenates_text_blocks_only():
    class _Block:
        def __init__(self, text=None):
            if text is not None:
                self.text = text

    response = SimpleNamespace(content=[_Block("Hello "), _Block(), _Block("world")])
    assert comparisons._claude_response_text(response) == "Hello world"
    assert comparisons._claude_response_text(SimpleNamespace(content=None)) == ""


def test_claude_chat_sends_system_separately_and_returns_text(monkeypatch):
    captured = {}

    class _FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(text='{"dimension": "Sample size"}')]
            )

    monkeypatch.setattr(
        llm,
        "get_claude_client",
        lambda: SimpleNamespace(messages=_FakeMessages()),
    )

    out = llm._claude_chat(
        model="claude-opus-4-8",
        messages=[
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "user prompt"},
        ],
        max_tokens=4321,
    )

    assert out == '{"dimension": "Sample size"}'
    assert captured["model"] == "claude-opus-4-8"
    assert captured["system"] == "sys prompt"
    assert "temperature" not in captured  # Opus 4.8 rejects temperature
    assert captured["max_tokens"] == 4321
    assert captured["messages"] == [{"role": "user", "content": "user prompt"}]


def test_claude_chat_maps_auth_error_to_friendly_message(monkeypatch):
    class _AuthBoom(Exception):
        status_code = 401

    class _FakeMessages:
        def create(self, **kwargs):
            raise _AuthBoom("invalid x-api-key")

    monkeypatch.setattr(
        llm,
        "get_claude_client",
        lambda: SimpleNamespace(messages=_FakeMessages()),
    )

    with pytest.raises(RuntimeError, match="CLAUDE_API_KEY"):
        llm._claude_chat(
            model="claude-opus-4-8",
            messages=[{"role": "user", "content": "hi"}],
        )


@pytest.mark.parametrize("ext", [".pdf", ".PDF", ".docx", ".txt", ".html", ".htm"])
def test_validate_doc_ext_accepts_supported_types(ext):
    from backend.routes.comparisons import _validate_doc_ext

    # Should not raise for any document type the pipeline can read.
    _validate_doc_ext(ext, kind="paper")


@pytest.mark.parametrize("ext", [".doc", ".rtf", ".pages", "", ".pdf "])
def test_validate_doc_ext_rejects_unsupported_paper_types(ext):
    # An unsupported paper must be rejected at submit (HTTP 400) rather than
    # passing through to the worker, where read_file_as_pdf raises a cryptic
    # "Unsupported file type" that surfaces only as a worker error. This is
    # independent of multiple-experiments (the read happens before that branch).
    from fastapi import HTTPException

    from backend.routes.comparisons import _validate_doc_ext

    with pytest.raises(HTTPException) as exc_info:
        _validate_doc_ext(ext, kind="paper")
    assert exc_info.value.status_code == 400
    assert "paper file type" in exc_info.value.detail


def test_qwen_is_not_openai_family_and_uses_groq_endpoint():
    # Qwen is open-weight (served via Groq), so it must NOT route through the
    # hosted OpenAI client.
    assert "qwen" not in comparisons._OPENAI_CLIENTS
    assert comparisons._qwen_model() == "qwen/qwen3.6-27b"


@pytest.mark.asyncio
async def test_qwen_routes_through_groq_openai_with_requested_params(monkeypatch):
    # Qwen must use Groq's OpenAI-compatible endpoint (not hosted OpenAI), send
    # temperature=0.6 + reasoning_effort="default" + hidden reasoning, and return
    # text with any <think> block stripped.
    def _no_openai():
        raise AssertionError("qwen must not touch the hosted OpenAI client")

    monkeypatch.setattr(comparisons, "get_openai_client", _no_openai)

    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="<think>deliberating</think>INTRO ... EXPERIMENT 2 ... DISCUSSION"
                    )
                )
            ]
        )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    # _qwen_chat lives in llm and calls llm.get_groq_openai_client, so patch there.
    monkeypatch.setattr(llm, "get_groq_openai_client", lambda: fake_client)

    out = await comparisons.extract_experiment_specific_paper_text(
        "Full paper text spanning several experiments.",
        "2",
        client_choice="qwen",
    )

    assert "EXPERIMENT 2" in out
    assert "<think>" not in out and "deliberating" not in out  # stray reasoning stripped
    assert captured["model"] == "qwen/qwen3.6-27b"
    assert captured["temperature"] == 0.6
    assert captured["reasoning_effort"] == "none"  # QWEN_REASONING_EFFORT default
    # NB: no reasoning_format — "hidden"/"parsed" route the answer into the reasoning
    # channel on this Groq model and return empty content (verified live).
    assert "extra_body" not in captured
    # Isolation call returns free text, so it must NOT force JSON-object mode.
    assert "response_format" not in captured


def test_qwen_uses_json_object_mode_only_for_comparison(monkeypatch):
    captured = {}

    def fake_create(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"deviation_judgement":"no"}'))]
        )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    monkeypatch.setattr(llm, "get_groq_openai_client", lambda: fake_client)

    # Comparison call → JSON-object structured output (strict json_schema is rejected
    # by this Groq model, so json_object is the reliable choice).
    llm._qwen_chat([{"role": "user", "content": "x"}], use_json_mode=True)
    assert captured["response_format"] == {"type": "json_object"}

    # Isolation call (free text) → no response_format.
    llm._qwen_chat([{"role": "user", "content": "x"}])
    assert "response_format" not in captured


@pytest.mark.asyncio
async def test_multi_study_isolation_failure_is_surfaced(tmp_path, monkeypatch):
    """Multi-study isolation is best-effort, but a failure must be VISIBLE: the run
    still completes on the full paper AND sets a persistent `multi_study_isolation`
    flag so a silently-degraded run isn't mistaken for a study-specific one."""
    import numpy as np
    import backend.services.embeddings as emb

    monkeypatch.setattr(
        emb, "openai_embed_segments",
        lambda s, model=None, **k: np.ones((max(1, len(list(s))), 8), dtype=np.float32),
    )

    async def boom(*a, **k):
        raise RuntimeError("model context exceeded")

    monkeypatch.setattr(comparisons, "extract_experiment_specific_paper_text", boom)

    def fake_runner(p, pa, c, d, **kw):
        return comparisons.ComparisonResult(items=[comparisons.ComparisonItem(
            dimension=d, paper_content_quotes="", paper_content_summary="x",
            registration_content_quotes="", registration_content_summary="y",
            deviation_judgement="no", deviation_information="ok")])

    class _R:
        def __init__(s): s.h = {}
        async def hset(s, k, mapping=None, **kw): s.h.setdefault(k, {}).update(mapping or kw)
        async def hgetall(s, k): return s.h.get(k, {})
        async def expire(s, *a, **k): return True
        async def ttl(s, *a, **k): return 3600
        async def persist(s, *a, **k): return True
        async def set(s, k, v, ex=None): s.h[k] = v
        async def get(s, k): return s.h.get(k)
        async def exists(s, k): return 1 if k in s.h else 0
        async def zadd(s, *a, **k): return 1

    fitz = pytest.importorskip("fitz")
    prereg = tmp_path / "p.txt"; prereg.write_text("Preregister Study 1.")
    paper = tmp_path / "paper.pdf"
    doc = fitz.open(); pg = doc.new_page(); pg.insert_text((72, 72), "Intro. Study 1 results. Discussion."); doc.save(paper); doc.close()

    redis = _R()
    res = await comparisons.general_preregistration_comparison(
        str(prereg), ".txt", str(paper), ".pdf", "claude", "pymupdf",
        task_id="t1", redis_client=redis,
        selected_dimensions=[{"dimension": "Hypotheses", "definition": "the hypotheses"}],
        comparison_runner=fake_runner, multiple_experiments="yes",
        experiment_number="1", experiment_text="Study 1")

    st = redis.h.get("t1", {})
    assert st.get("state") == "SUCCESS"                       # still completes
    assert st.get("multi_study_isolation") == "failed"        # ...but flagged
    assert "model context exceeded" in (st.get("multi_study_isolation_error") or "")
    assert len(res.items) == 1
