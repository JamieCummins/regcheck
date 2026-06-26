import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("DEEPSEEK_API_KEY", "test")
os.environ.setdefault("CLAUDE_API_KEY", "test")

import backend.services.comparisons as comparisons  # noqa: E402
import backend.services.llm as llm  # noqa: E402
from backend.services.comparisons import (  # noqa: E402
    ComparisonItem,
    ComparisonResult,
    _aggregate_dimension_votes,
    _normalize_verdict,
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
    # Avoid the embeddings API for the query; make every judgement attempt return
    # non-JSON prose (so both attempts fail to parse and the dimension degrades).
    monkeypatch.setattr(comparisons, "get_embedding", lambda text, model=None: np.array([1.0, 0.0, 0.0], dtype=np.float32))
    monkeypatch.setattr(comparisons, "_dispatch_judgement", lambda *a, **kw: "I considered the dimension but produced no JSON object.")

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


def _dim_defs(rows):
    """dimension/definition pairs only — ignore the retrieval 'keywords' field."""
    return [{"dimension": r["dimension"], "definition": r["definition"]} for r in rows]


def test_resolve_dimensions_user_values_win_verbatim():
    """User-specified dimensions (UI or API) are used exactly as given —
    including empty definitions. No name-based fallback may fire on definitions."""
    user = [
        {"dimension": "Eligibility – inclusion criteria", "definition": ""},
        {"name": "Custom dimension", "definition": "  my definition  "},
        {"dimension": "   "},
        "not-a-dict",
    ]
    resolved = comparisons._resolve_dimensions(user, comparisons.CLINICAL_DEFAULT_DIMENSIONS)
    assert _dim_defs(resolved) == [
        {"dimension": "Eligibility – inclusion criteria", "definition": ""},
        {"dimension": "Custom dimension", "definition": "my definition"},
    ]
    # Retrieval keywords are attached by name: a known dimension gets them, a custom
    # one gets none. (Keywords steer retrieval only; they never change definitions.)
    assert resolved[0]["keywords"]  # matched a known clinical dimension
    assert resolved[1]["keywords"] == []  # unknown custom dimension


def test_resolve_dimensions_falls_back_to_explicit_defaults_only():
    resolved = comparisons._resolve_dimensions(None, comparisons.CLINICAL_DEFAULT_DIMENSIONS)
    assert _dim_defs(resolved) == _dim_defs(comparisons.CLINICAL_DEFAULT_DIMENSIONS)
    # Copies, not aliases: mutating a resolved item must not corrupt the canon.
    resolved[0]["definition"] = "mutated"
    assert comparisons.CLINICAL_DEFAULT_DIMENSIONS[0]["definition"] != "mutated"
    # And with no defaults supplied (general flow), empty selection stays empty.
    assert comparisons._resolve_dimensions(None) == []
    assert comparisons._resolve_dimensions([]) == []


def test_discipline_presets_are_single_sourced():
    """The discipline presets now live in one backend data file (loaded into the
    registry, injected into the wizard, and offered to the CLI). All four load."""
    from backend.services import dimensions as dm

    assert dm.discipline_keys() == ["psychology", "clinical", "economics", "preclinical"]
    assert len(dm.get_discipline_dimensions("psychology")) == 9
    assert len(dm.get_discipline_dimensions("economics")) == 10
    assert dm.get_discipline_dimensions("nope") is None
    # UI payload uses the field name the wizard expects.
    ui = {s["key"]: s for s in dm.discipline_sets_for_ui()}
    assert ui["psychology"]["dims"][0]["name"] == "Hypotheses"


def test_clinical_preset_matches_backend_defaults():
    """The 'clinical' preset (single source) must stay identical to
    CLINICAL_DEFAULT_DIMENSIONS — guards against the comparison default drifting
    from the discipline preset."""
    from backend.services import dimensions as dm

    backend = [
        {"dimension": d["dimension"], "definition": d["definition"]}
        for d in dm.CLINICAL_DEFAULT_DIMENSIONS
    ]
    assert _dim_defs(dm.get_discipline_dimensions("clinical")) == backend


def test_preclinical_preset_matches_backend_defaults():
    """The 'preclinical' preset must stay identical to PRECLINICAL_DEFAULT_DIMENSIONS."""
    from backend.services import dimensions as dm

    backend = [
        {"dimension": d["dimension"], "definition": d["definition"]}
        for d in dm.PRECLINICAL_DEFAULT_DIMENSIONS
    ]
    assert _dim_defs(dm.get_discipline_dimensions("preclinical")) == backend


def test_hosted_route_rejects_unreachable_providers():
    """The hosted app/API validates the model provider (parity with parser
    validation). gpustack is reachable only from the Bern network, so the hosted
    worker can't run it — it's rejected here and offered on the CLI only."""
    from fastapi import HTTPException

    from backend.routes import comparisons as routes

    for ok in ("openai", "deepseek", "qwen", "claude", "OpenAI", " claude "):
        assert routes._normalize_client(ok) in {"openai", "deepseek", "qwen", "claude"}
    for bad in ("gpustack", "bogus", ""):
        with pytest.raises(HTTPException):
            routes._normalize_client(bad)


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


def test_gpustack_is_not_openai_family_and_uses_campus_endpoint():
    # gpt-oss-120b is served on Uni Bern's GPUStack, so it must NOT route through
    # the hosted OpenAI client, and its base URL is the campus endpoint.
    assert "gpustack" not in comparisons._OPENAI_CLIENTS
    assert comparisons._gpustack_model() == "gpt-oss-120b"
    assert llm.DEFAULT_GPUSTACK_BASE_URL == "https://gpustack.unibe.ch/v1"


@pytest.mark.asyncio
async def test_gpustack_routes_through_gpustack_endpoint(monkeypatch):
    # GPUStack must use its own OpenAI-compatible client (never hosted OpenAI), send
    # gpt-oss sampling defaults, and return text with any <think> block stripped.
    def _no_openai():
        raise AssertionError("gpustack must not touch the hosted OpenAI client")

    monkeypatch.setattr(comparisons, "get_openai_client", _no_openai)

    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="<think>weighing it</think>INTRO ... EXPERIMENT 2 ... DISCUSSION"
                    )
                )
            ]
        )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    # _gpustack_chat lives in llm and calls llm.get_gpustack_client, so patch there.
    monkeypatch.setattr(llm, "get_gpustack_client", lambda: fake_client)

    out = await comparisons.extract_experiment_specific_paper_text(
        "Full paper text spanning several experiments.",
        "2",
        client_choice="gpustack",
    )

    assert "EXPERIMENT 2" in out
    assert "<think>" not in out and "weighing it" not in out  # stray reasoning stripped
    assert captured["model"] == "gpt-oss-120b"
    assert captured["temperature"] == 1.0
    assert captured["top_p"] == 1.0
    # Isolation call returns free text, so it must NOT force JSON-object mode.
    assert "response_format" not in captured


def test_gpustack_json_mode_uses_prompt_not_response_format(monkeypatch):
    # gpt-oss-120b's json_object guided decoding is broken on GPUStack (it leaks
    # harmony tokens + invalid JSON), so the comparison call must NOT send
    # response_format; it appends a JSON-only instruction and relies on the prompt
    # + extraction instead.
    captured = {}

    def fake_create(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"deviation_judgement":"no"}'))]
        )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    monkeypatch.setattr(llm, "get_gpustack_client", lambda: fake_client)

    # Comparison call → no response_format; a JSON-only system message is appended.
    llm._gpustack_chat([{"role": "user", "content": "x"}], use_json_mode=True)
    assert "response_format" not in captured
    assert any(m["role"] == "system" and "JSON" in m["content"] for m in captured["messages"])

    # Isolation call (free text) → no response_format and no JSON nudge.
    llm._gpustack_chat([{"role": "user", "content": "x"}])
    assert "response_format" not in captured
    assert all("JSON" not in (m.get("content") or "") for m in captured["messages"])


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


def test_compute_top_k_scales_for_short_corpora():
    assert comparisons._compute_top_k(16) == 8    # ceil(16*0.45)=8 beats min_k=6
    assert comparisons._compute_top_k(100) == 10  # ~10% rate for large corpora
    assert comparisons._compute_top_k(200) == 20  # capped at max_k
    assert comparisons._compute_top_k(0) == 0
    assert comparisons._compute_top_k(3) == 3     # never exceeds total


def test_promote_keyword_hits_surfaces_out_of_topk_chunk():
    base = [("PAPER_0001", "introduction text", 0.60), ("PAPER_0002", "more text", 0.50)]
    extra = [
        ("PAPER_0011", "Eight participants were excluded (attention check), N = 192", 0.42),
        ("PAPER_0020", "wholly unrelated content", 0.05),
    ]
    all_rows = base + extra
    out = comparisons._promote_keyword_hits(base, all_rows, ["excluded", "attention check"])
    ids = [c[0] for c in out]
    assert "PAPER_0011" in ids       # promoted though it sat outside the base top-k
    assert "PAPER_0020" not in ids   # below the similarity floor -> not promoted
    # No keywords -> selection unchanged.
    assert comparisons._promote_keyword_hits(base, all_rows, []) == base


def test_keyword_index_resolves_known_dimensions():
    from backend.services import dimensions as dm

    assert any("attention check" in k for k in dm.keywords_for_dimension("Inclusion and exclusion criteria"))
    assert "n =" in [k.lower() for k in dm.keywords_for_dimension("Sample size")]
    assert dm.keywords_for_dimension("Totally novel custom dimension") == []
    assert dm.keywords_for_dimension("Sample size", ["custom-only"]) == ["custom-only"]


def _vote_item(verdict, rationale="", paper_summary="", reg_summary=""):
    return ComparisonItem(
        dimension="Sample size",
        deviation_judgement=verdict,
        deviation_information=rationale,
        paper_content_summary=paper_summary,
        registration_content_summary=reg_summary,
    )


def test_normalize_verdict_maps_label_variants():
    assert _normalize_verdict("yes") == "yes"
    assert _normalize_verdict("Deviation") == "yes"
    assert _normalize_verdict("inconsistent") == "yes"
    assert _normalize_verdict("no") == "no"
    assert _normalize_verdict("Consistent") == "no"
    assert _normalize_verdict("missing") == "missing"
    assert _normalize_verdict("Insufficient evidence") == "missing"  # degraded label
    assert _normalize_verdict("") == "missing"
    assert _normalize_verdict(None) == "missing"


def test_aggregate_votes_takes_plurality_and_appends_consensus_note():
    items = [_vote_item("yes", f"r{i}") for i in range(6)] + [_vote_item("no") for _ in range(2)]
    agg = _aggregate_dimension_votes(items, 8)
    assert _normalize_verdict(agg.deviation_judgement) == "yes"
    assert "6/8" in agg.deviation_information
    assert "6 deviation, 2 consistent" in agg.deviation_information
    assert "Consensus verdict from 8 of 8 parsed judgements" in agg.deviation_information


def test_aggregate_votes_excludes_unparseable_from_tally():
    # Only 6 of 8 voters produced a parseable judgement; the 2 failures are non-votes.
    items = [_vote_item("yes", f"r{i}") for i in range(4)] + [_vote_item("no") for _ in range(2)]
    agg = _aggregate_dimension_votes(items, 8)
    assert _normalize_verdict(agg.deviation_judgement) == "yes"
    assert "6 of 8 parsed judgements" in agg.deviation_information
    assert "4/6" in agg.deviation_information                       # tally is over parsed, not attempted
    assert "2 unparseable replies excluded" in agg.deviation_information
    # singular grammar
    agg1 = _aggregate_dimension_votes([_vote_item("yes") for _ in range(7)], 8)
    assert "1 unparseable reply excluded" in agg1.deviation_information


def test_judge_dimension_once_returns_none_on_unparseable(monkeypatch):
    # An unparseable reply (e.g. an unescaped quote from Claude) is a NON-VOTE: after a
    # retry it returns None rather than a spurious 'Insufficient evidence' verdict.
    calls = {"n": 0}

    def _bad(*_a, **_k):
        calls["n"] += 1
        return '{"dimension": "X", "deviation_judgement": "yes" "oops": unescaped}'

    monkeypatch.setattr(comparisons, "_dispatch_judgement", _bad)
    out = comparisons._judge_dimension_once(
        [{"role": "user", "content": "x"}],
        client_choice="claude",
        dimension_query="X",
        paper_top=["[PAPER_0001] q"],
        prereg_top=["[PREREG_0001] q"],
        reasoning_effort=None,
    )
    assert out is None
    assert calls["n"] == 2  # one retry before giving up

    # A valid reply on the retry is salvaged (returns a real item, not None).
    seq = iter(["garbage{", '{"dimension": "X", "deviation_judgement": "no"}'])
    monkeypatch.setattr(comparisons, "_dispatch_judgement", lambda *_a, **_k: next(seq))
    out2 = comparisons._judge_dimension_once(
        [{"role": "user", "content": "x"}],
        client_choice="claude",
        dimension_query="X",
        paper_top=["[PAPER_0001] q"],
        prereg_top=["[PREREG_0001] q"],
        reasoning_effort=None,
    )
    assert out2 is not None and _normalize_verdict(out2.deviation_judgement) == "no"


def test_aggregate_votes_tiebreak_is_recall_biased():
    # 4 yes / 4 no -> deviation (yes > no); 0 yes / 4 no / 4 missing -> no (no > missing).
    assert _normalize_verdict(_aggregate_dimension_votes(
        [_vote_item("yes") for _ in range(4)] + [_vote_item("no") for _ in range(4)], 8
    ).deviation_judgement) == "yes"
    assert _normalize_verdict(_aggregate_dimension_votes(
        [_vote_item("no") for _ in range(4)] + [_vote_item("missing") for _ in range(4)], 8
    ).deviation_judgement) == "no"


def test_aggregate_votes_canonical_is_most_grounded_winner():
    # Among the winning ('yes') runs, the rationale citing the most distinct evidence
    # IDs is carried into the report (quotes/summaries/rationale stay self-consistent).
    items = [
        _vote_item("yes", "thin", paper_summary="[PAPER_0001]"),
        _vote_item("yes", "grounded", paper_summary="[PAPER_0007]", reg_summary="[PREREG_0003] [PREREG_0009]"),
        _vote_item("no"),
    ]
    agg = _aggregate_dimension_votes(items, 3)
    assert _normalize_verdict(agg.deviation_judgement) == "yes"
    assert agg.deviation_information.startswith("grounded")


def test_aggregate_votes_missing_plurality():
    items = [_vote_item("missing") for _ in range(5)] + [_vote_item("yes") for _ in range(3)]
    agg = _aggregate_dimension_votes(items, 8)
    assert _normalize_verdict(agg.deviation_judgement) == "missing"
    assert "insufficient evidence — 5/8" in agg.deviation_information


@pytest.mark.asyncio
async def test_strand_consensus_runs_independent_voters(tmp_path):
    # num_voters>1 in the general flow => N independent strands, each with its OWN
    # append-previous chain, aggregated per dimension (not N coupled voters per dim).
    import threading

    prereg = tmp_path / "prereg.txt"
    prereg.write_text("prereg")
    paper = tmp_path / "paper.pdf"
    paper.write_text("paper")

    async def fake_pdf_parser(path: str) -> str:
        return '<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body>paper body</body></text></TEI>'

    calls = []
    lock = threading.Lock()

    def fake_run(prereg_in, paper_secs, client, dim, *, num_voters=1,
                 previous_dimension_responses=None, **kw):
        with lock:
            calls.append((dim, num_voters, len(previous_dimension_responses or [])))
        verdict = "yes" if dim == "DimA" else "no"
        return ComparisonResult(items=[ComparisonItem(
            dimension=dim, deviation_judgement=verdict,
            deviation_information="rationale", paper_content_summary="[PAPER_0001]")])

    res = await general_preregistration_comparison(
        str(prereg), ".txt", str(paper), ".pdf", "claude", "grobid",
        selected_dimensions=[{"dimension": "DimA", "definition": ""},
                             {"dimension": "DimB", "definition": ""}],
        append_previous_output=True,
        pdf_parser=fake_pdf_parser,
        comparison_runner=fake_run,
        num_voters=3,
    )
    # 3 strands x 2 dimensions = 6 single-judge calls (NOT 2 calls of num_voters=3)
    assert len(calls) == 6
    assert all(nv == 1 for _, nv, _ in calls)
    # DimA is first in every strand (no prior); DimB sees exactly its own strand's 1 prior
    assert sorted(prev for dim, _, prev in calls if dim == "DimA") == [0, 0, 0]
    assert sorted(prev for dim, _, prev in calls if dim == "DimB") == [1, 1, 1]
    # two aggregated dimensions, each a consensus over the 3 strands
    assert [it.dimension for it in res.items] == ["DimA", "DimB"]
    assert _normalize_verdict(res.items[0].deviation_judgement) == "yes"
    assert _normalize_verdict(res.items[1].deviation_judgement) == "no"
    assert "from 3 of 3 parsed judgements" in res.items[0].deviation_information
