"""Evaluate Registration Quality: single-document completeness assessment.

Same IDEA skeleton as the comparison flow — ingest, retrieve per dimension,
judge, report — but with ONE document (the registration) and a different
question at the judgement step: not "does the paper deviate from the plan?"
but "is the plan itself specified completely and unambiguously enough that a
deviation COULD later be detected?".

Reuses the comparison module's machinery (embeddings/retrieval, hybrid keyword
promotion, neighbour expansion, quotes-as-IDs, provider dispatch, the targeted
verification loop, evidence manifests) via module-attribute access so test
monkeypatches on ``comparisons`` apply here too. Results are stored as
``ComparisonItem``s with the paper-side fields empty: ``deviation_judgement``
carries the quality verdict ('complete' / 'partial' / 'absent') and
``deviation_information`` the rationale, so report persistence, the viewer, CSV
export, and regeneration all work unchanged.

PROMPT WORDING: the doctrine below is a DRAFT pending Jamie's sign-off.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ValidationError

from . import comparisons as _c
from .comparisons import ComparisonItem, ComparisonResult
from .dimensions import _resolve_dimensions, registration_quality_dimensions
from .documents import decode_bytes, extract_text_from_docx, extract_text_from_html, read_file
from .pdf_parsers import extract_pdf_text

logger = logging.getLogger(__name__)

COMPARISON_TYPE = "registration_quality"


class QualityItem(BaseModel):
    """Schema-constrained judgement shape for the quality flow (OpenAI parse())."""

    dimension: str = ""
    registration_content_quotes: str = ""
    completeness_judgement: str = ""
    completeness_rationale: str = ""
    unlocated_in_registration: str = ""


# Anthropic tool schema mirroring QualityItem (Claude has no response_format).
_QUALITY_TOOL: dict[str, Any] = {
    "name": "record_quality_assessment",
    "description": "Record the structured registration-quality assessment for this dimension.",
    "input_schema": {
        "type": "object",
        "properties": {
            "dimension": {"type": "string"},
            "registration_content_quotes": {
                "type": "string",
                "description": "Evidence IDs of the registration excerpts that ground the assessment (e.g. [PREREG_0003] [PREREG_0007]), space-separated, most relevant first.",
            },
            "completeness_judgement": {
                "type": "string",
                "description": "'complete' if every element this dimension calls for is stated precisely and unambiguously; 'partial' if the dimension is addressed but at least one element is missing, vague, ambiguous, or deferred to a document not provided; 'absent' if the excerpts contain nothing usable on this dimension. Prefer 'partial' over 'absent' whenever any usable material addresses the dimension.",
            },
            "completeness_rationale": {"type": "string"},
            "unlocated_in_registration": {
                "type": "string",
                "description": "Elements looked for but not found in the excerpts (semicolon-separated, with synonyms in parentheses); empty string if none.",
            },
        },
        "required": [
            "dimension",
            "registration_content_quotes",
            "completeness_judgement",
            "completeness_rationale",
            "unlocated_in_registration",
        ],
    },
}


def normalize_quality_verdict(value: str | None) -> str:
    """Map any quality verdict label to one of {'complete','partial','absent'}."""
    s = (value or "").strip().lower()
    if s.startswith("complete") or "fully" in s:
        return "complete"
    if s.startswith("partial"):
        return "partial"
    return "absent"


def _quality_static_doctrine() -> str:
    """The constant prompt prefix (cache-friendly: byte-identical across all
    dimensions of a run). DRAFT wording — pending sign-off."""
    if (os.environ.get("QUOTES_AS_IDS") or "1").strip().lower() in {"1", "true", "yes", "on"}:
        quotes_bullet = (
            "- For 'registration_content_quotes', output ONLY the evidence IDs of the excerpts that ground your assessment (e.g., [PREREG_0001] [PREREG_0007]), space-separated, most relevant first. Do NOT copy quote text.\n"
        )
    else:
        quotes_bullet = (
            "- For 'registration_content_quotes', include direct quotes from the provided excerpts, and keep the evidence IDs (e.g., [PREREG_0001]) in the text. Join multiple quotes with two newlines (\\n\\n). Do NOT return an array.\n"
        )
    return (
        "Critically evaluate the completeness and specificity of the following study registration on the below-specified quality dimension.\n\n"
        "You have two goals. First, identify the excerpts from the registration that are relevant to the specified quality dimension. "
        "Second, make a three-way judgement on how completely the registration specifies this dimension: fully specified, partially specified, or not specified.\n\n"
        "You are assessing the registration ON ITS OWN — there is no paper to compare against. The standard is VERIFIABILITY: a dimension is fully specified only when the registration commits to it precisely and unambiguously enough that an independent reader could later detect any deviation from the stated plan. A commitment that a wide range of different study decisions would all satisfy (e.g. 'we will collect a sufficient sample', 'appropriate statistical tests will be used') is NOT a full specification, however reasonable it sounds.\n\n"
        "Assess specification, not scientific merit: a precise plan you consider methodologically weak is still fully specified, and an admirable but vague plan is not. Judge only against the dimension as defined below — do not penalise the registration for omitting content this dimension does not call for.\n\n"
        "Conditional or branching plans (e.g. 'if assumption X fails we will instead do Y') count as specified when each branch and its trigger are stated precisely. A commitment delegated to a document that is not part of the provided text (e.g. 'as described in the attached analysis plan') is NOT verifiable here: treat that element as unverified and say so in the rationale.\n\n"
        "Your output must be a single JSON object (no arrays unless specified, no surrounding text, no code fences) with the following fields: "
        "'dimension', 'registration_content_quotes', 'completeness_judgement', 'completeness_rationale', and 'unlocated_in_registration'. Each field MUST be a string.\n"
        f"{quotes_bullet}"
        "- For 'completeness_rationale', name each element this dimension calls for and state whether the registration specifies it precisely, specifies it vaguely, or omits it — citing the evidence IDs you relied upon. Keep it concise prose.\n"
        "- 'unlocated_in_registration': elements you looked for but could not find in the provided excerpts, as a short semicolon-separated list, naming each element together with any synonyms or abbreviations it might appear under (in parentheses). Use an empty string when there are none.\n"
        "- 'completeness_judgement' must be exactly 'complete', 'partial', or 'absent':\n"
        "  1. 'complete' — every element the dimension (as defined) calls for is stated precisely and unambiguously; an independent reader could verify compliance with each.\n"
        "  2. 'partial' — the registration addresses the dimension, but at least one called-for element is missing, vague, ambiguous, or deferred to a document not provided.\n"
        "  3. 'absent' — the provided excerpts contain nothing usable on this dimension.\n"
        "Prefer 'partial' over 'absent' whenever ANY usable material addresses the dimension.\n"
        "You see only retrieved excerpts, never the full document. If an element does not appear in the provided excerpts, phrase this in 'completeness_rationale' as 'not found in the provided excerpts' — never assert it is absent from the document — and list the element in 'unlocated_in_registration'. Then still return your best verdict on the evidence you have: RegCheck will run a targeted search of the full document for every element you list and re-invoke you with anything it finds, so do not soften a verdict because you suspect a retrieval gap — flag the element instead.\n"
    )


_QUALITY_SYSTEM_MESSAGE = (
    "You are RegCheck, a large language model which excels in auditing the quality of study "
    "registrations. You assess how completely and unambiguously a registration specifies each "
    "aspect of the planned study, with a very critical eye for vagueness that would let a later "
    "deviation go undetected."
)

_QUALITY_FIELD_ALIASES = {
    "completeness_judgement": "deviation_judgement",
    "completeness_rationale": "deviation_information",
}


def _judge_quality_once(
    messages: list[dict[str, str]],
    *,
    client_choice: str,
    dimension_query: str,
    prereg_top: list[str],
    reasoning_effort: str | None,
) -> ComparisonItem | None:
    """One model judgement for a quality dimension; returns None on an unparseable
    reply (mirrors _judge_dimension_once). The quality field names are mapped onto
    the canonical ComparisonItem fields for storage/back-compat."""
    _c.cost_tracking.pop_reasoning()  # discard any stale stash from a prior call
    parsed_payload: Any = None
    for attempt in range(_c._JUDGEMENT_ATTEMPTS):
        result_json = _c._dispatch_judgement(
            messages,
            client_choice=client_choice,
            reasoning_effort=reasoning_effort,
            response_model=QualityItem,
            claude_tool=_QUALITY_TOOL,
        )
        cleaned_json = _c._extract_json_payload(result_json)
        if cleaned_json:
            try:
                parsed_payload = json.loads(cleaned_json)
                break
            except json.JSONDecodeError:
                logger.warning(
                    "Quality: failed to decode JSON completion (attempt %d/%d)",
                    attempt + 1,
                    _c._JUDGEMENT_ATTEMPTS,
                    extra={"client": client_choice, "raw_result": result_json},
                )
        else:
            logger.warning(
                "Quality: empty completion content (attempt %d/%d)",
                attempt + 1,
                _c._JUDGEMENT_ATTEMPTS,
                extra={"client": client_choice},
            )

    if not isinstance(parsed_payload, (dict, list)):
        return None
    if isinstance(parsed_payload, dict):
        for src, dst in _QUALITY_FIELD_ALIASES.items():
            if src in parsed_payload and dst not in parsed_payload:
                parsed_payload[dst] = parsed_payload.pop(src)
    normalized_payload = _c._normalize_comparison_payload(parsed_payload)
    try:
        parsed_item = ComparisonItem.model_validate(normalized_payload)
    except ValidationError:
        fallback = {k: ("" if v is None else str(v)) for k, v in normalized_payload.items()}
        parsed_item = ComparisonItem.model_validate(fallback)
    parsed_item.deviation_judgement = normalize_quality_verdict(parsed_item.deviation_judgement)
    cited = _c._judge_cited_ids(parsed_item)
    parsed_item.registration_content_quotes = _c._filter_display_quotes(prereg_top, cited)
    parsed_item.paper_content_quotes = ""
    parsed_item.paper_content_summary = ""
    parsed_item.registration_content_summary = ""
    parsed_item.chain_of_thought = _c.cost_tracking.pop_reasoning()
    return parsed_item


def run_quality_assessment(
    registration_input: str,
    client_choice: str,
    dimension_query: str,
    dimension_definition: str | None = None,
    top_k: int | None = None,
    corpus_cache: dict[str, Any] | None = None,
    previous_dimension_responses: list[ComparisonItem] | None = None,
    reasoning_effort: str | None = None,
    evidence_manifest: dict[str, Any] | None = None,
    query_embedding_cache: dict[str, Any] | None = None,
    dimension_keywords: list[str] | None = None,
) -> ComparisonResult:
    """Retrieve + judge ONE quality dimension over the registration corpus.
    Single judge with the targeted verification loop (registration side only)."""
    logger.info(
        "run_quality_assessment invoked",
        extra={"dimension": dimension_query, "client": client_choice, "reasoning_effort": reasoning_effort},
    )
    cache = corpus_cache if corpus_cache is not None else {}
    prereg_key = f"prereg:{_c.hashlib.sha256(registration_input.encode('utf-8')).hexdigest()}"
    embedding_model = _c._embedding_model()

    prereg_corpus = cache.get(prereg_key)
    if prereg_corpus is None:
        prereg_corpus = _c.build_corpus(
            registration_input,
            model=embedding_model,
            embeddings_path=None,
            chunk_prefix="PREREG",
            max_segments=_c._max_embedding_segments(),
            max_chunk_tokens=_c._embedding_max_chunk_tokens(),
        )
        cache[prereg_key] = prereg_corpus
    registration_input = ""

    definition_for_query = (dimension_definition or "").strip()
    augmented_query = _c._augmented_dimension_query(dimension_query, dimension_definition)
    prereg_top_k = top_k if top_k is not None else _c._compute_top_k(len(prereg_corpus.segments))

    _qcache = query_embedding_cache if query_embedding_cache is not None else {}
    _qkey = _c._query_embedding_key(augmented_query)
    query_embedding = _qcache.get(_qkey)
    if query_embedding is None:
        query_embedding = _c.get_embedding(augmented_query, model=embedding_model)
        _qcache[_qkey] = query_embedding

    prereg_scored = _c.retrieve_relevant_chunks(
        query_embedding, prereg_corpus, top_k=len(prereg_corpus.segments)
    )
    prereg_refs = _c._reference_chunk_ids(prereg_corpus)
    if prereg_refs:
        prereg_scored = [row for row in prereg_scored if row[0] not in prereg_refs]
    prereg_top_rows = _c._promote_keyword_hits(
        prereg_scored[:prereg_top_k], prereg_scored, dimension_keywords
    )
    prereg_top_rows = sorted(
        prereg_top_rows,
        key=lambda x: int(x[0].split("_")[-1]) if x[0].split("_")[-1].isdigit() else 0,
    )

    if evidence_manifest is not None:
        chunks = evidence_manifest.setdefault("chunks", {})
        for cid, _text, sim in prereg_top_rows:
            chunk_info = chunks.get(cid)
            if not isinstance(chunk_info, dict):
                continue
            score_map = chunk_info.setdefault("relevance_scores_by_dimension", {})
            score_map[dimension_query] = float(sim)
            current_max = chunk_info.get("max_relevance_score")
            if not isinstance(current_max, (int, float)) or sim > current_max:
                chunk_info["max_relevance_score"] = float(sim)

    prereg_top = [f"[{cid}, relevance_score={sim:.3f}] {text}" for cid, text, sim in prereg_top_rows]
    prereg_prompt = _c._expand_with_neighbors(prereg_top_rows, prereg_corpus, _c._judge_context_window())

    history_context = ""
    if previous_dimension_responses:
        titles = [
            (item.dimension or "").strip()
            for item in previous_dimension_responses
            if (item.dimension or "").strip()
        ]
        lines: list[str] = []
        if titles:
            lines.append(
                "Previously, you were asked to assess the following dimensions: " + ", ".join(titles) + "."
            )
        for item in previous_dimension_responses:
            label = (item.dimension or "this dimension").strip() or "this dimension"
            dumped = json.dumps(
                {
                    "dimension": item.dimension,
                    "completeness_judgement": item.deviation_judgement,
                    "completeness_rationale": item.deviation_information,
                }
            )
            lines.append(f"For {label}, you gave this output: {dumped}")
        history_context = "\n".join(lines).strip()

    static_doctrine = _quality_static_doctrine()
    variable_part = (
        f"The quality dimension on which you should assess the registration is: '{dimension_query}'; this is defined as "
        f"{definition_for_query if definition_for_query else 'not provided by the user.'}\n\n"
        "Use ONLY the provided evidence excerpts. Each excerpt is labeled with an ID in square brackets.\n\n"
        "Registration excerpts:\n"
        f"{' '.join(prereg_prompt)}\n"
        "\n"
        "Now produce the JSON object described above for this dimension. Reminder — 'complete' requires every called-for element to be precise and unambiguous; anything addressed but vague, ambiguous, or deferred is 'partial'; only judge 'absent' when nothing usable addresses the dimension, and list anything you could not find in 'unlocated_in_registration' instead of softening the verdict.\n"
    )
    if history_context:
        master_prompt = static_doctrine + history_context + "\n\n" + variable_part
    else:
        master_prompt = static_doctrine + variable_part

    messages = [
        {"role": "system", "content": _QUALITY_SYSTEM_MESSAGE},
        {"role": "user", "content": master_prompt},
    ]

    def _judge(msgs: list[dict[str, str]]) -> ComparisonItem | None:
        return _judge_quality_once(
            msgs,
            client_choice=client_choice,
            dimension_query=dimension_query,
            prereg_top=prereg_top,
            reasoning_effort=reasoning_effort,
        )

    # Verification pass (registration side): unlocated elements trigger a targeted
    # full-corpus search; new material -> one re-judgement on the augmented prompt.
    pass1 = _judge(messages)
    item: ComparisonItem | None = pass1
    searched: list[str] = []
    augmented = False
    if pass1 is not None:
        shown_ids = _c._ids_in_prompt_blocks(prereg_prompt)
        extra_blocks: list[str] = []
        for element in _c._split_unlocated(getattr(pass1, "unlocated_in_registration", "")):
            hits = _c._targeted_element_search(
                element, prereg_corpus, exclude_ids=shown_ids, embedding_model=embedding_model
            )
            searched.append(f"'{element}' (registration)")
            if hits:
                shown_ids.update(cid for cid, _t, _s in hits)
                labelled = " ".join(
                    f"[{cid}, relevance_score={sim:.3f}] {text}" for cid, text, sim in hits
                )
                extra_blocks.append(
                    f"Additional targeted registration excerpts (found by a full-document search for: {element}):\n{labelled}"
                )
        if extra_blocks:
            augmented = True
            messages = [
                messages[0],
                {"role": "user", "content": messages[1]["content"] + "\n\n" + "\n\n".join(extra_blocks)},
            ]
            item = _judge(messages) or pass1

    if item is None:
        item = _c._degraded_item(dimension_query, [], prereg_top)

    if searched and item is not None and item.deviation_information:
        if augmented:
            note = (
                "(A targeted full-document search retrieved additional excerpts for: "
                + "; ".join(searched)
                + " — the assessment above includes them.)"
            )
        else:
            note = (
                "(RegCheck ran a targeted full-document search for: "
                + "; ".join(searched)
                + ". No further mentions were found beyond the excerpts already provided.)"
            )
        base = (item.deviation_information or "").rstrip()
        item.deviation_information = f"{base}\n\n{note}" if base else note
    return ComparisonResult(items=[item])


async def registration_quality_assessment(
    prereg_path: str,
    prereg_ext: str,
    client_choice: str,
    parser_choice: str,
    task_id: str | None = None,
    redis_client: Any | None = None,
    selected_dimensions: list[dict[str, str]] | None = None,
    append_previous_output: bool = False,
    reasoning_effort: str | None = None,
    pdf_parser: Callable[[str], Awaitable[str]] | None = None,
    dpt_parser: Callable[[str], Awaitable[Any]] | None = None,
    docx_reader: Callable[[str], str] | None = None,
    assessment_runner: Callable[..., ComparisonResult] | None = None,
    evidence_out: dict[str, Any] | None = None,
) -> ComparisonResult:
    """Orchestrate a full registration-quality report: parse the registration,
    build the (single-source) evidence manifest, assess every dimension, persist
    progress/results to the task hash. Mirrors general_preregistration_comparison."""
    import asyncio

    _c.cost_tracking.start_run()  # live per-run token/cost accounting
    processed_count = 0
    parser_choice_normalized = (parser_choice or "pymupdf").lower()
    if prereg_ext == ".pdf":
        try:
            registration_input, prereg_parser_used = await extract_pdf_text(
                prereg_path,
                parser_choice=parser_choice_normalized,
                pdf_parser=pdf_parser,
                dpt_parser=dpt_parser,
            )
            if task_id and redis_client:
                fields: dict[str, str] = {"prereg_parser_used": prereg_parser_used}
                if prereg_parser_used != parser_choice_normalized:
                    fields["status"] = f"Scanned registration PDF detected; using {prereg_parser_used} fallback"
                await redis_client.hset(task_id, mapping=fields)
        except Exception as exc:
            if task_id and redis_client:
                await redis_client.hset(
                    task_id,
                    mapping={
                        "state": "FAILURE",
                        "status": f"Registration parsing failed: {exc}",
                        "processed_dimensions": processed_count,
                    },
                )
            raise
    else:
        try:
            if prereg_ext == ".docx" and docx_reader is not None:
                registration_input = docx_reader(prereg_path)
            else:
                registration_input = read_file(prereg_path, prereg_ext)
        except Exception as exc:
            raise ValueError(
                f"Couldn't read the registration ('{prereg_ext or 'unknown'}'): {exc}"
            ) from exc

    result_obj = ComparisonResult(items=[])
    dimensions_to_assess = _resolve_dimensions(selected_dimensions, registration_quality_dimensions())
    dimension_names = [item["dimension"] for item in dimensions_to_assess]
    total_dimensions = len(dimensions_to_assess)

    runner = assessment_runner or run_quality_assessment
    corpus_cache: dict[str, Any] = {}
    query_embedding_cache: dict[str, Any] = {}
    evidence_manifest: dict[str, Any] | None = None
    evidence_ttl_seconds = await _c._current_task_ttl(redis_client, task_id)
    want_evidence = bool(task_id and redis_client) or evidence_out is not None
    if want_evidence:
        if task_id and redis_client:
            await redis_client.hset(
                task_id,
                mapping={
                    "status": "Preparing evidence viewer sources",
                    "evidence_status": "preparing",
                    "evidence_error": "",
                },
            )
        prereg_payload = _c.build_file_evidence_source(
            source_id="registration",
            label="Registration",
            file_path=prereg_path,
            file_ext=prereg_ext,
            text=registration_input,
            chunk_prefix="PREREG",
            metadata={"role": "registration", "comparison_type": COMPARISON_TYPE},
            max_chunk_tokens=_c._embedding_max_chunk_tokens(),
            embedding_model=_c._embedding_model(),
        )
        registration_input = prereg_payload.get("text") or registration_input
        _c._add_evidence_corpus_to_cache(
            corpus_cache,
            role="prereg",
            chunk_prefix="PREREG",
            source_payload=prereg_payload,
        )
        source_payloads = [prereg_payload]
        if task_id and redis_client:
            evidence_manifest = await _c._store_evidence_manifest(
                redis_client=redis_client,
                task_id=task_id,
                comparison_type=COMPARISON_TYPE,
                source_payloads=source_payloads,
                ttl_seconds=evidence_ttl_seconds,
            )
        else:
            evidence_manifest, _ = _c._assemble_inline_bundle(
                task_id, COMPARISON_TYPE, source_payloads
            )
        if evidence_out is not None:
            inline_manifest, inline_render_data = _c._assemble_inline_bundle(
                task_id, COMPARISON_TYPE, source_payloads
            )
            evidence_out["manifest"] = inline_manifest
            evidence_out["render_data"] = inline_render_data

    logger.info(
        "registration_quality_assessment start",
        extra={"client_choice": client_choice, "total_dimensions": total_dimensions},
    )
    if task_id and redis_client:
        await redis_client.hset(
            task_id,
            mapping={
                "state": "IN_PROGRESS",
                "result_json": _c._result_json_with_cost(result_obj),
                "total_dimensions": total_dimensions,
                "processed_dimensions": 0,
                "dimensions": json.dumps(dimension_names),
                "status": "Embedding the registration",
            },
        )

    try:
        if runner is run_quality_assessment:
            query_embedding_cache = await _c._prebuild_query_embeddings(
                dimensions_to_assess, embedding_model=_c._embedding_model()
            )
        for index, dimension_info in enumerate(dimensions_to_assess, start=1):
            if not isinstance(dimension_info, dict):
                continue
            dimension_name = (dimension_info.get("dimension") or dimension_info.get("name") or "").strip()
            if not dimension_name:
                continue
            dimension_definition = (dimension_info.get("definition") or "").strip()
            previous_responses: list[ComparisonItem] | None = None
            if append_previous_output and result_obj.items:
                previous_responses = list(result_obj.items)
            if task_id and redis_client:
                await redis_client.hset(
                    task_id,
                    mapping={"status": f"Assessing '{dimension_name}'"},
                )
            assessment = await asyncio.to_thread(
                runner,
                registration_input,
                client_choice,
                dimension_name,
                dimension_definition=dimension_definition,
                dimension_keywords=dimension_info.get("keywords") or [],
                corpus_cache=corpus_cache,
                query_embedding_cache=query_embedding_cache,
                reasoning_effort=reasoning_effort,
                previous_dimension_responses=previous_responses,
                evidence_manifest=evidence_manifest,
            )
            result_obj.items.extend(assessment.items)
            processed_count = index
            await _c._persist_evidence_manifest(
                redis_client,
                task_id,
                evidence_manifest,
                evidence_ttl_seconds,
            )
            if task_id and redis_client:
                await redis_client.hset(
                    task_id,
                    mapping={
                        "state": "IN_PROGRESS",
                        "result_json": _c._result_json_with_cost(result_obj),
                        "processed_dimensions": index,
                        "total_dimensions": total_dimensions,
                        "status": f"Assessed {index}/{total_dimensions}: {dimension_name}",
                    },
                )
    except Exception as exc:
        if task_id and redis_client:
            await redis_client.hset(
                task_id,
                mapping={
                    "state": "FAILURE",
                    "status": f"Processing failed: {exc}",
                    "result_json": _c._result_json_with_cost(result_obj),
                    "processed_dimensions": processed_count,
                    "total_dimensions": total_dimensions,
                },
            )
        raise

    if task_id and redis_client:
        evidence_fields = await _c._evidence_success_fields(redis_client, task_id, evidence_manifest)
        await redis_client.hset(
            task_id,
            mapping={
                "state": "SUCCESS",
                "result_json": _c._result_json_with_cost(result_obj),
                "total_dimensions": total_dimensions,
                "processed_dimensions": total_dimensions,
                "dimensions": json.dumps(dimension_names),
                "status": "Report complete",
                **evidence_fields,
            },
        )
    tracker = _c.cost_tracking.current()
    if tracker is not None:
        result_obj.cost = tracker.snapshot()
    return result_obj


__all__ = [
    "COMPARISON_TYPE",
    "QualityItem",
    "normalize_quality_verdict",
    "run_quality_assessment",
    "registration_quality_assessment",
]
