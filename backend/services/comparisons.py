from __future__ import annotations

import asyncio
import csv
import hashlib
import math
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, TypeVar

from dotenv import load_dotenv

from groq import Groq
from openai import OpenAI
from pydantic import BaseModel, ValidationError, field_validator

from .documents import (
    extract_text_from_docx,
    read_file,
    read_file_as_pdf,
)
from .embeddings import (
    EmbeddingCorpus,
    build_corpus,
    build_corpus_from_segments,
    get_embedding,
    retrieve_relevant_chunks,
)
from .evidence import (
    build_file_evidence_source,
    build_json_evidence_source,
    build_text_evidence_source,
)
from .pdf_parsers import extract_pdf_text, pdf2dpt, pdf2grobid
from .report_artifacts import (
    store_manifest,
    store_source_artifacts,
    verify_manifest_artifacts,
)
from .trials import extract_nct_id, extract_nested_trial_with_metadata

logger = logging.getLogger(__name__)

load_dotenv()

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

DEFAULT_OPENAI_MODEL = "gpt-5"
DEFAULT_DEEPSEEK_MODEL = "deepseek-reasoner"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_MAX_SEGMENTS = 1200
DEFAULT_MAX_CONCURRENT_TASKS = 8

T = TypeVar("T")


def _env_str(name: str, default: str | None = None) -> str:
    value = (os.environ.get(name) or "").strip()
    if value:
        return value
    if default is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _is_provider_auth_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 401:
        return True
    exc_name = type(exc).__name__.lower()
    text = str(exc).lower()
    return (
        "authentication" in exc_name
        or "unauthorized" in text
        or "invalid_api_key" in text
        or "invalid api key" in text
    )


def _raise_provider_auth_error(provider: str, env_var: str, exc: Exception) -> None:
    raise RuntimeError(
        f"{provider} authentication failed. Check Heroku config var {env_var} "
        f"or choose a different model provider."
    ) from exc


def _is_response_format_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "response_format" in text or "json_object" in text or "json mode" in text


def _groq_chat_completion(*, model: str, messages: list[dict[str, str]], use_json_mode: bool):
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }
    if use_json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        return groq_client.chat.completions.create(**kwargs)
    except Exception as exc:
        if _is_provider_auth_error(exc):
            _raise_provider_auth_error("Groq", "GROQ_API_KEY", exc)
        if not use_json_mode or not _is_response_format_error(exc):
            raise
        logger.info(
            "Groq JSON response_format unsupported; retrying without response_format",
            extra={"model": model, "error": str(exc)},
        )
        try:
            return groq_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )
        except Exception as retry_exc:
            if _is_provider_auth_error(retry_exc):
                _raise_provider_auth_error("Groq", "GROQ_API_KEY", retry_exc)
            raise


def _openai_model() -> str:
    return _env_str("OPENAI_COMPARISON_MODEL", _env_str("OPENAI_MODEL", DEFAULT_OPENAI_MODEL))


def _openai_experiment_model() -> str:
    return _env_str("OPENAI_EXPERIMENT_MODEL", _openai_model())


def _deepseek_model() -> str:
    return _env_str("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)


def _groq_model() -> str:
    return _env_str("GROQ_MODEL", DEFAULT_GROQ_MODEL)


def _max_embedding_segments() -> int:
    configured = _env_int("MAX_EMBEDDING_SEGMENTS", DEFAULT_MAX_SEGMENTS)
    return max(100, configured)


def _embedding_model() -> str:
    return _env_str("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")


def _embedding_max_chunk_tokens() -> int:
    configured = _env_int("EMBEDDING_MAX_CHUNK_TOKENS", 300)
    return max(100, configured)


_comparison_semaphore = asyncio.Semaphore(
    max(1, _env_int("MAX_CONCURRENT_COMPARISON_TASKS", DEFAULT_MAX_CONCURRENT_TASKS))
)


async def run_with_concurrency_limit(func: Callable[[], Awaitable[T]]) -> T:
    """Run a coroutine factory under a shared semaphore to cap concurrent comparisons."""
    async with _comparison_semaphore:
        return await func()


def _normalize_reasoning_effort_value(value: str | None) -> str | None:
    normalized = (value or "").strip().lower()
    if not normalized:
        return None
    if normalized not in {"low", "medium", "high"}:
        return "medium"
    return normalized


def _openai_error_param(exc: Exception) -> str | None:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            param = error.get("param")
            if isinstance(param, str) and param.strip():
                return param.strip()

    message = str(exc)
    for candidate in ("reasoning_effort", "response_format"):
        if candidate in message:
            return candidate
    return None


def _openai_chat_json(
    openai_client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    reasoning_effort: str | None = None,
) -> str:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    normalized_effort = _normalize_reasoning_effort_value(reasoning_effort)
    if normalized_effort:
        kwargs["reasoning_effort"] = normalized_effort

    while True:
        try:
            response = openai_client.chat.completions.create(**kwargs)
            break
        except Exception as exc:
            param = _openai_error_param(exc)
            if param and param in kwargs:
                logger.info(
                    "OpenAI call failed with %s; retrying without it",
                    param,
                    extra={"model": model},
                    exc_info=exc,
                )
                kwargs.pop(param, None)
                continue
            raise
    return _message_content_to_text(response.choices[0].message)


def _openai_chat_text(
    openai_client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    reasoning_effort: str | None = None,
) -> str:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    normalized_effort = _normalize_reasoning_effort_value(reasoning_effort)
    if normalized_effort:
        kwargs["reasoning_effort"] = normalized_effort

    while True:
        try:
            response = openai_client.chat.completions.create(**kwargs)
            break
        except Exception as exc:
            param = _openai_error_param(exc)
            if param and param in kwargs:
                logger.info(
                    "OpenAI call failed with %s; retrying without it",
                    param,
                    extra={"model": model},
                    exc_info=exc,
                )
                kwargs.pop(param, None)
                continue
            raise
    return _message_content_to_text(response.choices[0].message)


def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Please contact administrators."
        )
    return OpenAI(api_key=api_key)


def get_deepseek_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing DEEPSEEK_API_KEY. Please contact administrators."
        )
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


# Canonical default comparison dimensions. Dimensions and their definitions
# are resolved ONCE, at selection time, via _resolve_dimensions():
# user-specified values (UI or API args) are used verbatim; otherwise the
# explicit default set for the flow applies. There is no other source — in
# particular, run_comparison() never substitutes a definition by name.

# Defaults for clinical trial (NCT) comparisons.
CLINICAL_DEFAULT_DIMENSIONS: list[dict[str, str]] = [
    {
        "dimension": "Eligibility – inclusion criteria",
        "definition": (
            "All explicitly stated conditions, characteristics, or thresholds that must be satisfied for a "
            "participant or study unit (e.g., individual, cluster, site) to be enrolled in the study. This "
            "includes required demographic characteristics (e.g., age range, sex/gender, pregnancy/menopausal "
            "status), diagnosis or target condition (including staging, severity, symptom duration, biomarker "
            "status), clinical status (e.g., performance status, treatment-naïve status), prior or current "
            "treatments that are required or permitted, contextual factors (e.g., recruitment setting, "
            "healthcare setting, geography), and other positive requirements (e.g., language proficiency, "
            "ability to comply with study procedures, capacity to consent, access to required technology or "
            "devices), typically expressed using terms such as “must have,” “must meet,” “only participants "
            "with,” or “eligible if.” Commonly appears as bullet points or prose under headings such as "
            "“Inclusion criteria,” “Eligibility criteria,” or “Participants” in trial registries, protocols, "
            "and manuscripts."
        ),
    },
    {
        "dimension": "Eligibility – exclusion criteria",
        "definition": (
            "All explicitly stated conditions or characteristics that disqualify a participant from enrollment "
            "or continued participation in the study. This includes safety-related exclusions (e.g., "
            "contraindications, relevant comorbidities, allergies, pregnancy where specified as exclusion, "
            "severe organ dysfunction), protocol- or confounding-related exclusions (e.g., concurrent trial "
            "participation, use of prohibited concomitant treatments, prior exposure to the study intervention "
            "beyond allowed limits), feasibility-related exclusions (e.g., inability or unwillingness to adhere "
            "to required visits, procedures, or follow-up), and history or condition-based exclusions indicated "
            "by phrases such as “must not,” “excluded if,” “no history of,” or “not eligible if.” Commonly "
            "appears under “Exclusion criteria” or within “Eligibility”/“Participants” sections in trial "
            "registries, protocols, and manuscripts."
        ),
    },
    {
        "dimension": "Intervention/treatment and control/placebo",
        "definition": (
            "The detailed specification of each study arm, including both experimental and comparator arms. "
            "This encompasses the name or identity of the intervention (e.g., drug, biologic, device, surgical "
            "procedure, behavioural or psychological program, digital/app-based intervention), the dose or "
            "intensity (e.g., strength, amount, frequency, duration of sessions), route or mode of "
            "administration (e.g., oral, IV, subcutaneous, inhaled, implanted, online, in-person, telephone), "
            "schedule and timing (including dosing schedules, titration, tapering, timing relative to baseline "
            "or randomisation), planned duration of administration, and a clear description of the control or "
            "comparator condition, such as placebo (including route and appearance), sham procedure, active "
            "control (with its own dose and schedule), or usual/standard care (with defining elements). It "
            "also includes any mandated co-interventions (e.g., background therapy) and any explicitly "
            "prohibited concomitant treatments. Commonly appears under headings such as “Interventions,” "
            "“Study treatments,” “Arm description,” “Trial arms,” or “Treatment protocol.”"
        ),
    },
    {
        "dimension": "Ethical approval – number",
        "definition": (
            "The specific identifier(s) associated with ethical approval of the study, including ethics "
            "committee/IRB/REC/REB approval numbers, protocol codes linked to ethics approval, national or "
            "institutional reference numbers, and identifiers for substantial amendments where explicitly "
            "reported. These identifiers are typically presented in sections or statements labelled “Ethics "
            "approval,” “Ethical considerations,” “IRB (Institutional Review Board) approval,” or similar, and "
            "may appear in trial registries, protocols, and manuscripts."
        ),
    },
    {
        "dimension": "Ethical approval – committee",
        "definition": (
            "The official name(s) of the ethics body or bodies responsible for reviewing and approving the "
            "study, such as Institutional Review Boards (IRBs), Research Ethics Committees (RECs), Research "
            "Ethics Boards (REBs), hospital or university ethics committees, and national or regional "
            "competent authorities performing ethics review. Names often include the institution and country "
            "or region (e.g., “University X Research Ethics Committee, Country Y”) and are typically reported "
            "in ethics-related sections or statements in trial registries, protocols, and manuscripts."
        ),
    },
    {
        "dimension": "Ethical approval – date",
        "definition": (
            "The calendar date on which ethics approval was granted for the primary study protocol, and, where "
            "explicitly reported, the dates on which major amendments were approved. Dates may be presented in "
            "full (e.g., “15 March 2022”) or in partial formats (e.g., month/year) and are usually mentioned "
            "alongside the ethics committee name or approval number in sections titled “Ethics approval,” "
            "“Ethical considerations,” or similar in trial registries, protocols, and manuscripts."
        ),
    },
    {
        "dimension": "Sample size",
        "definition": (
            "The numerical specification of the number of participants associated with the study, including "
            "planned or target total sample size, planned or target per-arm sample size (where reported), and "
            "actual numbers enrolled, randomized, allocated, or analyzed, overall and by arm (where reported). "
            "Expressions may include phrases such as “target sample size,” “planned enrollment,” “anticipated "
            "enrollment,” “a total of N participants,” “we aimed to recruit,” or “N participants were "
            "randomized/analyzed,” and typically appear in “Sample size,” “Methods,” “Participants,” “Study "
            "design,” “Statistical methods,” enrollment fields, or CONSORT-style flow diagrams."
        ),
    },
    {
        "dimension": "Date recruitment started",
        "definition": (
            "The planned or actual date on which enrollment of the first participant began. This may be "
            "expressed as an exact date (e.g., “Recruitment started on 01 June 2021”) or a less specific time "
            "reference (e.g., “Recruitment commenced in June 2021”) and is commonly found in fields or text "
            "labelled “Study start date,” “Date of first enrollment,” “Recruitment,” or within “Methods” and "
            "“Study design” sections in trial registries, protocols, and manuscripts."
        ),
    },
    {
        "dimension": "Outcomes – primary",
        "definition": (
            "All outcome measures explicitly designated as “primary,” “main,” or “primary endpoint(s)” that "
            "define the principal endpoints of interest for evaluating the intervention. A well-specified "
            "primary outcome typically includes: (a) the outcome domain or construct (e.g., overall survival, "
            "hospitalization, pain, HbA1c, depression severity), (b) the measurement instrument or operational "
            "definition (e.g., named clinical scale, laboratory test, imaging modality, diagnostic or response "
            "criteria, event definition algorithm), (c) the metric or scale (e.g., absolute value, change from "
            "baseline, proportion meeting a threshold, time-to-event, rate, composite index, including units "
            "where applicable), and (d) the assessment timepoint(s) (e.g., at 12 weeks, day 28, at discharge, "
            "6 months post-randomisation). Such information is typically reported in “Primary outcome(s)” or "
            "“Endpoints” fields, “Outcomes” or “Endpoints” sections, abstracts, and statistical analysis "
            "descriptions. Some primary outcomes may be composites of other primary or secondary outcomes."
        ),
    },
    {
        "dimension": "Outcomes – secondary",
        "definition": (
            "All outcome measures explicitly labelled as “secondary,” “key secondary,” or similar that are "
            "intended to provide supportive, exploratory, mechanistic, or safety-related information beyond "
            "the primary endpoints. A well-specified secondary outcome similarly includes: (a) the outcome "
            "domain or construct, (b) the measurement instrument or operational definition, (c) the metric or "
            "scale, and (d) the assessment timepoint(s). These outcomes may include safety endpoints, adverse "
            "events, laboratory parameters, quality of life, functional status, biomarker changes, "
            "intermediate clinical outcomes, or subgroup-specific effects, and are typically documented in "
            "“Secondary outcome(s)” fields and “Secondary outcomes” or “Endpoints” sections. Some secondary "
            "outcomes may be composites of other primary or secondary outcomes."
        ),
    },
    {
        "dimension": "Method of randomisation and allocation",
        "definition": (
            "The described method used to generate and implement the allocation of participants or clusters "
            "to study arms. This includes the sequence generation approach (e.g., computer-generated random "
            "numbers, random-number tables, permuted block randomisation with or without specified block "
            "sizes, stratified randomisation by site or baseline characteristics, minimiz(s)ation or dynamic "
            "allocation procedures), the allocation ratio between arms (e.g., 1:1, 2:1, 3:1:1), and, where "
            "described in the same context, the allocation concealment mechanism (e.g., central randomisation, "
            "secure web- or telephone-based systems such as IWRS/IVRS, pharmacy-controlled allocation, sealed "
            "opaque sequentially numbered envelopes). Common textual indicators include phrases such as "
            "“participants were randomiz(s)ed using…,” “random sequence generated by…,” “block size…,” "
            "“stratified by…,” and “allocation was concealed using…,” and this information typically appears "
            "in “Randomiz(s)ation,” “Study design,” or “Methods” sections and corresponding fields in trial "
            "documentation."
        ),
    },
]

# Defaults for preclinical/animal (PCT) comparisons. Definitions are aligned
# with the clinical/psychology wording where it transfers, but written for the
# preclinical in-vivo context (animals as the experimental unit, the 3Rs,
# ARRIVE 2.0 reporting, litter/cage/batch effects, humane endpoints).
PRECLINICAL_DEFAULT_DIMENSIONS: list[dict[str, str]] = [
    {
        "dimension": "Study type (exploratory vs. confirmatory)",
        "definition": (
            "Whether the study is described as exploratory (hypothesis-generating, with outcomes and analyses "
            "not fully pre-specified) or confirmatory (hypothesis-testing, with pre-specified predictions, "
            "outcomes, and analyses), and whether this designation is consistent between the protocol/"
            "preregistration and the publication. Confirmatory work states specific predictions before data "
            "collection and tests them with pre-specified analyses; exploratory work investigates patterns "
            "without prespecified hypotheses. Look for explicit statements (e.g., 'this study was exploratory/"
            "confirmatory', 'hypothesis-generating', 'pre-specified primary analysis') and for whether the "
            "paper retrospectively reframes exploratory analyses as confirmatory (or vice versa). Commonly "
            "indicated in the title, abstract, 'Study design', or 'Statistical analysis' sections."
        ),
    },
    {
        "dimension": "Total number of animals",
        "definition": (
            "The total number of animals planned for the study, the sum across all experimental and control "
            "groups as stated in the protocol, and, where reported, the number of animals actually used, "
            "randomised, or analysed overall. May be described as total sample size, number of animals, total "
            "N, planned cohort size, or the estimated number of animals declared to an ethics or animal-welfare "
            "body. The experimental unit (animal, litter, cage, or cohort) should be considered, and "
            "discrepancies may arise from attrition, mortality, humane endpoints, or exclusions; note any "
            "difference between the planned total and the number ultimately analysed. Typically reported in "
            "'Animals', 'Sample size', 'Methods', ethics/licence sections, or a study-flow diagram."
        ),
    },
    {
        "dimension": "Number of animals per group",
        "definition": (
            "The planned number of animals allocated to each experimental and control group (group size, "
            "per-group sample size, n per arm/condition) as stated in the protocol, and, where reported, the "
            "actual numbers per group. This is often justified by an a priori power or sample-size calculation "
            "(assumed effect size, variability, alpha, power) or by a resource-equation/3Rs rationale. Note any "
            "divergence between planned and realised group sizes and whether the experimental unit is the "
            "individual animal or a higher-level unit (litter, cage). Typically reported in 'Animals', 'Sample "
            "size', 'Experimental design', or 'Statistical methods' sections, or in figure legends."
        ),
    },
    {
        "dimension": "Intervention and control",
        "definition": (
            "The specific intervention(s) planned and the control or comparator condition against which they "
            "are assessed. For the intervention this includes its identity (e.g., compound/drug, biologic, "
            "genetic or surgical manipulation, dietary or behavioural manipulation, device), the dose or "
            "intensity (amount, concentration, frequency), the route or mode of administration (e.g., oral "
            "gavage, intraperitoneal, intravenous, subcutaneous, inhaled, surgical), and the schedule, timing, "
            "and planned duration of administration. For the control this includes the comparator condition "
            "(e.g., vehicle, sham surgery, untreated, wild-type or littermate control, naive animals) specified "
            "with the same level of detail (route, volume, timing). It also includes any mandated "
            "co-interventions and explicitly prohibited concomitant treatments. Typically reported under "
            "'Interventions', 'Treatments', 'Experimental groups', 'Study design', or 'Methods'."
        ),
    },
    {
        "dimension": "Measures to reduce bias",
        "definition": (
            "Methods planned to reduce the risk of bias in conducting and analysing the study. This includes "
            "randomisation of animals (or higher-level units) to groups and the sequence-generation method "
            "(e.g., computer-generated random numbers, block or stratified randomisation), allocation "
            "concealment, and blinding/masking of those administering interventions, caring for animals, "
            "assessing outcomes, and analysing data. It also covers other bias-mitigation steps such as "
            "pre-specified inclusion/exclusion rules for animals or data points, randomised order of treatment "
            "and measurement, and steps to control for cage, litter, batch, or experimenter effects. Note "
            "whether each measure is stated as planned and whether the paper reports it as implemented. "
            "Commonly reported in 'Randomisation', 'Blinding', 'Experimental design', or 'Methods' sections "
            "(aligned with ARRIVE 2.0 items)."
        ),
    },
    {
        "dimension": "Primary outcomes",
        "definition": (
            "All outcome measures explicitly designated as primary, main, or the principal endpoint(s) used to "
            "evaluate the intervention. A well-specified primary outcome includes (a) the outcome domain or "
            "construct (e.g., tumour volume, infarct size, lesion area, survival, a behavioural readout, a "
            "physiological or molecular marker), (b) the measurement instrument or operational definition "
            "(e.g., named assay, imaging modality, behavioural test, histological scoring method), (c) the "
            "metric or scale (e.g., absolute value, change from baseline, proportion meeting a threshold, "
            "time-to-event, including units), and (d) the assessment timepoint(s) (e.g., at day 14, 6 weeks "
            "post-induction, terminal). Note whether the primary outcome named in the paper matches the one "
            "pre-specified in the protocol. Typically reported in 'Primary outcome', 'Outcomes', 'Endpoints', "
            "or the statistical-analysis description. Some primary outcomes may be composites of other outcomes."
        ),
    },
    {
        "dimension": "Secondary outcomes",
        "definition": (
            "All outcome measures explicitly labelled as secondary, additional, or exploratory that provide "
            "supportive, mechanistic, or safety/welfare-related information beyond the primary endpoint. A "
            "well-specified secondary outcome similarly includes (a) the outcome domain or construct, (b) the "
            "measurement instrument or operational definition, (c) the metric or scale, and (d) the assessment "
            "timepoint(s). These may include additional behavioural, physiological, biochemical, histological, "
            "or molecular measures, adverse events or tolerability/welfare indicators, and subgroup or "
            "time-course analyses. Note any secondary outcomes reported in the paper that were not "
            "pre-specified, or pre-specified ones not reported. Typically reported in 'Secondary outcomes', "
            "'Outcomes', or 'Endpoints' sections."
        ),
    },
    {
        "dimension": "Statistical analyses",
        "definition": (
            "The statistical methods planned to test each hypothesis or estimate each effect, as specified "
            "before data collection. This includes the specific tests or models (e.g., t-test, ANOVA, "
            "mixed-effects/multilevel models, survival analysis, regression), the experimental unit used for "
            "analysis (animal, litter, cage), the handling of nested or repeated-measures data, covariates and "
            "factors included, planned corrections for multiple comparisons, the criterion for statistical "
            "significance, and rules for handling missing data, outliers, and excluded animals. Note any "
            "divergence between the planned analysis and the analysis reported, including added, removed, or "
            "changed tests. Typically reported in 'Statistical analysis', 'Data analysis', or 'Methods' sections."
        ),
    },
    {
        "dimension": "Hypotheses",
        "definition": (
            "The specific predictions the study set out to test, as stated prior to data collection, including "
            "their direction (e.g., which group is expected to differ and in which direction) and whether each "
            "is designated confirmatory or exploratory. A well-specified hypothesis links a predicted effect to "
            "the intervention and the primary outcome. Note whether the hypotheses tested in the paper match "
            "those registered in direction and operationalisation, and whether any predictions appear to have "
            "been added, dropped, or reversed after seeing the data. Typically reported in the introduction, "
            "'Objectives/Aims', 'Hypotheses', or 'Study design' sections."
        ),
    },
]


def _normalize_selected_dimensions(
    selected_dimensions: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    """Normalize user-specified dimensions (from the UI wizard or API args)
    into {dimension, definition} pairs, used verbatim — empty definitions
    stay empty."""
    normalized: list[dict[str, str]] = []
    for item in selected_dimensions or []:
        if not isinstance(item, dict):
            continue
        name = (item.get("dimension") or item.get("name") or "").strip()
        if not name:
            continue
        normalized.append({"dimension": name, "definition": (item.get("definition") or "").strip()})
    return normalized


def _resolve_dimensions(
    selected_dimensions: list[dict[str, str]] | None,
    defaults: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """The single resolution point for comparison dimensions: user-specified
    values win; otherwise the explicitly selected default set applies."""
    normalized = _normalize_selected_dimensions(selected_dimensions)
    if normalized:
        return normalized
    return [dict(item) for item in defaults or []]


class ComparisonItem(BaseModel):
    dimension: str = ""
    paper_content_quotes: str = ""
    paper_content_summary: str = ""
    registration_content_quotes: str = ""
    registration_content_summary: str = ""
    deviation_judgement: str = ""
    deviation_information: str = ""

    @field_validator(
        "paper_content_quotes",
        "registration_content_quotes",
        mode="before",
    )
    @classmethod
    def _quotes_to_string(cls, v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            parts: list[str] = []
            for item in v:
                try:
                    s = str(item).strip()
                except Exception:
                    s = ""
                if s:
                    parts.append(s)
            return "\n\n".join(parts)
        if isinstance(v, dict):
            # Try common containers for quoted text
            candidates: list[str] = []
            for key in ("quotes", "items", "values", "data"):
                if key in v and isinstance(v[key], list):
                    for item in v[key]:
                        s = str(item).strip()
                        if s:
                            candidates.append(s)
            if candidates:
                return "\n\n".join(candidates)
            try:
                return json.dumps(v, ensure_ascii=False)
            except Exception:
                return str(v)
        return str(v)

    @field_validator(
        "dimension",
        "paper_content_summary",
        "registration_content_summary",
        "deviation_judgement",
        "deviation_information",
        mode="before",
    )
    @classmethod
    def _coerce_to_string(cls, v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            parts: list[str] = []
            for item in v:
                try:
                    s = str(item).strip()
                except Exception:
                    s = ""
                if s:
                    parts.append(s)
            return " ".join(parts)
        if isinstance(v, dict):
            # Prefer concatenating string-like values
            vals: list[str] = []
            for val in v.values():
                if isinstance(val, (str, int, float)):
                    vals.append(str(val).strip())
            if vals:
                return " ".join([t for t in vals if t])
            try:
                return json.dumps(v, ensure_ascii=False)
            except Exception:
                return str(v)
        return str(v)


class ComparisonResult(BaseModel):
    items: list[ComparisonItem]


def _compute_top_k(total_segments: int, pct: float = 0.1, min_k: int = 6, max_k: int = 20) -> int:
    """Compute a bounded top-k based on a proportion of available segments."""
    if total_segments <= 0:
        return 0
    estimated = math.ceil(total_segments * pct)
    bounded = max(min_k, estimated)
    bounded = min(max_k, bounded)
    return min(total_segments, bounded)


def _corpus_cache_key(role: str, text: str) -> str:
    return f"{role}:{hashlib.sha256((text or '').encode('utf-8')).hexdigest()}"


def _add_evidence_corpus_to_cache(
    corpus_cache: dict[str, EmbeddingCorpus],
    *,
    role: str,
    chunk_prefix: str,
    source_payload: dict[str, Any],
) -> None:
    corpus_cache[_corpus_cache_key(role, source_payload.get("text", ""))] = build_corpus_from_segments(
        source_payload.get("segments") or [],
        model=_embedding_model(),
        chunk_prefix=chunk_prefix,
        max_segments=_max_embedding_segments(),
        metadata=source_payload.get("metadata") or [],
    )


async def _store_evidence_manifest(
    *,
    redis_client: Any,
    task_id: str,
    comparison_type: str,
    source_payloads: list[dict[str, Any]],
    ttl_seconds: int,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "version": 1,
        "task_id": task_id,
        "comparison_type": comparison_type,
        "sources": {},
        "chunks": {},
    }
    for payload in source_payloads:
        source = await store_source_artifacts(
            redis_client,
            task_id=task_id,
            source=payload["source"],
            raw_bytes=payload.get("raw_bytes"),
            raw_content_type=payload.get("raw_content_type"),
            render_data=payload.get("render_data") or {},
            ttl_seconds=ttl_seconds,
        )
        manifest["sources"][source["id"]] = source
        manifest["chunks"].update(payload.get("chunks") or {})
    await store_manifest(
        redis_client,
        task_id=task_id,
        manifest=manifest,
        ttl_seconds=ttl_seconds,
    )
    try:
        stats = await verify_manifest_artifacts(
            redis_client,
            task_id=task_id,
            manifest=manifest,
        )
        await redis_client.hset(
            task_id,
            mapping={
                "evidence_status": "ready",
                "evidence_error": "",
                "evidence_storage": "redis",
                "evidence_source_count": stats["source_count"],
                "evidence_chunk_count": stats["chunk_count"],
                "evidence_artifact_count": stats["artifact_count"],
                "evidence_artifact_bytes": stats["artifact_bytes"],
            },
        )
    except Exception as exc:
        raise RuntimeError(f"Evidence manifest save verification failed: {exc}") from exc
    return manifest


async def _persist_evidence_manifest(
    redis_client: Any | None,
    task_id: str | None,
    evidence_manifest: dict[str, Any] | None,
    ttl_seconds: int,
) -> None:
    if not redis_client or not task_id or evidence_manifest is None:
        return
    await store_manifest(
        redis_client,
        task_id=task_id,
        manifest=evidence_manifest,
        ttl_seconds=ttl_seconds,
    )


async def _current_task_ttl(redis_client: Any | None, task_id: str | None) -> int:
    fallback_ttl = max(60, _env_int("TASK_TTL_SECONDS", 3 * 24 * 60 * 60))
    if not redis_client or not task_id:
        return fallback_ttl
    try:
        ttl = await redis_client.ttl(task_id)
    except Exception:
        ttl = None
    if isinstance(ttl, int) and ttl > 0:
        return ttl
    try:
        await redis_client.expire(task_id, fallback_ttl)
    except Exception:
        pass
    return fallback_ttl


async def _evidence_success_fields(
    redis_client: Any | None,
    task_id: str | None,
    evidence_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    if not redis_client or not task_id:
        return {}
    if evidence_manifest is None:
        return {
            "evidence_status": "missing",
            "evidence_error": (
                "Evidence manifest was not created by the worker. On Heroku, "
                "ensure the worker dyno is running the same release as the web dyno and rerun the comparison."
            ),
        }
    try:
        stats = await verify_manifest_artifacts(
            redis_client,
            task_id=task_id,
            manifest=evidence_manifest,
        )
        return {
            "evidence_status": "ready",
            "evidence_error": "",
            "evidence_storage": "redis",
            "evidence_source_count": stats["source_count"],
            "evidence_chunk_count": stats["chunk_count"],
            "evidence_artifact_count": stats["artifact_count"],
            "evidence_artifact_bytes": stats["artifact_bytes"],
        }
    except Exception as exc:
        return {
            "evidence_status": "error",
            "evidence_error": f"Could not verify evidence manifest in Redis: {exc}",
        }



async def extract_experiment_specific_paper_text(
    full_paper_text: str,
    experiment_label: str,
    experiment_note: str | None = None,
    client_choice: str = "openai",
    reasoning_effort: str | None = None,
) -> str:
    """Use a generative LLM to isolate intro, relevant experiment, and general discussion text.

    The model is also instructed to inline summaries of referenced experiments in square brackets.
    """
    if not full_paper_text.strip() or not experiment_label.strip():
        return full_paper_text

    note = (experiment_note or "").strip()
    user_prompt = (
        "You will receive the full text of a multi-experiment paper and will be required to extract a subset of its content. "
        f"The relevant experiment identifier to focus on is '{experiment_label}'. \n\n"
        "When the relevant experiment refers to another experiment rather than providing content (e.g., 'our method was identical to Experiment X'), "
        "append in square brackets direct quotes of the referenced experiment relevant to this portion immediately after that reference. "
        "Preserve the paper's wording in all cases. Do not add extra commentary or headings. \n"
        "For example, if the relevant experiment states 'we used the same procedure as Experiment 2', "
        "and Experiment 2's procedure section states 'Participants were shown images for 500ms each', "
        "then the extracted text should be: 'we used the same procedure as Experiment 2 [\"Participants were shown images for 500ms each.\"]'."
        "If a requested section is missing, simply omit it; do not invent content under any circumstances.\n\n"
        "Return ONLY the following paper content, in order, as plain text:\n"
        "1) The Full Introduction section of the paper.\n"
        f"2) The full text of the relevant experiment ({experiment_label}), including its methods, results, "
        "and any discussion specific to that experiment, as well as square brackets quotes of referenced experiments.\n"
        "3) The General Discussion section.\n\n"
    )
    if note:
        user_prompt += f"\n\nAdditional context from the user about the relevant experiment: {note}"
    user_prompt += f"\n\nFull paper text:\n{full_paper_text}"

    def _invoke_llm() -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert academic text extractor. Your job is to extract only the requested sections "
                    "from a multi-experiment paper while preserving the original language, annotating relevant "
                    "information of other studies where referenced in-text in square brackets following that reference."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]
        if client_choice == "openai":
            openai_client = get_openai_client()
            model = _openai_experiment_model()
            normalized_effort = _normalize_reasoning_effort_value(
                reasoning_effort or _env_str("OPENAI_EXPERIMENT_REASONING_EFFORT", "high")
            )
            return _openai_chat_text(
                openai_client,
                model=model,
                messages=messages,
                reasoning_effort=normalized_effort,
            )
        if client_choice == "deepseek":
            deepseek_client = get_deepseek_client()
            response = deepseek_client.chat.completions.create(
                model=_deepseek_model(),
                messages=messages,
                temperature=0,
            )
            raw_content = _message_content_to_text(response.choices[0].message)
            return _strip_deepseek_reasoning(raw_content)
        if client_choice == "groq":
            response = _groq_chat_completion(
                model=_groq_model(),
                messages=messages,
                use_json_mode=False,
            )
            return _message_content_to_text(response.choices[0].message)
        raise ValueError(f"Invalid client selection for experiment extraction: {client_choice}")

    content = await asyncio.to_thread(_invoke_llm)
    cleaned = (content or "").strip()
    if not cleaned:
        raise ValueError("Received empty experiment-focused extraction from the model")
    return cleaned


async def general_preregistration_comparison(
    prereg_path: str,
    prereg_ext: str,
    paper_path: str,
    paper_ext: str,
    client_choice: str,
    parser_choice: str,
    task_id: str | None = None,
    redis_client: Any | None = None,
    selected_dimensions: list[dict[str, str]] | None = None,
    append_previous_output: bool = False,
    pdf_parser: Callable[[str], Awaitable[str]] | None = None,
    dpt_parser: Callable[[str], Awaitable[Any]] | None = None,
    docx_reader: Callable[[str], str] | None = None,
    comparison_runner: Callable[..., ComparisonResult] | None = None,
    reasoning_effort: str | None = None,
    multiple_experiments: str | bool | None = None,
    experiment_number: str | None = None,
    experiment_text: str | None = None,
) -> ComparisonResult:
    processed_count = 0
    if prereg_ext == ".pdf":
        try:
            preregistration_input, prereg_parser_used = await extract_pdf_text(
                prereg_path,
                parser_choice=parser_choice,
                pdf_parser=pdf_parser,
                dpt_parser=dpt_parser,
            )
            if task_id and redis_client and prereg_parser_used != (parser_choice or "grobid").lower():
                await redis_client.hset(
                    task_id,
                    mapping={"status": f"Scanned prereg PDF detected; using {prereg_parser_used} fallback"},
                )
        except Exception as exc:
            if task_id and redis_client:
                await redis_client.hset(
                    task_id,
                    mapping={
                        "state": "FAILURE",
                        "status": f"Preregistration parsing failed: {exc}",
                        "processed_dimensions": processed_count,
                    },
                )
            raise
    else:
        preregistration_input = read_file(prereg_path, prereg_ext)
    paper_input = read_file_as_pdf(paper_path, paper_ext)
    parser_choice_normalized = (parser_choice or "grobid").lower()

    if task_id and redis_client:
        await redis_client.hset(
            task_id,
            mapping={
                "status": f"Parsing paper with {parser_choice_normalized}",
            },
        )
    try:
        if paper_ext == ".pdf":
            extracted_paper_sections, used_parser = await extract_pdf_text(
                paper_input,
                parser_choice=parser_choice_normalized,
                pdf_parser=pdf_parser,
                dpt_parser=dpt_parser,
            )
            if task_id and redis_client and used_parser != parser_choice_normalized:
                await redis_client.hset(
                    task_id,
                    mapping={"status": f"Scanned PDF detected; using {used_parser} fallback"},
                )
        elif paper_ext == ".docx":
            reader = docx_reader or extract_text_from_docx
            extracted_paper_sections = reader(paper_input)
        else:
            raise ValueError("Problem parsing paper input - try a .pdf for optimal results.")
    except Exception as exc:
        if task_id and redis_client:
            await redis_client.hset(
                task_id,
                mapping={
                    "state": "FAILURE",
                    "status": f"Parsing failed: {exc}",
                    "processed_dimensions": processed_count,
                },
            )
        raise

    result_obj = ComparisonResult(items=[])
    dimensions_to_compare = _resolve_dimensions(selected_dimensions)
    dimension_names = [item["dimension"] for item in dimensions_to_compare]
    total_dimensions = len(dimensions_to_compare)

    experiment_label = (experiment_number or "").strip()
    experiment_note = (experiment_text or "").strip()
    has_multiple_experiments = False
    if isinstance(multiple_experiments, str):
        has_multiple_experiments = multiple_experiments.strip().lower() == "yes"
    else:
        has_multiple_experiments = bool(multiple_experiments)

    if has_multiple_experiments and experiment_label:
        if task_id and redis_client:
            await redis_client.hset(
                task_id,
                mapping={
                    "state": "IN_PROGRESS",
                    "result_json": result_obj.model_dump_json(),
                    "total_dimensions": total_dimensions,
                    "processed_dimensions": 0,
                    "dimensions": json.dumps(dimension_names),
                    "status": f"Isolating Experiment {experiment_label} text with the model",
                },
            )
        try:
            canonical_paper_text = await extract_experiment_specific_paper_text(
                extracted_paper_sections,
                experiment_label=experiment_label,
                experiment_note=experiment_note,
                client_choice=client_choice,
                reasoning_effort=reasoning_effort,
            )
            extracted_paper_sections = canonical_paper_text
            if task_id and redis_client:
                await redis_client.hset(
                    task_id,
                    mapping={
                        "state": "IN_PROGRESS",
                        "result_json": result_obj.model_dump_json(),
                        "total_dimensions": total_dimensions,
                        "processed_dimensions": 0,
                        "dimensions": json.dumps(dimension_names),
                        "status": (
                            f"Experiment {experiment_label} isolated; embedding preregistration and paper"
                        ),
                    },
                )
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.warning(
                "Experiment-focused paper extraction failed; using full paper text",
                extra={"task_id": task_id, "experiment_label": experiment_label},
                exc_info=exc,
            )
            if task_id and redis_client:
                await redis_client.hset(
                    task_id,
                    mapping={
                        "status": (
                            f"Continuing without experiment-specific extraction for Experiment {experiment_label}"
                        )
                    },
                )

    runner = comparison_runner or run_comparison
    corpus_cache: dict[str, EmbeddingCorpus] = {}
    evidence_manifest: dict[str, Any] | None = None
    evidence_ttl_seconds = await _current_task_ttl(redis_client, task_id)
    if task_id and redis_client:
        await redis_client.hset(
            task_id,
            mapping={
                "status": "Preparing evidence viewer sources",
                "evidence_status": "preparing",
                "evidence_error": "",
            },
        )
        prereg_payload = build_file_evidence_source(
            source_id="registration",
            label="Preregistration",
            file_path=prereg_path,
            file_ext=prereg_ext,
            text=preregistration_input,
            chunk_prefix="PREREG",
            metadata={"role": "registration", "comparison_type": "general_preregistration"},
            max_chunk_tokens=_embedding_max_chunk_tokens(),
            embedding_model=_embedding_model(),
        )
        if has_multiple_experiments and experiment_label:
            raw_bytes = None
            raw_content_type = None
            raw_filename = None
            try:
                paper_raw_path = Path(paper_path)
                raw_bytes = paper_raw_path.read_bytes()
                raw_content_type = "application/pdf" if paper_ext == ".pdf" else None
                raw_filename = paper_raw_path.name
            except Exception:
                pass
            paper_payload = build_text_evidence_source(
                source_id="paper",
                label="Paper Evidence Text",
                text=extracted_paper_sections,
                chunk_prefix="PAPER",
                kind="text",
                metadata={
                    "role": "paper",
                    "comparison_type": "general_preregistration",
                    "fallback_reason": "Experiment-specific text was isolated before embedding",
                },
                raw_bytes=raw_bytes,
                raw_content_type=raw_content_type,
                raw_filename=raw_filename,
                max_chunk_tokens=_embedding_max_chunk_tokens(),
                embedding_model=_embedding_model(),
            )
        else:
            paper_payload = build_file_evidence_source(
                source_id="paper",
                label="Paper",
                file_path=paper_input,
                file_ext=paper_ext,
                text=extracted_paper_sections,
                chunk_prefix="PAPER",
                metadata={"role": "paper", "comparison_type": "general_preregistration"},
                max_chunk_tokens=_embedding_max_chunk_tokens(),
                embedding_model=_embedding_model(),
            )
        preregistration_input = prereg_payload.get("text") or preregistration_input
        extracted_paper_sections = paper_payload.get("text") or extracted_paper_sections
        _add_evidence_corpus_to_cache(
            corpus_cache,
            role="prereg",
            chunk_prefix="PREREG",
            source_payload=prereg_payload,
        )
        _add_evidence_corpus_to_cache(
            corpus_cache,
            role="paper",
            chunk_prefix="PAPER",
            source_payload=paper_payload,
        )
        evidence_manifest = await _store_evidence_manifest(
            redis_client=redis_client,
            task_id=task_id,
            comparison_type="general_preregistration",
            source_payloads=[prereg_payload, paper_payload],
            ttl_seconds=evidence_ttl_seconds,
        )
    logger.info(
        "general_preregistration_comparison start",
        extra={
            "client_choice": client_choice,
            "reasoning_effort": reasoning_effort,
            "total_dimensions": total_dimensions,
        },
    )

    if task_id and redis_client:
        await redis_client.hset(
            task_id,
            mapping={
                "state": "IN_PROGRESS",
                "result_json": result_obj.model_dump_json(),
                "total_dimensions": total_dimensions,
                "processed_dimensions": 0,
                "dimensions": json.dumps(dimension_names),
                "status": "Embedding preregistration and paper",
            },
        )

    try:
        for index, dimension_info in enumerate(dimensions_to_compare, start=1):
            if not isinstance(dimension_info, dict):
                continue
            dimension_name = (dimension_info.get("dimension") or dimension_info.get("name") or "").strip()
            if not dimension_name:
                continue
            dimension_definition = (dimension_info.get("definition") or "").strip()
            previous_responses: list[ComparisonItem] | None = None
            if append_previous_output and result_obj.items:
                previous_responses = list(result_obj.items)
                logger.info(
                    "Appending %d prior dimension responses for '%s' in preregistration flow",
                    len(previous_responses),
                    dimension_name,
                )
            logger.info(
                "general_preregistration_comparison running dimension",
                extra={
                    "dimension": dimension_name,
                    "reasoning_effort": reasoning_effort,
                },
            )
            if task_id and redis_client:
                await redis_client.hset(
                    task_id,
                    mapping={"status": f"Embedding and retrieving for '{dimension_name}'"},
                )
            comparison = await asyncio.to_thread(
                runner,
                preregistration_input,
                extracted_paper_sections,
                client_choice,
                dimension_name,
                dimension_definition=dimension_definition,
                corpus_cache=corpus_cache,
                reasoning_effort=reasoning_effort,
                previous_dimension_responses=previous_responses,
                comparison_context="preregistration",
                evidence_manifest=evidence_manifest,
            )
            result_obj.items.extend(comparison.items)
            processed_count = index
            await _persist_evidence_manifest(
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
                        "result_json": result_obj.model_dump_json(),
                        "processed_dimensions": index,
                        "total_dimensions": total_dimensions,
                        "status": f"Processed {index}/{total_dimensions}: {dimension_name}",
                    },
                )
    except Exception as exc:
        if task_id and redis_client:
            await redis_client.hset(
                task_id,
                mapping={
                    "state": "FAILURE",
                    "status": f"Processing failed: {exc}",
                    "result_json": result_obj.model_dump_json(),
                    "processed_dimensions": processed_count,
                    "total_dimensions": total_dimensions,
                },
            )
        raise

    if task_id and redis_client:
        evidence_fields = await _evidence_success_fields(redis_client, task_id, evidence_manifest)
        await redis_client.hset(
            task_id,
            mapping={
                "state": "SUCCESS",
                "result_json": result_obj.model_dump_json(),
                "total_dimensions": total_dimensions,
                "processed_dimensions": total_dimensions,
                "dimensions": json.dumps(dimension_names),
                "status": "Report complete",
                **evidence_fields,
            },
        )
    return result_obj


async def clinical_trial_comparison(
    registration_id: str,
    paper_path: str,
    paper_ext: str,
    client_choice: str,
    task_id: str | None = None,
    redis_client=None,
    parser_choice: str = "grobid",
    pdf_parser: Callable[[str], Awaitable[str]] | None = None,
    dpt_parser: Callable[[str], Awaitable[Any]] | None = None,
    docx_reader: Callable[[str], str] | None = None,
    nct_extractor: Callable[[str], str] | None = None,
    trial_fetcher: Callable[[str], dict[str, dict[str, str]]] | None = None,
    comparison_runner: Callable[..., ComparisonResult] | None = None,
    selected_dimensions: list[dict[str, str]] | None = None,
    append_previous_output: bool = False,
    reasoning_effort: str | None = None,
) -> ComparisonResult:
    logger.info("Started clinical trial comparison", extra={"task_id": task_id})
    extract_nct = nct_extractor or extract_nct_id
    nct_id = extract_nct(registration_id)
    trial_metadata: dict[str, Any] = {}
    if trial_fetcher is None:
        nested_trial, trial_metadata = extract_nested_trial_with_metadata(nct_id)
    else:
        nested_trial = trial_fetcher(nct_id)
    prereg_text = "\n\n".join(
        f"{dimension}\n\n" + "\n".join(f"{sub}\n{text}" for sub, text in subcomponents.items())
        for dimension, subcomponents in nested_trial.items()
    )
    dimensions_to_compare = _resolve_dimensions(selected_dimensions, CLINICAL_DEFAULT_DIMENSIONS)
    processed_count = 0
    paper_input = read_file_as_pdf(paper_path, paper_ext)
    parser_choice_normalized = (parser_choice or "grobid").lower()
    if redis_client and task_id:
        await redis_client.hset(
            task_id,
            mapping={"status": f"Parsing paper with {parser_choice_normalized}"},
        )
    try:
        if paper_ext == ".pdf":
            extracted_paper_sections, used_parser = await extract_pdf_text(
                paper_input,
                parser_choice=parser_choice_normalized,
                pdf_parser=pdf_parser,
                dpt_parser=dpt_parser,
            )
            if task_id and redis_client and used_parser != parser_choice_normalized:
                await redis_client.hset(
                    task_id,
                    mapping={"status": f"Scanned PDF detected; using {used_parser} fallback"},
                )
        elif paper_ext == ".docx":
            reader = docx_reader or extract_text_from_docx
            extracted_paper_sections = reader(paper_input)
        else:
            raise ValueError("Problem parsing paper input - try a .pdf for optimal results.")
    except Exception as exc:
        if redis_client and task_id:
            await redis_client.hset(
                task_id,
                mapping={
                    "state": "FAILURE",
                    "status": f"Parsing failed: {exc}",
                    "processed_dimensions": processed_count,
                },
            )
        raise

    result_obj = ComparisonResult(items=[])
    dimension_names = [
        (item.get("dimension") or "").strip()
        for item in dimensions_to_compare
        if (item.get("dimension") or "").strip()
    ]
    total_dimensions = len(dimension_names)
    if redis_client and task_id:
        await redis_client.hset(
            task_id,
            mapping={
                "state": "IN_PROGRESS",
                "result_json": result_obj.model_dump_json(),
                "total_dimensions": total_dimensions,
                "processed_dimensions": 0,
                "dimensions": json.dumps(dimension_names),
                "status": "Embedding preregistration and paper",
            },
        )
    runner = comparison_runner or run_comparison
    corpus_cache: dict[str, EmbeddingCorpus] = {}
    evidence_manifest: dict[str, Any] | None = None
    evidence_ttl_seconds = await _current_task_ttl(redis_client, task_id)
    if task_id and redis_client:
        await redis_client.hset(
            task_id,
            mapping={
                "status": "Preparing evidence viewer sources",
                "evidence_status": "preparing",
                "evidence_error": "",
            },
        )
        registration_payload = build_json_evidence_source(
            source_id="registration",
            label="ClinicalTrials.gov Registration",
            data=nested_trial,
            chunk_prefix="PREREG",
            metadata={"role": "registration", "comparison_type": "clinical_trials", **trial_metadata},
            max_chunk_tokens=_embedding_max_chunk_tokens(),
            embedding_model=_embedding_model(),
        )
        paper_payload = build_file_evidence_source(
            source_id="paper",
            label="Paper",
            file_path=paper_input,
            file_ext=paper_ext,
            text=extracted_paper_sections,
            chunk_prefix="PAPER",
            metadata={"role": "paper", "comparison_type": "clinical_trials"},
            max_chunk_tokens=_embedding_max_chunk_tokens(),
            embedding_model=_embedding_model(),
        )
        prereg_text = registration_payload.get("text") or prereg_text
        extracted_paper_sections = paper_payload.get("text") or extracted_paper_sections
        _add_evidence_corpus_to_cache(
            corpus_cache,
            role="prereg",
            chunk_prefix="PREREG",
            source_payload=registration_payload,
        )
        _add_evidence_corpus_to_cache(
            corpus_cache,
            role="paper",
            chunk_prefix="PAPER",
            source_payload=paper_payload,
        )
        evidence_manifest = await _store_evidence_manifest(
            redis_client=redis_client,
            task_id=task_id,
            comparison_type="clinical_trials",
            source_payloads=[registration_payload, paper_payload],
            ttl_seconds=evidence_ttl_seconds,
        )
    try:
        for index, dimension_info in enumerate(dimensions_to_compare, start=1):
            dimension = dimension_info.get("dimension", "").strip()
            if not dimension:
                continue
            dimension_definition = (dimension_info.get("definition") or "").strip()
            previous_responses: list[ComparisonItem] | None = None
            if append_previous_output and result_obj.items:
                previous_responses = list(result_obj.items)
                logger.info(
                    "Appending %d prior dimension responses for '%s' in clinical trial flow",
                    len(previous_responses),
                    dimension,
                )
            if redis_client and task_id:
                await redis_client.hset(
                    task_id,
                    mapping={"status": f"Embedding and retrieving for '{dimension}'"},
                )
            comparison = await asyncio.to_thread(
                runner,
                prereg_text,
                extracted_paper_sections,
                client_choice,
                dimension,
                dimension_definition=dimension_definition,
                corpus_cache=corpus_cache,
                reasoning_effort=reasoning_effort,
                previous_dimension_responses=previous_responses,
                comparison_context="clinical_trial",
                evidence_manifest=evidence_manifest,
            )
            result_obj.items.extend(comparison.items)
            processed_count = index
            await _persist_evidence_manifest(
                redis_client,
                task_id,
                evidence_manifest,
                evidence_ttl_seconds,
            )
            if redis_client and task_id:
                await redis_client.hset(
                    task_id,
                    mapping={
                        "state": "IN_PROGRESS",
                        "result_json": result_obj.model_dump_json(),
                        "processed_dimensions": index,
                        "total_dimensions": total_dimensions,
                        "status": f"LLM judgement complete for '{dimension}' ({index}/{total_dimensions})",
                    },
                )
    except Exception as exc:
        if redis_client and task_id:
            await redis_client.hset(
                task_id,
                mapping={
                    "state": "FAILURE",
                    "status": f"Processing failed: {exc}",
                    "result_json": result_obj.model_dump_json(),
                    "processed_dimensions": processed_count,
                    "total_dimensions": total_dimensions,
                },
            )
        raise

    if redis_client and task_id:
        evidence_fields = await _evidence_success_fields(redis_client, task_id, evidence_manifest)
        await redis_client.hset(
            task_id,
            mapping={
                "state": "SUCCESS",
                "result_json": result_obj.model_dump_json(),
                "total_dimensions": total_dimensions,
                "processed_dimensions": total_dimensions,
                "dimensions": json.dumps(dimension_names),
                "status": "Report complete",
                **evidence_fields,
            },
        )
    return result_obj


async def animals_trial_comparison(
    registration_id: str,
    paper_path: str,
    paper_ext: str,
    client_choice: str,
    registration_csv_path: str | None = None,
    task_id: str | None = None,
    redis_client=None,
    parser_choice: str = "grobid",
    pdf_parser: Callable[[str], Awaitable[str]] | None = None,
    dpt_parser: Callable[[str], Awaitable[Any]] | None = None,
    docx_reader: Callable[[str], str] | None = None,
    comparison_runner: Callable[..., ComparisonResult] | None = None,
    selected_dimensions: list[dict[str, str]] | None = None,
    append_previous_output: bool = False,
    reasoning_effort: str | None = None,
) -> ComparisonResult:
    logger.info(
        "Started animals trial comparison",
        extra={"task_id": task_id, "pct_id": registration_id},
    )
    if not registration_csv_path:
        raise ValueError(
            "Animal trial comparisons currently require a registration CSV with a pct_id column."
        )

    prereg_text = _load_pct_registration_text(registration_id, registration_csv_path)

    dimensions_to_compare = _resolve_dimensions(selected_dimensions, PRECLINICAL_DEFAULT_DIMENSIONS)
    processed_count = 0
    paper_input = read_file_as_pdf(paper_path, paper_ext)
    parser_choice_normalized = (parser_choice or "grobid").lower()
    if redis_client and task_id:
        await redis_client.hset(
            task_id,
            mapping={"status": f"Parsing paper with {parser_choice_normalized}"},
        )
    try:
        if paper_ext == ".pdf":
            extracted_paper_sections, used_parser = await extract_pdf_text(
                paper_input,
                parser_choice=parser_choice_normalized,
                pdf_parser=pdf_parser,
                dpt_parser=dpt_parser,
            )
            if task_id and redis_client and used_parser != parser_choice_normalized:
                await redis_client.hset(
                    task_id,
                    mapping={"status": f"Scanned PDF detected; using {used_parser} fallback"},
                )
        elif paper_ext == ".docx":
            reader = docx_reader or extract_text_from_docx
            extracted_paper_sections = reader(paper_input)
        else:
            raise ValueError("Problem parsing paper input - try a .pdf for optimal results.")
    except Exception as exc:
        if redis_client and task_id:
            await redis_client.hset(
                task_id,
                mapping={
                    "state": "FAILURE",
                    "status": f"Parsing failed: {exc}",
                    "processed_dimensions": processed_count,
                },
            )
        raise

    result_obj = ComparisonResult(items=[])
    dimension_names = [
        (item.get("dimension") or "").strip()
        for item in dimensions_to_compare
        if (item.get("dimension") or "").strip()
    ]
    total_dimensions = len(dimension_names)
    if redis_client and task_id:
        await redis_client.hset(
            task_id,
            mapping={
                "state": "IN_PROGRESS",
                "result_json": result_obj.model_dump_json(),
                "total_dimensions": total_dimensions,
                "processed_dimensions": 0,
                "dimensions": json.dumps(dimension_names),
            },
        )

    runner = comparison_runner or run_comparison
    corpus_cache: dict[str, EmbeddingCorpus] = {}
    evidence_manifest: dict[str, Any] | None = None
    evidence_ttl_seconds = await _current_task_ttl(redis_client, task_id)
    if task_id and redis_client:
        await redis_client.hset(
            task_id,
            mapping={
                "status": "Preparing evidence viewer sources",
                "evidence_status": "preparing",
                "evidence_error": "",
            },
        )
        registration_payload = build_text_evidence_source(
            source_id="registration",
            label="PCT Registration",
            text=prereg_text,
            chunk_prefix="PREREG",
            kind="text",
            metadata={"role": "registration", "comparison_type": "animals_trials", "registration_id": registration_id},
            raw_bytes=prereg_text.encode("utf-8"),
            raw_content_type="text/plain; charset=utf-8",
            raw_filename=f"{registration_id or 'registration'}.txt",
            max_chunk_tokens=_embedding_max_chunk_tokens(),
            embedding_model=_embedding_model(),
        )
        paper_payload = build_file_evidence_source(
            source_id="paper",
            label="Paper",
            file_path=paper_input,
            file_ext=paper_ext,
            text=extracted_paper_sections,
            chunk_prefix="PAPER",
            metadata={"role": "paper", "comparison_type": "animals_trials"},
            max_chunk_tokens=_embedding_max_chunk_tokens(),
            embedding_model=_embedding_model(),
        )
        prereg_text = registration_payload.get("text") or prereg_text
        extracted_paper_sections = paper_payload.get("text") or extracted_paper_sections
        _add_evidence_corpus_to_cache(
            corpus_cache,
            role="prereg",
            chunk_prefix="PREREG",
            source_payload=registration_payload,
        )
        _add_evidence_corpus_to_cache(
            corpus_cache,
            role="paper",
            chunk_prefix="PAPER",
            source_payload=paper_payload,
        )
        evidence_manifest = await _store_evidence_manifest(
            redis_client=redis_client,
            task_id=task_id,
            comparison_type="animals_trials",
            source_payloads=[registration_payload, paper_payload],
            ttl_seconds=evidence_ttl_seconds,
        )
    try:
        for index, dimension_info in enumerate(dimensions_to_compare, start=1):
            dimension = dimension_info.get("dimension", "").strip()
            if not dimension:
                continue
            dimension_definition = (dimension_info.get("definition") or "").strip()
            previous_responses: list[ComparisonItem] | None = None
            if append_previous_output and result_obj.items:
                previous_responses = list(result_obj.items)
                logger.info(
                    "Appending %d prior dimension responses for '%s' in animals trial flow",
                    len(previous_responses),
                    dimension,
                )
            if redis_client and task_id:
                await redis_client.hset(
                    task_id,
                    mapping={"status": f"Embedding and retrieving for '{dimension}'"},
                )
            comparison = await asyncio.to_thread(
                runner,
                prereg_text,
                extracted_paper_sections,
                client_choice,
                dimension,
                dimension_definition=dimension_definition,
                corpus_cache=corpus_cache,
                reasoning_effort=reasoning_effort,
                previous_dimension_responses=previous_responses,
                comparison_context="clinical_trial",
                evidence_manifest=evidence_manifest,
            )
            result_obj.items.extend(comparison.items)
            processed_count = index
            await _persist_evidence_manifest(
                redis_client,
                task_id,
                evidence_manifest,
                evidence_ttl_seconds,
            )
            if redis_client and task_id:
                await redis_client.hset(
                    task_id,
                    mapping={
                        "state": "IN_PROGRESS",
                        "result_json": result_obj.model_dump_json(),
                        "processed_dimensions": index,
                        "total_dimensions": total_dimensions,
                        "status": f"LLM judgement complete for '{dimension}' ({index}/{total_dimensions})",
                    },
                )
    except Exception as exc:
        if redis_client and task_id:
            await redis_client.hset(
                task_id,
                mapping={
                    "state": "FAILURE",
                    "status": f"Processing failed: {exc}",
                    "result_json": result_obj.model_dump_json(),
                    "processed_dimensions": processed_count,
                    "total_dimensions": total_dimensions,
                },
            )
        raise

    if redis_client and task_id:
        evidence_fields = await _evidence_success_fields(redis_client, task_id, evidence_manifest)
        await redis_client.hset(
            task_id,
            mapping={
                "state": "SUCCESS",
                "result_json": result_obj.model_dump_json(),
                "total_dimensions": total_dimensions,
                "processed_dimensions": total_dimensions,
                "dimensions": json.dumps(dimension_names),
                "status": "Report complete",
                **evidence_fields,
            },
        )
    return result_obj


def _message_content_to_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for element in content:
            if hasattr(element, "text"):
                part_type = (getattr(element, "type", "") or "").lower()
                if part_type in {"reasoning", "thinking", "tool_calls"}:
                    continue
                text_value = getattr(element, "text", None)
                if text_value:
                    parts.append(str(text_value))
                continue
            if isinstance(element, dict):
                part_type = (element.get("type") or "").lower()
                if part_type in {"reasoning", "thinking", "tool_calls"}:
                    continue
                text_value = element.get("text")
                if text_value:
                    parts.append(str(text_value))
            elif isinstance(element, str):
                parts.append(element)
        return "".join(parts).strip()
    if content is None:
        return ""
    return str(content)


def _strip_deepseek_reasoning(content: str) -> str:
    if not content:
        return content
    closing_tag = "</think>"
    closing_index = content.find(closing_tag)
    if closing_index != -1:
        content = content[closing_index + len(closing_tag) :]
    return content.lstrip()


def _extract_json_payload(raw_text: str) -> str:
    if not raw_text:
        return raw_text
    text = raw_text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    return text.strip()


def _normalize_comparison_payload(payload: Any) -> dict[str, Any]:
    """Attempt to coerce LLM output to the expected ComparisonItem shape.

    - Accept a top-level list and take the first object.
    - Accept a top-level object with an 'items' list and take its first element.
    - Ensure all expected string fields exist, coercing lists/dicts to strings.
    - Join multiple quotes into a single string for the two quotes fields.
    """
    # Drill into common wrappers
    candidate = payload
    if isinstance(candidate, list):
        candidate = next((x for x in candidate if isinstance(x, dict)), {})
    if isinstance(candidate, dict) and "items" in candidate and isinstance(candidate["items"], list):
        inner = next((x for x in candidate["items"] if isinstance(x, dict)), None)
        if inner is not None:
            candidate = inner

    if not isinstance(candidate, dict):
        try:
            # As a last resort try to parse a stringified JSON inside
            if isinstance(candidate, str):
                maybe = _extract_json_payload(candidate)
                candidate = json.loads(maybe)
        except Exception:
            candidate = {}

    expected_keys = [
        "dimension",
        "paper_content_quotes",
        "paper_content_summary",
        "registration_content_quotes",
        "registration_content_summary",
        "deviation_judgement",
        "deviation_information",
    ]

    normalized: dict[str, Any] = {}
    for key in expected_keys:
        value = candidate.get(key)
        if key in ("paper_content_quotes", "registration_content_quotes"):
            if value is None:
                normalized[key] = ""
            elif isinstance(value, list):
                normalized[key] = "\n\n".join(str(x).strip() for x in value if f"{x}".strip())
            elif isinstance(value, dict):
                # try to pull list-like content
                parts: list[str] = []
                for k in ("quotes", "items", "values", "data"):
                    v = value.get(k)
                    if isinstance(v, list):
                        parts.extend(str(x).strip() for x in v if f"{x}".strip())
                if parts:
                    normalized[key] = "\n\n".join(parts)
                else:
                    try:
                        normalized[key] = json.dumps(value, ensure_ascii=False)
                    except Exception:
                        normalized[key] = str(value)
            else:
                normalized[key] = str(value)
        else:
            if value is None:
                normalized[key] = ""
            elif isinstance(value, list):
                normalized[key] = " ".join(str(x).strip() for x in value if f"{x}".strip())
            elif isinstance(value, dict):
                vals = [str(v).strip() for v in value.values() if isinstance(v, (str, int, float)) and f"{v}".strip()]
                if vals:
                    normalized[key] = " ".join(vals)
                else:
                    try:
                        normalized[key] = json.dumps(value, ensure_ascii=False)
                    except Exception:
                        normalized[key] = str(value)
            else:
                normalized[key] = str(value)

    return normalized


def _search_first_text_fragment(payload: Any) -> str:
    if isinstance(payload, str):
        candidate = payload.strip()
        if candidate:
            return candidate
        return ""
    if isinstance(payload, dict):
        prioritized_keys = ("content", "text", "output_text", "answer", "message", "value")
        for key in prioritized_keys:
            if key in payload:
                found = _search_first_text_fragment(payload[key])
                if found:
                    return found
        for value in payload.values():
            found = _search_first_text_fragment(value)
            if found:
                return found
        return ""
    if isinstance(payload, list):
        for item in payload:
            found = _search_first_text_fragment(item)
            if found:
                return found
    return ""


ComparisonContext = Literal["preregistration", "clinical_trial"]


def run_comparison(
    preregistration_input: str,
    extracted_paper_sections: str,
    client_choice: str,
    dimension_query: str,
    dimension_definition: str | None = None,
    top_k: int | None = None,
    embeddings_prefix: str | None = None,
    append_rows: bool = False,
    corpus_cache: dict[str, EmbeddingCorpus] | None = None,
    previous_dimension_responses: list[ComparisonItem] | None = None,
    reasoning_effort: str | None = None,
    comparison_context: ComparisonContext = "clinical_trial",
    evidence_manifest: dict[str, Any] | None = None,
) -> ComparisonResult:
    prereg_path = f"{embeddings_prefix}_prereg.pkl" if embeddings_prefix else None
    paper_path = f"{embeddings_prefix}_paper.pkl" if embeddings_prefix else None
    logger.info(
        "run_comparison invoked",
        extra={
            "dimension": dimension_query,
            "client": client_choice,
            "reasoning_effort": reasoning_effort,
            "comparison_context": comparison_context,
        },
    )

    cache = corpus_cache if corpus_cache is not None else {}
    prereg_key = f"prereg:{hashlib.sha256(preregistration_input.encode('utf-8')).hexdigest()}"
    paper_key = f"paper:{hashlib.sha256(extracted_paper_sections.encode('utf-8')).hexdigest()}"

    max_segments = _max_embedding_segments()
    embedding_model = _embedding_model()
    max_chunk_tokens = _embedding_max_chunk_tokens()

    prereg_corpus = cache.get(prereg_key)
    if prereg_corpus is None:
        prereg_corpus = build_corpus(
            preregistration_input,
            model=embedding_model,
            embeddings_path=prereg_path,
            chunk_prefix="PREREG",
            max_segments=max_segments,
            max_chunk_tokens=max_chunk_tokens,
        )
        cache[prereg_key] = prereg_corpus

    paper_corpus = cache.get(paper_key)
    if paper_corpus is None:
        paper_corpus = build_corpus(
            extracted_paper_sections,
            model=embedding_model,
            embeddings_path=paper_path,
            chunk_prefix="PAPER",
            max_segments=max_segments,
            max_chunk_tokens=max_chunk_tokens,
        )
        cache[paper_key] = paper_corpus

    # Drop references to large raw texts once corpora are built to ease memory pressure on small dynos.
    preregistration_input = ""
    extracted_paper_sections = ""

    # Definitions are resolved by the caller (_resolve_dimensions): explicitly
    # selected defaults or user-specified values. No name-based fallback here.
    definition_for_query = (dimension_definition or "").strip()
    augmented_query = f"{dimension_query}. {definition_for_query}" if definition_for_query else dimension_query

    prereg_top_k = top_k if top_k is not None else _compute_top_k(len(prereg_corpus.segments))
    paper_top_k = top_k if top_k is not None else _compute_top_k(len(paper_corpus.segments))

    query_embedding = get_embedding(augmented_query, model=embedding_model)

    candidate_factor = 3
    prereg_candidate_k = min(
        len(prereg_corpus.segments), max(prereg_top_k * candidate_factor, prereg_top_k + 5)
    )
    paper_candidate_k = min(
        len(paper_corpus.segments), max(paper_top_k * candidate_factor, paper_top_k + 5)
    )

    prereg_candidates = retrieve_relevant_chunks(
        query_embedding, prereg_corpus, top_k=prereg_candidate_k
    )
    paper_candidates = retrieve_relevant_chunks(
        query_embedding, paper_corpus, top_k=paper_candidate_k
    )

    prereg_top_rows = prereg_candidates[:prereg_top_k]
    paper_top_rows = paper_candidates[:paper_top_k]

    def _sort_by_numeric_id(rows: list[tuple[str, str, float]]) -> list[tuple[str, str, float]]:
        def _id_num(cid: str) -> int:
            try:
                # Expect format PREFIX_#### or PREFIX_###### etc.
                parts = cid.split("_")
                return int(parts[-1])
            except Exception:
                return 0

        return sorted(rows, key=lambda x: _id_num(x[0]))

    prereg_top_rows = _sort_by_numeric_id(prereg_top_rows)
    paper_top_rows = _sort_by_numeric_id(paper_top_rows)

    if evidence_manifest is not None:
        chunks = evidence_manifest.setdefault("chunks", {})
        for cid, _text, sim in prereg_top_rows + paper_top_rows:
            chunk_info = chunks.get(cid)
            if not isinstance(chunk_info, dict):
                continue
            score_map = chunk_info.setdefault("relevance_scores_by_dimension", {})
            score_map[dimension_query] = float(sim)
            current_max = chunk_info.get("max_relevance_score")
            if not isinstance(current_max, (int, float)) or sim > current_max:
                chunk_info["max_relevance_score"] = float(sim)

    prereg_top = [f"[{cid}, relevance_score={sim:.3f}] {text}" for cid, text, sim in prereg_top_rows]
    paper_top = [f"[{cid}, relevance_score={sim:.3f}] {text}" for cid, text, sim in paper_top_rows]

    history_context = ""
    if previous_dimension_responses:
        dimension_titles = [
            (item.dimension or "").strip()
            for item in previous_dimension_responses
            if (item.dimension or "").strip()
        ]
        history_lines: list[str] = []
        if dimension_titles:
            history_lines.append(
                "Previously, you were asked to provide information about the following dimensions: "
                + ", ".join(dimension_titles)
                + "."
            )
        for item in previous_dimension_responses:
            label = (item.dimension or "this dimension").strip() or "this dimension"
            dumped = json.dumps(item.model_dump())
            history_lines.append(f"For {label}, you gave this output: {dumped}")
        history_context = "\n".join(history_lines).strip()
        logger.debug(
            "History context for '%s' includes dimensions %s",
            dimension_query,
            dimension_titles or ["<unknown>"],
        )
        logger.debug("Full history context for '%s':\n%s", dimension_query, history_context)

    if comparison_context == "preregistration":
        intro_line = (
            "Critically compare the following study preregistration with content from its corresponding published paper based on the below-specified specified study dimension."
        )
    else:
        intro_line = (
            "Critically compare the following clinical trial registration with content from its corresponding published paper based on the below-specified specified study dimension."
        )

    master_prompt = (
        f"{intro_line}\n\n"
        "You have two goals. First, identify and extract quotes from the sources that are relevant to the specified dimension from both the registration and the paper. You will also provide a concise summary of this information for both the registration and paper."
        " Second, make a judgement as to whether the content of the registration and paper relative to the specified dimension are consistent or not."
        " You are looking closely for any deviation or divergence between the paper and the registration, particularly those that might cause conceptual, statistical, or interpretative issues with the study.\n\n"
        f"The dimension along which you should compare the registration and paper is: '{dimension_query}'; this is defined as "
        f"{definition_for_query if definition_for_query else 'not provided by the user.'}\n\n"
        "Use ONLY the provided evidence excerpts. Each excerpt is labeled with an ID in square brackets.\n\n"
        "Registration excerpts:\n"
        f"{' '.join(prereg_top)}\n\n"
        "Paper excerpts:\n"
        f"{' '.join(paper_top)}\n"
        "Your output must be a single JSON object (no arrays unless specified, no surrounding text, no code fences) with the following fields: "
        "'dimension', 'paper_content_quotes', 'paper_content_summary', 'registration_content_quotes', 'registration_content_summary', "
        "'deviation_judgement', and 'deviation_information'. Each field MUST be a string.\n"
        "- For 'paper_content_quotes' and 'registration_content_quotes', include direct quotes from the provided excerpts, and keep the evidence IDs (e.g., [PAPER_0001]) in the text. Join multiple quotes with two newlines (\\n\\n). Do NOT return an array.\n"
        "- For the summaries and deviation information, also cite the evidence IDs you relied upon.\n"
        "- 'deviation_judgement' should be 'yes', 'no', or 'missing' if you lack enough evidence.\n"
        "If evidence is insufficient to judge, set deviation_judgement to 'missing' and explain briefly.\n"
    )
    if history_context:
        master_prompt = history_context + "\n\n" + master_prompt

    messages = [
        {
            "role": "system",
            "content": (
                "You are RegCheck, a large language model which excels in comparing registered protocols for scientific studies "
                "to the corresponding published papers. You check and flag both consistencies and deviations between the documents "
                "in an easy-to-read format. You are rigorous and comprehensive in your comparisons, and have a very critical eye "
                "for detail."
            ),
        },
        {"role": "user", "content": master_prompt},
    ]

    if client_choice == "openai":
        openai_client = get_openai_client()
        model = _openai_model()
        normalized_effort = (reasoning_effort or "medium").strip().lower()
        if normalized_effort not in {"low", "medium", "high"}:
            normalized_effort = "medium"
        try:
            response = openai_client.chat.completions.parse(
                model=model,
                messages=messages,
                reasoning_effort=normalized_effort,
                response_format=ComparisonItem,
            )
            result_json = response.choices[0].message.content
        except Exception as exc:
            logger.info(
                "OpenAI parse() failed; falling back to JSON mode",
                extra={"model": model},
                exc_info=exc,
            )
            result_json = _openai_chat_json(
                openai_client,
                model=model,
                messages=messages,
                reasoning_effort=normalized_effort,
            )
    elif client_choice == "deepseek":
        deepseek_client = get_deepseek_client()
        response = deepseek_client.chat.completions.create(
            model=_deepseek_model(),
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        message = response.choices[0].message
        raw_content = _message_content_to_text(message)
        if not raw_content:
            message_dump = None
            if hasattr(message, "model_dump_json"):
                try:
                    message_dump = json.loads(message.model_dump_json())
                except Exception:
                    message_dump = message.model_dump()
            elif hasattr(message, "model_dump"):
                message_dump = message.model_dump()
            raw_content = _search_first_text_fragment(message_dump)
            if not raw_content:
                response_dump = None
                if hasattr(response, "model_dump_json"):
                    try:
                        response_dump = json.loads(response.model_dump_json())
                    except Exception:
                        response_dump = response.model_dump()
                elif hasattr(response, "model_dump"):
                    response_dump = response.model_dump()
                raw_content = _search_first_text_fragment(response_dump)
            if not raw_content:
                logger.warning(
                    "DeepSeek response returned empty content",
                    extra={
                        "response_id": getattr(response, "id", None),
                        "message_dump": message_dump,
                    },
                )
        result_json = _strip_deepseek_reasoning(raw_content)
    elif client_choice == "groq":
        groq_model = _groq_model()
        response = _groq_chat_completion(
            model=groq_model,
            messages=messages,
            use_json_mode=True,
        )
        result_json = _message_content_to_text(response.choices[0].message)
    else:
        raise ValueError("Invalid client selection")

    cleaned_json = _extract_json_payload(result_json)
    if not cleaned_json:
        raise ValueError(f"Received empty completion content from provider '{client_choice}'")
    try:
        parsed_payload = json.loads(cleaned_json)
    except json.JSONDecodeError as exc:
        logger.error(
            "Failed to decode JSON completion",
            extra={
                "client": client_choice,
                "raw_result": result_json,
                "cleaned_result": cleaned_json,
            },
        )
        raise

    # Normalize common LLM deviations before validation
    normalized_payload = _normalize_comparison_payload(parsed_payload)

    try:
        parsed_item = ComparisonItem.model_validate(normalized_payload)
    except ValidationError as ve:
        logger.warning(
            "Validation failed for ComparisonItem; attempting salvage",
            extra={"errors": ve.errors(), "payload_keys": list(normalized_payload.keys())},
        )
        # As a last resort, coerce everything to string
        fallback = {k: ("\n\n".join(map(str, v)) if isinstance(v, list) else (json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else ("" if v is None else str(v)))) for k, v in normalized_payload.items()}
        parsed_item = ComparisonItem.model_validate(fallback)
    # Override quote fields with deterministic top retrievals (highest similarity chunks, with IDs)
    parsed_item.paper_content_quotes = "\n\n".join(paper_top)
    parsed_item.registration_content_quotes = "\n\n".join(prereg_top)

    return ComparisonResult(items=[parsed_item])


def _load_pct_registration_text(pct_id: str, csv_path: str) -> str:
    """Load a preclinical trials registration row by pct_id and stringify its fields."""
    normalized_id = (pct_id or "").strip().lower()
    if not normalized_id:
        raise ValueError("A PCT identifier is required.")

    path = Path(csv_path)
    if not path.exists():
        raise ValueError(f"Registration CSV not found: {csv_path}")

    def _decode_csv_text(bytes_data: bytes) -> str:
        attempts = [
            ("utf-8", "strict"),
            ("utf-8-sig", "strict"),
            ("latin-1", "replace"),
        ]
        last_error: Exception | None = None
        for encoding, errors in attempts:
            try:
                return bytes_data.decode(encoding, errors=errors)
            except UnicodeDecodeError as exc:  # pragma: no cover - defensive decode
                last_error = exc
                continue
        raise ValueError(f"Failed to decode registration CSV '{csv_path}': {last_error}")  # pragma: no cover

    csv_text = _decode_csv_text(path.read_bytes())
    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames:
        raise ValueError("Registration CSV is missing headers.")
    pct_column = next(
        (field for field in reader.fieldnames if (field or "").strip().lower() == "pct_id"),
        None,
    )
    if pct_column is None:
        raise ValueError("Registration CSV must include a 'pct_id' column.")

    for row in reader:
        raw_id = (row.get(pct_column) or "").strip().lower()
        if raw_id != normalized_id:
            continue
        normalized_row: dict[str, str] = {}
        for key, value in row.items():
            if key is None:
                continue
            cleaned_key = str(key).strip()
            if not cleaned_key:
                continue
            cleaned_value = "" if value is None else str(value).strip()
            normalized_row[cleaned_key] = cleaned_value
        if not normalized_row:
            raise ValueError(f"No registration data found for PCT ID '{pct_id}'.")
        lines = [f"{k}: {v}" if v else f"{k}:" for k, v in normalized_row.items()]
        return "\n".join(lines)

    raise ValueError(f"PCT ID '{pct_id}' not found in registration CSV '{csv_path}'.")
