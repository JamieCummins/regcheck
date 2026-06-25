"""Canonical default comparison dimensions + the single dimension-resolution
point. Dimensions and their definitions are resolved ONCE, at selection time,
via :func:`_resolve_dimensions`: user-specified values (UI wizard or API args)
are used verbatim; otherwise the explicit default set for the flow applies.
There is no other source — run_comparison never substitutes one by name.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path


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


# ---- discipline dimension presets (single source of truth) -------------------
# The named presets the web wizard offers ("Psychology", "Clinical / Medical",
# "Economics", "Preclinical / Animal") live in one data file so the web app, the
# API, and the CLI all resolve the same dimensions + definitions. The wizard loads
# them via the /compare template (injected JSON); the CLI via --dimension-set.
_DISCIPLINE_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "discipline_dimensions.json"


@lru_cache(maxsize=1)
def _discipline_data() -> dict:
    try:
        return json.loads(_DISCIPLINE_DATA_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # pragma: no cover - data file ships with the app
        return {"order": [], "sets": {}}


def discipline_keys() -> list[str]:
    """Preset keys in display order (e.g. ['psychology', 'clinical', ...])."""
    data = _discipline_data()
    return list(data.get("order") or data.get("sets", {}).keys())


def get_discipline_dimensions(key: str) -> list[dict[str, str]] | None:
    """Return the {dimension, definition} list for a preset key, or None if unknown."""
    entry = _discipline_data().get("sets", {}).get((key or "").strip().lower())
    if not entry:
        return None
    return [dict(d) for d in entry.get("dimensions", [])]


def _norm_dim_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").replace("–", "-").replace("—", "-").strip().lower())


@lru_cache(maxsize=1)
def _dimension_keyword_index() -> dict[str, tuple[str, ...]]:
    """Normalized dimension name -> retrieval keywords, unioned across all preset sets.
    Lets the hybrid retrieval guarantee surface high-value chunks (sample size,
    exclusions, …) for any dimension whose name matches a known preset dimension —
    including the *_DEFAULT_DIMENSIONS comparison defaults and user-supplied CSV/API
    dimensions, none of which carry keywords themselves."""
    index: dict[str, list[str]] = {}
    for entry in _discipline_data().get("sets", {}).values():
        for dim in entry.get("dimensions", []):
            kws = dim.get("keywords") or []
            if not kws:
                continue
            merged = index.setdefault(_norm_dim_name(dim.get("dimension", "")), [])
            for k in kws:
                if k not in merged:
                    merged.append(k)
    return {name: tuple(kws) for name, kws in index.items()}


def keywords_for_dimension(name: str, explicit: list[str] | None = None) -> list[str]:
    """Resolve retrieval keywords for a dimension: an explicit per-dimension list (from a
    preset) wins; otherwise fall back to the name-matched index."""
    if explicit:
        return list(explicit)
    return list(_dimension_keyword_index().get(_norm_dim_name(name), ()))


def discipline_sets_for_ui() -> list[dict]:
    """Presets shaped for the wizard: [{key, label, dims:[{name, definition}]}].
    Uses the ``name`` field the front-end expects; the glyph stays in the UI."""
    data = _discipline_data()
    sets = data.get("sets", {})
    out: list[dict] = []
    for key in discipline_keys():
        entry = sets.get(key) or {}
        out.append(
            {
                "key": key,
                "label": entry.get("label", key.title()),
                "dims": [
                    {"name": d.get("dimension", ""), "definition": d.get("definition", "")}
                    for d in entry.get("dimensions", [])
                ],
            }
        )
    return out


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
        normalized.append(
            {
                "dimension": name,
                "definition": (item.get("definition") or "").strip(),
                "keywords": keywords_for_dimension(name, item.get("keywords")),
            }
        )
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
    # Default sets (e.g. CLINICAL_/PRECLINICAL_DEFAULT_DIMENSIONS) carry no keywords;
    # enrich by name so the hybrid retrieval guarantee applies to them too.
    resolved: list[dict[str, str]] = []
    for item in defaults or []:
        enriched = dict(item)
        enriched["keywords"] = keywords_for_dimension(item.get("dimension", ""), item.get("keywords"))
        resolved.append(enriched)
    return resolved
