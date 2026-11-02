"""Label-based study isolation: label the paper's units, then filter deterministically.

1. ``segment(text)``          -> numbered paragraphs / sentences (ids like "12" and "12.3")
2. ``LABEL_PROMPT``           -> one text-only LLM call returning LABELS ONLY (no text): the
                                 study map, paragraph labels, sentence overrides, cross-references
3. ``isolate(units, labels, target)`` -> the target study's text, built only from the paper's
                                 own sentences (shared + target units, in order)

Unlike generative isolation, nothing is rewritten, so nothing can be paraphrased or silently
dropped by regeneration; the only model-written text is the short scope prefix on sentences
imported through a cross-reference. One labelling call covers every study in the paper.

Cross-references: when the target study defers to another study for one aspect ("our sampling
approach was the same as in Study 2"), ONLY the sentences describing that aspect are imported,
each on its own line with an inline scope prefix (a single header could be separated from the
content it qualifies by chunking).

Pure functions only: the LLM call, caching and fallbacks live in ``comparisons``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .embeddings import _is_heading_line

# Bump whenever LABEL_PROMPT or segmentation changes (keys the CLI label cache).
LABEL_PROMPT_VERSION = "2026-11-02.1"

_SENT_SPLIT = re.compile(r"(?<=[.!?])[\"”’)\]]?\s+(?=[A-Z(\"“‘\[])")
# A paragraph longer than this many sentences is split into several numbered units, so
# text without blank-line paragraph breaks (common in PDF text layers) stays labellable.
_MAX_SENTENCES_PER_UNIT = 8


@dataclass
class Para:
    pid: int
    text: str
    sentences: list[str] = field(default_factory=list)
    block: bool = False  # headings and tables: one unit, never split into sentences


def _is_table_block(lines: list[str]) -> bool:
    return sum(1 for line in lines if "|" in line) >= 2


def segment(text: str) -> list[Para]:
    """Number the paper's paragraphs and sentences for the labeller.

    Paragraphs are blank-line separated; within one, heading lines become their own unit
    and hard line-wraps are joined, so PDF text layers (which wrap every visual line and
    often lack blank lines) segment like clean transcripts. Tables (" | " rows, as the
    XML extractor emits) stay whole."""
    paras: list[Para] = []

    def _emit(unit_text: str, *, block: bool) -> None:
        unit_text = unit_text.strip()
        if not unit_text:
            return
        if block:
            paras.append(Para(len(paras) + 1, unit_text, [unit_text], block=True))
            return
        sentences = [s.strip() for s in _SENT_SPLIT.split(unit_text) if s.strip()] or [unit_text]
        for start in range(0, len(sentences), _MAX_SENTENCES_PER_UNIT):
            group = sentences[start : start + _MAX_SENTENCES_PER_UNIT]
            paras.append(Para(len(paras) + 1, " ".join(group), group))

    for raw_block in re.split(r"\n\s*\n", text or ""):
        lines = [line for line in raw_block.splitlines() if line.strip()]
        if not lines:
            continue
        if _is_table_block(lines):
            _emit("\n".join(line.strip() for line in lines), block=True)
            continue
        current: list[str] = []
        for line in lines:
            if _is_heading_line(line):
                _emit(" ".join(current), block=False)
                current = []
                _emit(line, block=True)
            else:
                current.append(line.strip())
        _emit(" ".join(current), block=False)
    return paras


def render_for_labeller(paras: list[Para]) -> str:
    out = []
    for p in paras:
        if len(p.sentences) == 1:
            out.append(f"<P{p.pid}> {p.text}")
        else:
            out.append(
                f"<P{p.pid}> "
                + " ".join(f"⟨{p.pid}.{j}⟩ {s}" for j, s in enumerate(p.sentences, start=1))
            )
    return "\n\n".join(out)


LABEL_PROMPT = """You label the structure of a scientific paper that may report several studies (also called experiments). You do NOT rewrite text: you only return labels that refer to the numbered units.

The paper is given as numbered paragraphs <P12>; multi-sentence paragraphs also number their sentences ⟨12.1⟩, ⟨12.2⟩.

Return JSON with exactly these keys:

"studies": the empirical studies the paper reports, in order: [{"id": "S1", "label": "<the paper's own label, e.g. Study 1 / Experiment 1 / Study 3a>"}]. If the paper reports only one study, return one entry.

"paragraphs": labels for EVERY paragraph, as ranges: [{"from": 1, "to": 9, "label": "front"}, {"from": 10, "to": 40, "label": "shared"}, {"from": 41, "to": 80, "label": ["S1"]}, ...]. Labels:
  - "front": title, authors, affiliations, abstract, keywords, article metadata, and the reference list.
  - "shared": content that is not specific to one study: the general Introduction, general/overview method or materials that govern several studies, open-practice / data-availability / ethics approval / consent / funding / transparency statements and author notes (these are ALWAYS "shared" unless they explicitly name specific studies — wherever they are printed, including footnotes), the General Discussion, appendices that are not study-specific.
  - a list of study ids, e.g. ["S2"] or ["S3", "S4"]: content belonging to those studies — their section, method, participants, measures, procedure, exclusions, analyses, results, study-specific discussion, and their tables, figures, captions, table notes and footnotes. A method section written jointly for several studies gets all of them.

"sentences": overrides for single sentences whose label differs from their paragraph's: [{"id": "12.3", "label": ["S2"]}]. This is REQUIRED, sentence by sentence, inside every "shared" Introduction, overview ("the current research", "overview of studies") and General Discussion paragraph: any sentence that states the hypotheses, predictions, design, sample or results of particular studies must be labelled with exactly those studies (e.g. "In Study 2 we found ..." -> ["S2"]; "Studies 3 and 4 showed ..." -> ["S3", "S4"]). Sentences that are general (theory, background, implications, limitations of the whole research) stay "shared". Sentences about the properties or limitations of a MEASURE, material or procedure (e.g. its reliability, validity, a flaw in its design) are labelled with EVERY study that used that same measure/material/procedure — even if the evidence came from only one study. Only that study's own result statements (e.g. "In Experiment 1, the correlations were ...") stay with the study that produced them. Conversely, a sentence inside a study's section that explicitly applies to all or several studies ("for this and all subsequent experiments", "materials for all experiments are at ...") gets "shared" or the list of studies it covers.

"cross_refs": every place where a study's text explicitly defers to ANOTHER study instead of restating something (e.g. "the procedure was the same as Study 3, except that ...", "our sampling approach was the same as in Study 2"):
  [{"citing": "<sentence or paragraph id of the deferring sentence>", "study": "<id of the study doing the citing>", "referenced_study": "<id>", "aspect": "<exactly what is said to be the same, in a few words, including any stated exceptions>", "referenced": ["<ids of ONLY the units in the referenced study that describe that aspect>"]}]
  Be minimal and literal: "referenced" must cover only what the citing sentence says is the same. If it says the sampling approach was the same, list only the sampling sentences — not the sample size actually obtained, not the measures, not the results. Never list a whole section unless the sentence explicitly says the whole section/method was identical. References to other papers (e.g. "identical to Payne et al., 2013") are not cross_refs.

Every paragraph must be covered by exactly one range. Output only the JSON."""


# ── label validation ─────────────────────────────────────────────────────────


def _norm_label(value: Any) -> Any:
    """Coerce a unit label to "front" | "shared" | [study ids]."""
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"front", "shared"}:
            return low
        return [value.strip()] if value.strip() else "shared"
    if isinstance(value, (list, tuple)):
        ids = [str(v).strip() for v in value if str(v).strip()]
        return ids or "shared"
    return "shared"


def _labels_by_unit(paras: list[Para], labels: dict) -> tuple[dict[int, Any], dict[str, Any], list[str]]:
    para_label: dict[int, Any] = {}
    problems: list[str] = []
    for r in labels.get("paragraphs") or []:
        try:
            lo, hi = int(r["from"]), int(r["to"])
        except (KeyError, TypeError, ValueError):
            problems.append(f"bad paragraph range {r!r}")
            continue
        for pid in range(lo, hi + 1):
            if pid in para_label:
                problems.append(f"P{pid} labelled twice")
            para_label[pid] = _norm_label(r.get("label"))
    missing = [p.pid for p in paras if p.pid not in para_label]
    if missing:
        problems.append(f"{len(missing)} paragraphs unlabelled: {missing[:10]}")
    sent_label = {
        str(s.get("id")): _norm_label(s.get("label"))
        for s in labels.get("sentences") or []
        if isinstance(s, dict) and s.get("id") is not None
    }
    # Guard: a study's own section heading ("Experiment 7: ...") always belongs to that
    # study (the labeller occasionally puts a boundary heading in the preceding range).
    for st in labels.get("studies") or []:
        lab = re.escape((st.get("label") or "").strip())
        if not lab:
            continue
        for p in paras:
            if p.block and len(p.text.split()) <= 25 and re.match(rf"{lab}(?:\s*[:.—-]|\s*$)", p.text, re.IGNORECASE):
                if para_label.get(p.pid) != [st["id"]]:
                    para_label[p.pid] = [st["id"]]
    return para_label, sent_label, problems


def validate_labels(paras: list[Para], labels: Any) -> list[str]:
    """Fatal problems that make a label set unusable (empty = usable). Partial coverage
    is tolerated (unlabelled paragraphs are kept as shared), but most of the paper must
    be labelled and at least one study identified."""
    if not isinstance(labels, dict):
        return ["labels are not a JSON object"]
    studies = labels.get("studies")
    if not isinstance(studies, list) or not any(isinstance(s, dict) and s.get("id") for s in studies):
        return ["no studies identified"]
    para_label, _sent, _problems = _labels_by_unit(paras, labels)
    covered = sum(1 for p in paras if p.pid in para_label)
    if paras and covered < 0.8 * len(paras):
        return [f"only {covered}/{len(paras)} paragraphs labelled"]
    return []


# ── target resolution ────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"(\d+[a-z]?|\b[a-z])\s*$", re.IGNORECASE)


def _label_token(label: str) -> str | None:
    match = _TOKEN_RE.search(" ".join((label or "").split()))
    return match.group(1).lower() if match else None


def resolve_target(labels: dict, user_label: str) -> str | None:
    """Map the user's target label ("2", "Study 2", "Experiment 2", "3a", "S2") onto a
    labelled study id; a paper with a single study resolves to it. None if ambiguous
    or absent."""
    studies = [s for s in labels.get("studies") or [] if isinstance(s, dict) and s.get("id")]
    wanted = " ".join((user_label or "").split()).lower()
    if not studies or not wanted:
        return None
    for s in studies:  # exact id or exact label
        if wanted in {str(s["id"]).lower(), " ".join(str(s.get("label") or "").split()).lower()}:
            return str(s["id"])
    token = _label_token(wanted)
    if token:
        hits = [s for s in studies if _label_token(str(s.get("label") or "")) == token]
        if len(hits) == 1:
            return str(hits[0]["id"])
    if len(studies) == 1:
        return str(studies[0]["id"])
    return None


# ── deterministic isolation ──────────────────────────────────────────────────


def _applies(label: Any, target: str) -> bool:
    if label == "shared":
        return True
    return isinstance(label, list) and target in label


def _unit_text(paras_by_id: dict[int, Para], uid: str) -> str | None:
    pid, _, sid = str(uid).partition(".")
    p = paras_by_id.get(int(pid)) if pid.isdigit() else None
    if not p:
        return None
    if not sid:
        return p.text
    j = int(sid) - 1 if sid.isdigit() else -1
    return p.sentences[j] if 0 <= j < len(p.sentences) else None


def cross_ref_prefix(study_name: str, aspect: str) -> str:
    """Short scope prefix for an imported sentence (kept short: it is embedded with the
    sentence and dilutes its vector)."""
    aspect = " ".join((aspect or "").split())
    return f"[From {study_name}, cited for: {aspect}]" if aspect else f"[From {study_name}]"


def isolate(paras: list[Para], labels: dict, target: str) -> tuple[str, dict]:
    """Target study text: shared + target units in paper order (front matter and other
    studies dropped), with scoped cross-referenced sentences inserted after each citing
    paragraph."""
    para_label, sent_label, problems = _labels_by_unit(paras, labels)
    by_id = {p.pid: p for p in paras}
    names = {s["id"]: s.get("label") or s["id"] for s in labels.get("studies") or [] if isinstance(s, dict)}
    refs_after: dict[int, list[str]] = {}
    imported = 0
    for xr in labels.get("cross_refs") or []:
        if not isinstance(xr, dict) or xr.get("study") != target or xr.get("referenced_study") == target:
            continue
        try:
            cpid = int(str(xr.get("citing", "0")).split(".")[0] or 0)
        except ValueError:
            problems.append(f"bad cross_ref citing id {xr.get('citing')!r}")
            continue
        prefix = cross_ref_prefix(
            str(names.get(xr.get("referenced_study"), xr.get("referenced_study"))), str(xr.get("aspect") or "")
        )
        for uid in xr.get("referenced") or []:
            text = _unit_text(by_id, uid)
            if text is None:
                problems.append(f"cross_ref unit {uid} not found")
                continue
            # Every imported sentence carries its scope, so no chunk can hold it unqualified.
            units = [text] if "|" in text else [s for s in _SENT_SPLIT.split(text) if s.strip()]
            for sentence in units:
                refs_after.setdefault(cpid, []).append(f"{prefix} {sentence.strip()}")
                imported += 1

    blocks: list[str] = []
    kept_sents = total_sents = 0
    for p in paras:
        plabel = para_label.get(p.pid, "shared")  # unlabelled -> keep (recall over precision)
        if isinstance(plabel, list) and target in plabel:
            # The target's own text is never trimmed: sentence overrides only refine
            # SHARED sections (e.g. its rationale citing another study's result stays).
            kept = list(p.sentences)
        else:
            kept = [
                s
                for j, s in enumerate(p.sentences, start=1)
                if _applies(sent_label.get(f"{p.pid}.{j}", plabel), target)
            ]
        total_sents += len(p.sentences)
        kept_sents += len(kept)
        if kept:
            blocks.append(p.text if p.block else " ".join(kept))
        if p.pid in refs_after:
            blocks.extend(refs_after[p.pid])
    stats = {
        "kept_sentences": kept_sents,
        "total_sentences": total_sents,
        "imported_ref_sentences": imported,
        "problems": problems,
    }
    return "\n\n".join(blocks), stats
