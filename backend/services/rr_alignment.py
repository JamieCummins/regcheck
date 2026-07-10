"""Registered Report Track A: carried-forward-text integrity (Stage 1 vs Stage 2).

Deterministic, lexical alignment of the two manuscripts' paragraph blocks —
exact matching first, fuzzy pairing for edited blocks, verbatim-move detection —
followed by word-level diffs inside modified pairs. The point of a diff-based
track (vs the per-dimension retrieval judge) is COVERAGE: every changed character
of the carried-forward text is found by construction, and the output is the
enumerated change list an RR editor wants. An LLM classifier then labels each
change (licensed / substantive / grey); until that pass runs, changes carry
classification="pending". No API calls happen in this module.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Any

from .text_normalize import normalize_text, reflow_text

# Below this block-level similarity, an edited paragraph no longer reads as "the
# same block, changed" and is reported as a delete + insert pair instead.
FUZZY_PAIR_THRESHOLD = 0.5


@dataclass
class RRChange:
    kind: str  # "modified" | "deleted" | "inserted"
    stage1_index: int | None
    stage2_index: int | None
    stage1_text: str = ""
    stage2_text: str = ""
    similarity: float = 0.0
    # Word-level opcodes for modified pairs: [{"op", "stage1", "stage2"}] where op
    # is "equal" | "delete" | "insert" | "replace". The report renders del/ins from
    # these; removed_text/added_text concatenate them for the classifier.
    diff: list[dict[str, str]] = field(default_factory=list)
    removed_text: str = ""
    added_text: str = ""
    classification: str = "pending"
    category: str = ""
    note: str = ""


def _blocks(text: str) -> list[str]:
    """Paragraph blocks of the canonical text. Reflow first so PDF-style hard
    line wraps don't masquerade as block boundaries or textual changes."""
    cleaned = reflow_text(normalize_text(text or ""))
    return [b.strip() for b in re.split(r"\n\s*\n", cleaned) if b.strip()]


def _norm(block: str) -> str:
    return " ".join(block.lower().split())


def _tokens(block: str) -> list[str]:
    return block.split()


def _word_diff(a: str, b: str) -> tuple[list[dict[str, str]], str, str]:
    """Word-level opcodes between two blocks + concatenated removed/added text."""
    ta, tb = _tokens(a), _tokens(b)
    ops: list[dict[str, str]] = []
    removed: list[str] = []
    added: list[str] = []
    for op, i1, i2, j1, j2 in SequenceMatcher(a=ta, b=tb, autojunk=False).get_opcodes():
        seg_a = " ".join(ta[i1:i2])
        seg_b = " ".join(tb[j1:j2])
        ops.append({"op": op, "stage1": seg_a, "stage2": seg_b})
        if op in ("delete", "replace") and seg_a:
            removed.append(seg_a)
        if op in ("insert", "replace") and seg_b:
            added.append(seg_b)
    return ops, " … ".join(removed), " … ".join(added)


def _block_similarity_norm(a_norm: str, b_norm: str, a_tokens: set[str], b_tokens: set[str]) -> float:
    """Similarity with a cheap token-set gate first. difflib's quick_ratio is a
    character-multiset bound — nearly useless for prose, where a wholesale
    rewrite still shares most of the alphabet — so dissimilar pairs are rejected
    on word-set overlap in O(tokens) before paying for the quadratic ratio().
    A ratio() of 0.5 requires a long common subsequence, which requires shared
    words; 0.6x leaves generous headroom for word-order effects."""
    if not a_tokens or not b_tokens:
        return 0.0
    overlap = len(a_tokens & b_tokens) / max(len(a_tokens), len(b_tokens))
    if overlap < FUZZY_PAIR_THRESHOLD * 0.6:
        return 0.0
    return SequenceMatcher(a=a_norm, b=b_norm, autojunk=False).ratio()


# Inside a replace run, a Stage 1 block is only compared against Stage 2 blocks
# near its position-mapped counterpart. Real manuscripts are near-monotonic, so
# this bounds the pairing cost to O(run1 x window) instead of O(run1 x run2).
PAIRING_WINDOW = 15


def align_stages(stage1_text: str, stage2_text: str, *, fuzzy_threshold: float = FUZZY_PAIR_THRESHOLD) -> dict[str, Any]:
    """Align Stage 1 blocks to Stage 2 blocks and enumerate every change.

    Returns {"stats": {...}, "changes": [RRChange dicts in document order]}.
    Verbatim moves (identical block at a different position) are counted in the
    stats but are NOT changes — fixity is about content, not pagination.
    """
    b1, b2 = _blocks(stage1_text), _blocks(stage2_text)
    n1, n2 = len(b1), len(b2)
    norms1, norms2 = [_norm(b) for b in b1], [_norm(b) for b in b2]
    toks1, toks2 = [set(n.split()) for n in norms1], [set(n.split()) for n in norms2]

    identical = 0
    moved = 0
    changes: list[RRChange] = []
    unmatched1: list[int] = []
    unmatched2: list[int] = []

    for op, i1, i2, j1, j2 in SequenceMatcher(a=norms1, b=norms2, autojunk=False).get_opcodes():
        if op == "equal":
            identical += i2 - i1
            continue
        run1 = list(range(i1, i2))
        run2 = list(range(j1, j2))
        if op in ("replace",):
            # Pair edited blocks inside the replace run by best lexical similarity,
            # windowed around each block's position-mapped counterpart.
            scale = len(run2) / len(run1) if run1 else 1.0
            pairs = []
            for offset1, i in enumerate(run1):
                expected = j1 + int(offset1 * scale)
                lo = max(j1, expected - PAIRING_WINDOW)
                hi = min(j2, expected + PAIRING_WINDOW + 1)
                for j in range(lo, hi):
                    sim = _block_similarity_norm(norms1[i], norms2[j], toks1[i], toks2[j])
                    if sim >= fuzzy_threshold:
                        pairs.append((sim, i, j))
            candidates = sorted(pairs, key=lambda t: -t[0])
            used1: set[int] = set()
            used2: set[int] = set()
            for sim, i, j in candidates:
                if i in used1 or j in used2:
                    continue
                used1.add(i)
                used2.add(j)
                diff, removed, added = _word_diff(b1[i], b2[j])
                changes.append(
                    RRChange(
                        kind="modified",
                        stage1_index=i,
                        stage2_index=j,
                        stage1_text=b1[i],
                        stage2_text=b2[j],
                        similarity=round(sim, 3),
                        diff=diff,
                        removed_text=removed,
                        added_text=added,
                    )
                )
            unmatched1.extend(i for i in run1 if i not in used1)
            unmatched2.extend(j for j in run2 if j not in used2)
        elif op == "delete":
            unmatched1.extend(run1)
        elif op == "insert":
            unmatched2.extend(run2)

    # Verbatim-move rescue: an unmatched Stage 1 block whose normalised text exists
    # among unmatched Stage 2 blocks moved position without changing content.
    remaining2 = {}
    for j in unmatched2:
        remaining2.setdefault(norms2[j], []).append(j)
    still1: list[int] = []
    for i in unmatched1:
        slot = remaining2.get(norms1[i])
        if slot:
            slot.pop(0)
            moved += 1
        else:
            still1.append(i)
    still2 = [j for j in unmatched2 if norms2[j] in remaining2 and j in _flatten(remaining2)]

    for i in still1:
        changes.append(RRChange(kind="deleted", stage1_index=i, stage2_index=None, stage1_text=b1[i], removed_text=b1[i]))
    for j in still2:
        changes.append(RRChange(kind="inserted", stage1_index=None, stage2_index=j, stage2_text=b2[j], added_text=b2[j]))

    changes.sort(key=lambda c: (c.stage2_index if c.stage2_index is not None else c.stage1_index or 0, c.kind))
    carried = identical + moved
    stats = {
        "stage1_blocks": n1,
        "stage2_blocks": n2,
        "identical": identical,
        "moved": moved,
        "modified": sum(1 for c in changes if c.kind == "modified"),
        "deleted": sum(1 for c in changes if c.kind == "deleted"),
        "inserted": sum(1 for c in changes if c.kind == "inserted"),
        "carried_forward_identical_pct": round(100.0 * carried / n1, 1) if n1 else 0.0,
    }
    return {"stats": stats, "changes": [asdict(c) for c in changes]}


def _flatten(remaining: dict[str, list[int]]) -> set[int]:
    out: set[int] = set()
    for slots in remaining.values():
        out.update(slots)
    return out


# ── LLM change classifier (Track A second stage) ────────────────────────────
RR_CLASSIFIER_ENABLED = True

RR_CLASSIFIER_PROMPT = (
    "You are reviewing a Registered Report. Stage 1 is a complete manuscript given "
    "in-principle acceptance before data collection; Stage 2 extends it after data "
    "collection. Carried-forward text is expected to match Stage 1 except for licensed "
    "changes. You will receive an enumerated list of detected textual changes between "
    "the two stages. Classify EACH change:\n"
    "- 'licensed': format-expected changes — appended results of registered analyses and "
    "their discussion; filled placeholders Stage 1 explicitly left open; future-to-past "
    "tense conversion; copyediting, reference, or formatting updates that leave identity, "
    "values, quantities, entities, and the strength and epistemic status of claims "
    "unchanged.\n"
    "- 'substantive': the change alters registered content or its epistemic status — "
    "hypothesis wording, theoretical rationale, emphasis, hedging or qualifier changes, "
    "altered values, procedures, or analyses, deleted commitments or caveats, or "
    "unregistered additions to carried-forward sections.\n"
    "- 'grey': genuinely ambiguous between the two; do not use it to avoid a judgement "
    "call.\n"
    "Also give a 'category' (tense | placeholder | copyedit | reference_format | "
    "results_discussion | hypothesis_wording | rationale | emphasis_hedging | "
    "value_or_procedure | deleted_commitment | unregistered_addition | other) and a "
    "one-sentence 'note' saying what changed. Direction matters: a prediction demoted to "
    "exploratory framing is substantive. Severity is not your call — classification "
    "describes the kind of change, and a human editor judges importance.\n"
    'Return ONLY a JSON array: [{"id": <number>, "classification": "...", '
    '"category": "...", "note": "..."}].'
)

_VALID_CLASSIFICATIONS = {"licensed", "substantive", "grey"}
_CHANGE_EXCERPT_CHARS = 600


def _change_line(index: int, change: dict[str, Any]) -> str:
    removed = (change.get("removed_text") or "")[:_CHANGE_EXCERPT_CHARS]
    added = (change.get("added_text") or "")[:_CHANGE_EXCERPT_CHARS]
    parts = [f"{index}. kind={change.get('kind')}"]
    if removed:
        parts.append(f'removed: "{removed}"')
    if added:
        parts.append(f'added: "{added}"')
    return " | ".join(parts)


def _classifier_completion(prompt: str, *, reasoning_effort: str | None = None) -> str:
    """One classifier call. Kept as a seam for tests; uses the OpenAI comparison
    model regardless of the run's judgement provider — classification is a small,
    cheap, structured task and one provider path keeps it dependable."""
    from .llm import _openai_family_model, get_openai_client

    client = get_openai_client()
    response = client.chat.completions.create(
        model=_openai_family_model("openai"),
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort=(reasoning_effort or "low"),
    )
    return response.choices[0].message.content or ""


def classify_changes(
    changes: list[dict[str, Any]],
    *,
    client_choice: str = "openai",
    reasoning_effort: str | None = None,
    batch_size: int = 25,
) -> list[dict[str, Any]]:
    """Label each detected change licensed/substantive/grey with a category, via
    batched LLM calls. Failures leave the affected changes at 'pending' — the
    deterministic change list is never lost to a classifier hiccup."""
    if not RR_CLASSIFIER_ENABLED or not changes:
        return changes
    import json as _json
    import logging

    logger = logging.getLogger(__name__)
    for start in range(0, len(changes), batch_size):
        batch = changes[start : start + batch_size]
        lines = "\n".join(_change_line(i + 1, c) for i, c in enumerate(batch))
        prompt = f"{RR_CLASSIFIER_PROMPT}\n\nChanges:\n{lines}"
        try:
            raw = _classifier_completion(prompt, reasoning_effort=reasoning_effort)
            match = re.search(r"\[.*\]", raw, re.S)
            labels = _json.loads(match.group(0)) if match else []
        except Exception as exc:  # pragma: no cover - network/parse degradation
            logger.warning("RR change classifier batch failed; leaving 'pending'", exc_info=exc)
            continue
        for entry in labels:
            try:
                idx = int(entry.get("id", 0)) - 1
            except (TypeError, ValueError):
                continue
            if not (0 <= idx < len(batch)):
                continue
            classification = str(entry.get("classification", "")).strip().lower()
            if classification not in _VALID_CLASSIFICATIONS:
                continue
            batch[idx]["classification"] = classification
            batch[idx]["category"] = str(entry.get("category", "")).strip()[:40]
            batch[idx]["note"] = str(entry.get("note", "")).strip()[:300]
    return changes
