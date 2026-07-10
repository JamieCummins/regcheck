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


def _block_similarity(a: str, b: str) -> float:
    return SequenceMatcher(a=_norm(a), b=_norm(b), autojunk=False).ratio()


def align_stages(stage1_text: str, stage2_text: str, *, fuzzy_threshold: float = FUZZY_PAIR_THRESHOLD) -> dict[str, Any]:
    """Align Stage 1 blocks to Stage 2 blocks and enumerate every change.

    Returns {"stats": {...}, "changes": [RRChange dicts in document order]}.
    Verbatim moves (identical block at a different position) are counted in the
    stats but are NOT changes — fixity is about content, not pagination.
    """
    b1, b2 = _blocks(stage1_text), _blocks(stage2_text)
    n1, n2 = len(b1), len(b2)
    norms1, norms2 = [_norm(b) for b in b1], [_norm(b) for b in b2]

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
            # Pair edited blocks inside the replace run by best lexical similarity.
            candidates = sorted(
                ((_block_similarity(b1[i], b2[j]), i, j) for i in run1 for j in run2),
                key=lambda t: -t[0],
            )
            used1: set[int] = set()
            used2: set[int] = set()
            for sim, i, j in candidates:
                if sim < fuzzy_threshold:
                    break
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
# GATED: the pipeline only invokes this once the prompt wording below has been
# approved; until then every change ships with classification="pending".
RR_CLASSIFIER_ENABLED = False

RR_CLASSIFIER_PROMPT = """PENDING WORDING APPROVAL — see RR_CLASSIFIER_ENABLED."""


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
    raise NotImplementedError("Classifier prompt pending approval")
