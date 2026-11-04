"""Fetch an ISRCTN registration and recover the version in force at recruitment start.

ISRCTN's public API (``https://www.isrctn.com/api/trial/{id}/format/default``,
anonymous, XML) only serves the *current* record: there is no version-history
endpoint like ClinicalTrials.gov's. What it does keep is an editorial convention
inside the free-text fields. When a field changes, the editors relabel it,
newest first, e.g.::

    Current primary outcomes as of 03/11/2026:
    <text>

    Previous primary outcomes as of 13/03/2026:
    <text>

    Previous primary outcomes:
    <text>            <- undated: the text entered at registration

plus "Added DD/MM/YYYY:" for text appended later, and older variants ("...
(updated on DD/MM/YYYY):", "Original ...:", "Previous ... until DD/MM/YYYY:").
Newer records store outcomes in structured ``<outcomeMeasure>`` elements, in
which case the structured list is the current version and the legacy text field
holds only the earlier ones.

This module parses that convention into a version chain per field and picks the
version in force on the recruitment start date. It is best-effort, so each field
carries a recovery tier that says how far the record supports the choice:

* ``confirmed``     the chosen text was in force at the start, bounded by the
                    record's own dates (or the record was never edited after the
                    start);
* ``probable``      no change is recorded for this field, but the record was
                    edited after the start, so a silent edit can't be ruled out;
* ``unverified``    the record can't tell which version was in force (undated
                    or ambiguous change notes), or the field is a structured
                    value (sample size, dates) whose changes ISRCTN doesn't keep;
* ``unrecoverable`` the text in force at the start isn't in the record (the
                    earliest kept version post-dates the start) or the field's
                    layout couldn't be parsed;
* ``retrospective`` the trial was registered after recruitment started, so no
                    prospective version exists; the earliest kept text is used.

Two quirks of the convention, found in a sample of ~840 records, shape the
interval logic below: "Previous X as of D" normally means the version that
*became* current on D (a relabelled "Current X as of D"), but when it is the
first block of a field whose current version lives in the structured outcome
list, editors also use it to mean the version *replaced* on D. Both readings are
tried and a field is only ``confirmed`` when they agree.
"""

from __future__ import annotations

import itertools
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable

import requests

logger = logging.getLogger(__name__)

__all__ = [
    "extract_isrctn_id",
    "find_isrctn_id",
    "fetch_isrctn_xml",
    "split_versions",
    "recover_field",
    "extract_nested_trial_with_metadata",
    "TIER_LABELS",
]

NS = "{http://www.67bricks.com/isrctn}"
TRIAL_URL = "https://www.isrctn.com/api/trial/{isrctn_id}/format/default"

_ID_RE = re.compile(r"ISRCTN[\s-]*(\d{8})(?!\d)", re.IGNORECASE)

TIERS = ("confirmed", "probable", "unverified", "unrecoverable", "retrospective")
TIER_LABELS = {
    "confirmed": "Confirmed prospective",
    "probable": "Probably prospective",
    "unverified": "Unverified",
    "unrecoverable": "Unrecoverable",
    "retrospective": "Registered after start",
}


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int((os.environ.get(name) or "").strip()))
    except (TypeError, ValueError):
        return default


_CONNECT_TIMEOUT = 10
_READ_TIMEOUT = _int_env("ISRCTN_API_TIMEOUT", 30)
_MAX_ATTEMPTS = _int_env("ISRCTN_MAX_ATTEMPTS", 3)
_RETRY_STATUS = {502, 503, 504}


# --------------------------------------------------------------------------- ids


def find_isrctn_id(text: str | None) -> str | None:
    """Return the canonical ``ISRCTNnnnnnnnn`` id in ``text`` (URL, DOI or bare
    id), or None. Accepts ``ISRCTN 12345678`` and ``10.1186/ISRCTN12345678``."""
    match = _ID_RE.search(text or "")
    return f"ISRCTN{match.group(1)}" if match else None


def extract_isrctn_id(text: str) -> str:
    isrctn_id = find_isrctn_id(text)
    if not isrctn_id:
        raise ValueError("Unable to parse an ISRCTN identifier from input")
    return isrctn_id


# ------------------------------------------------------------------------- fetch


def fetch_isrctn_xml(isrctn_id: str) -> bytes:
    """GET the full record XML, retrying transient failures. ISRCTN answers an
    unknown id with an HTTP 500 HTML page, so a non-XML reply means "not found"."""
    url = TRIAL_URL.format(isrctn_id=isrctn_id)
    headers = {"Accept": "application/xml", "User-Agent": "RegCheck"}
    last_exc: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            resp = requests.get(url, headers=headers, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT))
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            logger.warning("ISRCTN request failed (%d/%d): %s", attempt + 1, _MAX_ATTEMPTS, exc)
            if attempt < _MAX_ATTEMPTS - 1:
                time.sleep(1.5 * (attempt + 1))
            continue
        if resp.status_code in _RETRY_STATUS and attempt < _MAX_ATTEMPTS - 1:
            last_exc = requests.HTTPError(f"ISRCTN returned {resp.status_code}")
            time.sleep(1.5 * (attempt + 1))
            continue
        content_type = (resp.headers.get("content-type") or "").lower()
        if resp.status_code == 200 and "xml" in content_type:
            return resp.content
        raise ValueError(
            f"ISRCTN has no public record for {isrctn_id} (HTTP {resp.status_code}). "
            "Check the identifier."
        )
    raise ValueError(
        "Couldn't reach the ISRCTN registry after several attempts. Try again in a moment."
    ) from last_exc


# ------------------------------------------------------------------------ dates

_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    )
}
_MONTH_WORD = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_DATE_PATTERNS = [
    # 30/04/2024, 30.04.2024, 30-04-24 (ISRCTN is UK-edited: day first)
    (re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4}|\d{2})\b"), "dmy"),
    # 12th February 2013
    (re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+" + _MONTH_WORD + r"\s+(\d{4})\b", re.I), "d_mon_y"),
    # February 12, 2013
    (re.compile(r"\b" + _MONTH_WORD + r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", re.I), "mon_d_y"),
]


def _find_dates(text: str) -> list[tuple[date, int, int]]:
    """Every date in ``text`` as ``(date, start, end)``, in reading order."""
    found: list[tuple[date, int, int]] = []
    taken: list[tuple[int, int]] = []
    for pattern, kind in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            if any(a < match.end() and match.start() < b for a, b in taken):
                continue
            parsed = _date_from_match(match, kind)
            if parsed:
                found.append((parsed, match.start(), match.end()))
                taken.append((match.start(), match.end()))
    return sorted(found, key=lambda item: item[1])


def _find_date(text: str) -> date | None:
    dates = _find_dates(text)
    return dates[0][0] if dates else None


def _date_from_match(match: re.Match, kind: str) -> date | None:
    try:
        if kind == "dmy":
            day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
            if year < 100:
                year += 2000
        elif kind == "d_mon_y":
            day, month, year = int(match.group(1)), _MONTHS[match.group(2).lower()[:3]], int(match.group(3))
        else:
            month, day, year = _MONTHS[match.group(1).lower()[:3]], int(match.group(2)), int(match.group(3))
        return date(year, month, day)
    except (ValueError, KeyError):
        return None


def _iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value.strip()[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


# ------------------------------------------------------------ version splitting

_KEYWORD_RE = re.compile(r"\b(current|previous|original|updated|amended|revised|as of)\b", re.I)
_OLDER_RE = re.compile(r"\b(previous|original)\b", re.I)
_UNTIL_RE = re.compile(r"\b(until|before|prior to|up to)\b", re.I)
# A candidate heading: a line whose text up to the first colon names a version.
_HEAD_RE = re.compile(r"(?m)^[ \t]*(?P<head>[^\n:]{3,200}?):[ \t]*(?P<rest>[^\n]*)$")
_ADDED_RE = re.compile(r"(?im)^[ \t]*(?P<verb>added|updated|amended|revised)\b(?P<head>[^\n:]{0,60}):?[ \t]*")
_RULE_RE = re.compile(r"(?m)^[ \t]*[_\-=*]{3,}[ \t]*$")


@dataclass
class Block:
    """One version of a field's text as laid out in the record."""

    text: str
    kind: str  # "current" | "older" | "unlabeled" | "structured"
    heading: str | None = None
    date: date | None = None  # "as of" date: when this version became current
    until: date | None = None  # "until" date: when this version was replaced
    speculative: bool = False  # unlabelled text guessed to be an earlier version


def _is_heading(head: str, rest: str) -> bool:
    """Tell a version heading ("Previous primary outcome measures as of 30/04/2024")
    from content that happens to contain a keyword and a colon ("1. Previous
    abdominal surgery: excluded")."""
    head = head.strip()
    if not _KEYWORD_RE.search(head) or re.match(r"^\(?\d+[.)]", head):
        return False
    if len(head.split()) > 16:
        return False
    if re.match(r"(?i)^(added|updated|amended|revised)\b", head):
        return False  # "Updated 13/03/2026: ..." is an in-text addition, not a version
    if re.search(r"(?i)\bpreviously\b", head):
        return False  # "... (updated 08/12/2020, previously: ...)" is an inline correction
    starts_with_keyword = bool(re.match(r"(?i)^(current|previous|original)\b", head))
    has_date = _find_date(head) is not None
    if starts_with_keyword:
        # Dated headings may carry text after the colon; undated ones stand alone.
        return has_date or not rest.strip()
    # "Inclusion criteria (updated on 19/09/2016):" or
    # "Participant inclusion criteria as of 24/09/2018:" on a line of its own.
    if has_date and not rest.strip():
        return bool(re.search(r"(?i)\((?:updated|amended|revised)\b|\bas (?:of|from)\b", head))
    return False


def split_versions(text: str | None) -> tuple[list[Block], bool]:
    """Split a field's text into version blocks, newest first.

    Returns ``(blocks, regular)``. ``regular`` is False when the layout doesn't
    follow the convention (several "current" headings, a "current" heading after
    an older one, ...); callers should not trust the chain then."""
    text = (text or "").replace("\r\n", "\n")
    text = _RULE_RE.sub("", text)
    heads = [m for m in _HEAD_RE.finditer(text) if _is_heading(m.group("head"), m.group("rest"))]
    if not heads:
        body = _tidy(text)
        return ([Block(body, "unlabeled")] if body else []), True

    blocks: list[Block] = []
    lead = _tidy(text[: heads[0].start()])
    if lead:
        blocks.append(Block(lead, "unlabeled"))
    for i, match in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        head = match.group("head").strip()
        body = _tidy((match.group("rest") or "") + text[match.end() : end])
        kind = "older" if _OLDER_RE.search(head) else "current"
        began, until = _heading_dates(head)
        blocks.append(Block(body, kind, heading=head, date=began, until=until))

    regular = True
    kinds = [b.kind for b in blocks]
    if kinds.count("current") > 1:
        regular = False
    if "current" in kinds and kinds.index("current") != 0:
        # e.g. lead text then "Current ... as of": the lead may be a preamble.
        regular = False
    if blocks[0].kind == "unlabeled" and len(blocks) > 1 and blocks[1].kind == "current":
        regular = False
    return blocks, regular


def _heading_dates(head: str) -> tuple[date | None, date | None]:
    """(became-current, replaced) dates of a heading such as "Previous X as of
    10/03/2010 and until 12/02/2013". A date preceded by "until"/"before" is the
    replacement date; any other is the "as of" date."""
    began = until = None
    cursor = 0
    for when, start, end in _find_dates(head):
        between = head[cursor:start]
        if _UNTIL_RE.search(between) or (began and re.fullmatch(r"\s*(?:to|-|–)\s*", between)):
            until = until or when
        else:
            began = began or when
        cursor = end
    return began, until


def _tidy(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text or "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


_INLINE_UPDATE_RE = re.compile(
    r"(?P<new>[^\n()]*?)[ \t]*\((?:updated|amended|changed)(?:\s+on)?\s+(?P<when>[^,;()\n]{6,25}?)[,;]?\s*"
    r"previously:?\s*(?P<old>[^()\n]*)\)",
    re.I,
)


def _resolve_inline_updates(text: str, ref: date) -> tuple[str, list[date]]:
    """Resolve "NEW (updated DD/MM/YYYY, previously: OLD)" notes, as ISRCTN puts
    on dates and contact details in plain English summaries: OLD if the update
    post-dates ``ref``, else NEW. Notes that don't parse are left as they are."""
    reverted: list[date] = []

    def swap(match: re.Match) -> str:
        when = _find_date(match.group("when"))
        if when is None:
            return match.group(0)
        if when > ref:
            reverted.append(when)
            prefix = re.match(r"\s*", match.group("new")).group(0)
            return prefix + match.group("old").strip()
        return match.group("new")

    return _INLINE_UPDATE_RE.sub(swap, text), reverted


def _strip_late_additions(text: str, start: date) -> tuple[str, list[date], bool]:
    """Drop "Added DD/MM/YYYY: ..." (or "Updated ...", "Amended ...") segments
    dated after ``start``. A segment runs to the next such line (version headings
    were already split off). Returns ``(text, removed dates, uncertain)``;
    ``uncertain`` is set when a removed segment was an "Updated"/"Amended" note,
    which may have replaced earlier text rather than added to it."""
    matches = [m for m in _ADDED_RE.finditer(text) if _find_date(m.group("head") or "")]
    if not matches:
        return text, [], False
    removed: list[date] = []
    uncertain = False
    pieces = [text[: matches[0].start()]]
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        added_on = _find_date(match.group("head"))
        if added_on and added_on > start:
            removed.append(added_on)
            uncertain = uncertain or match.group("verb").lower() != "added"
        else:
            pieces.append(text[match.start() : end])
    return _tidy("".join(pieces)), removed, uncertain


# ------------------------------------------------------------ version selection


@dataclass
class FieldRecovery:
    text: str
    tier: str
    reason: str
    versions: int = 1
    removed_additions: list[str] = field(default_factory=list)


def _possible_versions(
    blocks: list[Block], start: date, registered: date, last_updated: date
) -> tuple[set[int], dict[int, date]] | None:
    """Indices of the versions that could have been in force on ``start`` under
    some consistent reading of the dated headings. Also returns each version's
    latest-possible start (to prefer versions certainly begun by ``start``).
    None means no reading is consistent.

    A reading under which no kept version covers ``start`` adds nothing to the
    union: ISRCTN keeps a field's registration-time text, so a reading that
    implies it was dropped is less plausible than one that doesn't. Only when
    every reading leaves ``start`` uncovered is the result empty."""
    n = len(blocks)
    has_current_heading = any(b.kind == "current" and b.heading for b in blocks)

    # Each dated heading yields one or two candidate constraints:
    #   ("b", k, d): boundary k (version k replaced version k+1) happened on d
    #   ("s", d):    the oldest kept version itself began on d
    options: list[list[tuple]] = []
    for k, block in enumerate(blocks):

        def began(k=k, d=block.date):
            return ("b", k, d) if k < n - 1 else ("s", d)

        def replaced(k=k, d=block.date):
            return ("b", k - 1, d) if k >= 1 else None

        if block.until is not None and k >= 1:
            options.append([("b", k - 1, block.until)])
        if block.date is None:
            continue
        if block.kind == "current" or has_current_heading or block.until is not None:
            opt = [began()]
        else:
            # "Previous X as of D" with the current version outside the text
            # (structured outcomes / unlabeled lead): D is either when it began
            # or when it was replaced. Try both.
            opt = [began(), replaced()]
        opt = [o for o in opt if o is not None]
        if opt:
            options.append(opt)

    exact_dates = [o[-1] for opts in options for o in opts]
    floor = min([registered, *exact_dates])
    ceiling = max([last_updated, *exact_dates])

    union: set[int] = set()
    latest_start: dict[int, date] = {}
    any_consistent = False
    for combo in itertools.product(*options) if options else [()]:
        exact: dict[int, date] = {}
        oldest_start = registered
        ok = True
        for c in combo:
            if c[0] == "s":
                oldest_start = c[1]
                continue
            _, k, d = c
            if exact.get(k, d) != d:
                ok = False
            exact[k] = d
        lo = [exact.get(k, floor) for k in range(n - 1)]
        hi = [exact.get(k, ceiling) for k in range(n - 1)]
        # Boundaries run newest (k=0) to oldest: b0 >= b1 >= ... >= oldest start.
        if n >= 2:
            lo[n - 2] = max(lo[n - 2], oldest_start)
        for k in range(n - 3, -1, -1):
            lo[k] = max(lo[k], lo[k + 1])
        for k in range(1, n - 1):
            hi[k] = min(hi[k], hi[k - 1])
        if any(lo[k] > hi[k] for k in range(n - 1)):
            ok = False
        if n >= 2 and oldest_start > hi[n - 2]:
            ok = False
        if not ok:
            continue
        any_consistent = True
        for v in range(n):
            start_lo = lo[v] if v < n - 1 else oldest_start
            start_hi = hi[v] if v < n - 1 else oldest_start
            end_hi = hi[v - 1] if v >= 1 else date.max
            if start_lo <= start and end_hi >= start:
                union.add(v)
                prev = latest_start.get(v)
                latest_start[v] = start_hi if prev is None else max(prev, start_hi)
    if not any_consistent:
        return None
    return union, latest_start


def recover_field(
    blocks: list[Block],
    regular: bool,
    *,
    start: date | None,
    registered: date | None,
    last_updated: date | None,
    scalar: bool = False,
) -> FieldRecovery | None:
    """Choose the version of one field in force on ``start`` and grade it.

    A trial registered after recruitment started has no prospective version;
    its text is reconstructed as first registered instead, tiered
    ``retrospective``."""
    if not blocks:
        return None
    if start is None or registered is None or last_updated is None:
        return FieldRecovery(
            _joined(blocks) if not regular else blocks[0].text, "unverified",
            "The record has no recruitment start date; the current text is used.", len(blocks),
        )
    if start < registered:
        rec = _recover_at(blocks, regular, registered, registered, last_updated, scalar, "registration")
        rec.tier = "retrospective"
        rec.reason = "Registered after recruitment started, so the text as first registered is used. " + rec.reason
        return rec
    return _recover_at(blocks, regular, start, registered, last_updated, scalar, "recruitment start")


def _joined(blocks: list[Block]) -> str:
    return "\n\n".join((f"{b.heading}:\n{b.text}" if b.heading else b.text) for b in blocks)


def _recover_at(
    blocks: list[Block],
    regular: bool,
    ref: date,
    registered: date,
    last_updated: date,
    scalar: bool,
    ref_name: str,
) -> FieldRecovery:
    n = len(blocks)
    if not regular:
        return FieldRecovery(
            _joined(blocks), "unrecoverable",
            "The field's change notes don't follow the usual layout; the full text is shown.", n,
        )
    if last_updated <= ref:
        return FieldRecovery(blocks[0].text, "confirmed", f"The record was not edited after {ref_name}.", n)

    if n == 1 and blocks[0].date is None and blocks[0].until is None:
        chosen, tier = 0, ("unverified" if scalar else "probable")
        reason = "ISRCTN doesn't keep earlier values of this field." if scalar else "No change is recorded for this field."
    elif n == 2 and blocks[1].speculative:
        chosen, tier = None, "unverified"
        reason = "The record has a structured outcome list and separate, undated outcome text; both are shown."
    else:
        found = _possible_versions(blocks, ref, registered, last_updated)
        if found is None:
            return FieldRecovery(
                _joined(blocks), "unrecoverable",
                "The field's change dates contradict each other; the full text is shown.", n,
            )
        possible, latest_start = found
        if not possible:
            return FieldRecovery(
                blocks[-1].text, "unrecoverable",
                f"The earliest kept version was entered after {ref_name}; the text in force then isn't in the record.",
                n,
            )
        if len(possible) == 1:
            chosen = next(iter(possible))
            tier = "confirmed"
            reason = (
                f"Current text, unchanged since {ref_name}."
                if chosen == 0
                else f"Earlier version ({_ordinal_from_newest(chosen)}) in force at {ref_name}."
            )
        else:
            # Prefer the newest version that had certainly begun by the reference date.
            begun = [v for v in sorted(possible) if latest_start.get(v, date.max) <= ref]
            chosen = begun[0] if begun else max(possible)
            tier = "unverified"
            reason = (
                f"The change notes can't settle which of {len(possible)} versions was in force at "
                f"{ref_name}; the {_ordinal_from_newest(chosen)} is used."
            )

    text = _joined(blocks) if chosen is None else blocks[chosen].text
    text, removed, uncertain = _strip_late_additions(text, ref)
    text, reverted = _resolve_inline_updates(text, ref)
    if removed:
        reason += f" Text added after {ref_name} was removed."
        if uncertain:
            reason += " One removed note was an update, which may have replaced text that isn't kept."
            if tier in ("confirmed", "probable"):
                tier = "unverified"
    if reverted:
        reason += f" Inline values updated after {ref_name} were reverted."
    changed = sorted({*removed, *reverted})
    return FieldRecovery(text, tier, reason, n, [d.isoformat() for d in changed])


def _ordinal_from_newest(index: int) -> str:
    if index == 0:
        return "current version"
    if index == 1:
        return "previous version"
    return f"version {index + 1} back from the current one"


# ----------------------------------------------------------------- record shape


def _text(el: ET.Element | None) -> str:
    return (el.text or "").strip() if el is not None else ""


def _find(parent: ET.Element | None, path: str) -> ET.Element | None:
    if parent is None:
        return None
    return parent.find("/".join(NS + p for p in path.split("/")))


def _findall(parent: ET.Element | None, path: str) -> list[ET.Element]:
    if parent is None:
        return []
    return parent.findall("/".join(NS + p for p in path.split("/")))


def _structured_outcomes(container: ET.Element | None) -> str:
    lines = []
    for i, measure in enumerate(_findall(container, "outcomeMeasure"), start=1):
        variable = _text(_find(measure, "variable"))
        method = _text(_find(measure, "method"))
        timepoints = _text(_find(measure, "timepoints"))
        line = variable
        if method:
            line += f" measured using {method}"
        if timepoints:
            line += f" at {timepoints}"
        if line:
            lines.append(f"{i}. {line}")
    return "\n".join(lines)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _outcome_blocks(structured: str, legacy: str) -> tuple[list[Block], bool]:
    """Blocks for an outcome field that may have a structured current list."""
    blocks, regular = split_versions(legacy)
    if not structured:
        return blocks, regular
    current = Block(structured, "structured")
    if not blocks:
        return [current], True
    if blocks[0].kind == "current":
        # Legacy text still carries its own current block: trust the text chain.
        return blocks, regular
    if len(blocks) == 1 and blocks[0].kind == "unlabeled":
        if _norm(blocks[0].text) == _norm(structured):
            return [current], True
        # Unlabelled legacy text alongside a structured list: an earlier version
        # whose replacement date isn't recorded.
        return [current, Block(blocks[0].text, "older", speculative=True)], regular
    if blocks[0].kind == "unlabeled":
        return blocks, False
    return [current, *blocks], regular


@dataclass
class _FieldSpec:
    section: str
    label: str
    blocks: list[Block]
    regular: bool
    scalar: bool = False
    tracked: bool = True  # report a recovery tier for this field


def _field_specs(trial: ET.Element) -> list[_FieldSpec]:
    desc = _find(trial, "trialDescription")
    design = _find(trial, "trialDesign")
    parts = _find(trial, "participants")
    specs: list[_FieldSpec] = []

    def add(section: str, label: str, value: str, *, scalar: bool = False, tracked: bool = True):
        blocks, regular = split_versions(value)
        if blocks:
            specs.append(_FieldSpec(section, label, blocks, regular, scalar, tracked))

    add("Identification", "Title", _text(_find(desc, "title")), tracked=False)
    add("Identification", "Scientific title", _text(_find(desc, "scientificTitle")))
    add("Identification", "Acronym", _text(_find(desc, "acronym")), scalar=True, tracked=False)

    add("Description", "Study hypothesis", _text(_find(desc, "studyHypothesis")))
    add("Description", "Plain English summary", _text(_find(desc, "plainEnglishSummary")))

    add("Design", "Study design", _text(_find(design, "studyDesign")))
    add("Design", "Primary study design", _text(_find(design, "primaryStudyDesign")), scalar=True, tracked=False)
    add("Design", "Secondary study design", _text(_find(design, "secondaryStudyDesign")), scalar=True, tracked=False)
    trial_types = ", ".join(t for t in (_text(e) for e in _findall(design, "trialTypes/trialType")) if t)
    add("Design", "Trial type", trial_types, scalar=True, tracked=False)

    for i, cond in enumerate(_findall(trial, "conditions/condition"), start=1):
        suffix = f" {i}" if len(_findall(trial, "conditions/condition")) > 1 else ""
        add("Conditions", f"Condition{suffix}", _text(_find(cond, "description")))
        classes = ", ".join(t for t in (_text(_find(cond, "diseaseClass1")), _text(_find(cond, "diseaseClass2"))) if t)
        add("Conditions", f"Disease class{suffix}", classes, scalar=True, tracked=False)

    interventions = _findall(trial, "interventions/intervention")
    for i, iv in enumerate(interventions, start=1):
        suffix = f" {i}" if len(interventions) > 1 else ""
        add("Interventions", f"Intervention{suffix}", _text(_find(iv, "description")))
        add("Interventions", f"Intervention type{suffix}", _text(_find(iv, "interventionType")), scalar=True, tracked=False)
        add("Interventions", f"Phase{suffix}", _text(_find(iv, "phase")), scalar=True, tracked=False)
        add("Interventions", f"Drug names{suffix}", _text(_find(iv, "drugNames")), scalar=True)

    p_types = ", ".join(t for t in (_text(e) for e in _findall(parts, "participantTypes/participantType")) if t)
    add("Participants", "Participant type", p_types, scalar=True, tracked=False)
    add("Participants", "Inclusion criteria", _text(_find(parts, "inclusion")))
    add("Participants", "Exclusion criteria", _text(_find(parts, "exclusion")))
    add("Participants", "Age range", _text(_find(parts, "ageRange")), scalar=True, tracked=False)
    add("Participants", "Lower age limit", _text(_find(parts, "lowerAgeLimit")), scalar=True)
    add("Participants", "Upper age limit", _text(_find(parts, "upperAgeLimit")), scalar=True)
    add("Participants", "Sex", _text(_find(parts, "gender")), scalar=True, tracked=False)
    add("Participants", "Target sample size", _text(_find(parts, "targetEnrolment")), scalar=True)
    countries = ", ".join(t for t in (_text(e) for e in _findall(parts, "recruitmentCountries/country")) if t)
    add("Participants", "Recruitment countries", countries, scalar=True, tracked=False)

    for kind, label in (("primary", "Primary outcome"), ("secondary", "Secondary outcomes")):
        structured = _structured_outcomes(_find(desc, f"{kind}Outcomes"))
        legacy = _text(_find(desc, f"{kind}Outcome"))
        blocks, regular = _outcome_blocks(structured, legacy)
        if blocks:
            specs.append(_FieldSpec("Outcomes", label, blocks, regular))
    return specs


def _parse_trial(xml_bytes: bytes) -> ET.Element:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise ValueError("ISRCTN returned a record that couldn't be read.") from exc
    trial = root.find(f"{NS}trial") if root.tag != f"{NS}trial" else root
    if trial is None:
        trial = next(root.iter(f"{NS}trial"), None)
    if trial is None:
        raise ValueError("ISRCTN returned a record without trial data.")
    return trial


def extract_nested_trial_with_metadata(
    isrctn_id: str,
    *,
    xml_fetcher: Callable[[str], bytes] | None = None,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    """Fetch ``isrctn_id`` and reconstruct the registration in force at
    recruitment start, as ``{section: {field: text}}`` (the same shape
    ``trials.extract_nested_trial_with_metadata`` returns for CT.gov) plus
    metadata describing how each field was recovered."""
    isrctn_id = extract_isrctn_id(isrctn_id)
    trial = _parse_trial((xml_fetcher or fetch_isrctn_xml)(isrctn_id))

    registered = _iso_date((_find(trial, "isrctn").attrib if _find(trial, "isrctn") is not None else {}).get("dateAssigned"))
    last_updated = _iso_date(trial.get("lastUpdated"))
    parts = _find(trial, "participants")
    start = _iso_date(_text(_find(parts, "recruitmentStart")))
    end = _iso_date(_text(_find(parts, "recruitmentEnd")))

    nested: dict[str, dict[str, str]] = {
        "Identification": {"ISRCTN": isrctn_id},
    }
    if registered:
        nested["Identification"]["Date registered"] = registered.isoformat()
    fields_meta: list[dict[str, Any]] = []
    for spec in _field_specs(trial):
        rec = recover_field(
            spec.blocks, spec.regular,
            start=start, registered=registered, last_updated=last_updated, scalar=spec.scalar,
        )
        if rec is None:
            continue
        if rec.text:
            nested.setdefault(spec.section, {})[spec.label] = rec.text
        if spec.tracked:
            fields_meta.append(
                {
                    "field": spec.label,
                    "tier": rec.tier,
                    "reason": rec.reason if rec.text else rec.reason + " The field was empty at recruitment start.",
                    "versions": rec.versions,
                    "removed_additions": rec.removed_additions,
                }
            )
    if start:
        nested.setdefault("Participants", {})["Recruitment start"] = start.isoformat()
    if end:
        nested.setdefault("Participants", {})["Recruitment end"] = end.isoformat()
    overall_end = _iso_date(_text(_find(_find(trial, "trialDesign"), "overallEndDate")))
    if overall_end:
        nested.setdefault("Design", {})["Overall end date"] = overall_end.isoformat()

    summary = {tier: 0 for tier in TIERS}
    for item in fields_meta:
        summary[item["tier"]] = summary.get(item["tier"], 0) + 1
    prospective = None if (start is None or registered is None) else registered <= start
    metadata: dict[str, Any] = {
        "registry": "isrctn",
        "isrctn_id": isrctn_id,
        "registration_date": registered.isoformat() if registered else None,
        "start_date": start.isoformat() if start else None,
        "last_updated": last_updated.isoformat() if last_updated else None,
        "record_version": trial.get("version"),
        "prospective": prospective,
        "source": "reconstructed" if start and last_updated and last_updated > start else "latest",
        "fields": fields_meta,
        "summary": summary,
    }
    return nested, metadata
