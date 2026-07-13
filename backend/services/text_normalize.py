"""Robust text decoding + normalisation for extracted document text.

Two boundary guards used across the parsers/readers so that mojibake never reaches
the comparison, the spans/manifest, or the displayed quotes:

* ``decode_bytes`` — bytes → str, robust to the common Western encodings.
* ``normalize_text`` — str → clean, well-formed Unicode (idempotent).

Applied at the *extraction boundary* (before spans/chunks/manifest are computed), so
length-changing maps (e.g. ellipsis → "...") keep the quote locator consistent.
"""

from __future__ import annotations

import re
import unicodedata

# Typographic punctuation + Latin ligatures → ASCII. Genuine letters (accents, Greek,
# currency) are deliberately NOT in this table — they survive as clean Unicode.
_REPLACEMENTS: dict[str, str] = {
    # single quotes / prime
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    # double quotes / double prime / guillemets
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
    "«": '"', "»": '"',
    # hyphens / dashes / minus
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "―": "-", "−": "-",
    # ellipsis
    "…": "...",
    # bullets / middle dots
    "•": "-", "‧": "-", "·": "-",
    # non-standard spaces → plain space
    "\u00A0": " ", "\u2007": " ", "\u2009": " ", "\u200A": " ", "\u202F": " ", "\u3000": " ",
    # zero-width characters / BOM → removed
    "\u200B": "", "\u200C": "", "\u200D": "", "\uFEFF": "",
    # common Latin ligatures (NFC leaves these alone; fold them for matching/readability)
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "ﬅ": "st", "ﬆ": "st",
}
_TRANSLATION = {ord(k): v for k, v in _REPLACEMENTS.items()}


def decode_bytes(data: bytes | str) -> str:
    """Decode file bytes to text, robust to the common Western encodings.

    Tries UTF-8 (BOM-aware, then plain) strictly, then falls back to cp1252
    (Windows-1252) — a superset of Latin-1 that maps 0x91-0x97 back to smart quotes
    and dashes, so a mis-saved Word/Windows export is *recovered* rather than dropped
    (which is what ``utf-8, errors='ignore'`` silently did). cp1252 with
    ``errors='replace'`` never raises.
    """
    if isinstance(data, str):
        return data
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("cp1252", errors="replace")


def normalize_text(text: str) -> str:
    """Normalise extracted text to clean, well-formed Unicode. Idempotent.

    Canonicalises (NFC), folds Latin ligatures, maps typographic punctuation to ASCII,
    normalises line endings, and strips control/format/replacement characters — while
    PRESERVING genuine letters (``é``, ``ö``, ``η``, ``µ``) and notation (``²``, ``½``).
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(_TRANSLATION)
    out: list[str] = []
    for ch in text:
        if ch in ("\n", "\t"):
            out.append(ch)
            continue
        if ch == "�":  # Unicode replacement char (decode garbage) — drop
            continue
        if unicodedata.category(ch).startswith("C"):  # control / format / surrogate / unassigned
            continue
        out.append(ch)
    result = "".join(out)
    # Tidy the "ladder of empty lines" PDF extraction leaves around page footers/headers:
    # drop trailing spaces (so whitespace-only lines are truly blank), then collapse any
    # run of 3+ newlines to a single paragraph break. Done here, at the extraction
    # boundary (before chunks/spans are computed), so corpus + display stay aligned.
    result = re.sub(r"[ \t]+\n", "\n", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result


_ISOLATED_NEWLINE = re.compile(r"(?<!\n)\n(?!\n)")


def reflow_text(text: str) -> str:
    """Collapse hard line-wraps from PDF extraction into spaces, keeping paragraphs.

    PDF text extraction inserts a newline at every *visual* line end, so a flowing
    paragraph arrives full of mid-sentence breaks. This turns each *isolated* newline
    (a line wrap) into a single space while leaving runs of two-or-more newlines
    (paragraph breaks) intact — so the displayed document reads as prose with its
    structure preserved, instead of a ladder of short lines.

    Length-preserving (one isolated ``\\n`` → one space), so it is safe to apply to
    DISPLAY text the quote locator matches against — provided it is applied
    *consistently* to both the rendered document text and the chunk/quote text.
    """
    if not text:
        return ""
    return _ISOLATED_NEWLINE.sub(" ", text)


def decode_and_normalize(data: bytes | str) -> str:
    """Convenience: ``normalize_text(decode_bytes(data))``."""
    return normalize_text(decode_bytes(data))
