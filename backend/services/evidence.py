from __future__ import annotations

import mimetypes
import string
import tempfile
from pathlib import Path
from typing import Any

from .embeddings import extract_chunks_tokens_with_spans
from .text_normalize import normalize_text, reflow_text

try:  # pragma: no cover - optional dependency
    import fitz
except ModuleNotFoundError:  # pragma: no cover
    fitz = None


def _display_text(text: str) -> str:
    """Text as SHOWN in the viewer: normalised, then reflowed so PDF hard line-wraps
    read as prose (paragraphs kept). Reflow is length-preserving, so applying it to
    BOTH the rendered document text and the chunk/quote text keeps the quote locator
    aligned. Never use for the raw cache-key text or the chunk spans."""
    return reflow_text(normalize_text(text))


def corpus_segments(
    text: str, *, max_chunk_tokens: int, embedding_model: str
) -> tuple[list[str], list[dict[str, Any]]]:
    """The segments (display-normalised chunks of ``text``) and per-segment metadata
    (``in_references`` flags) that EVERY retrieval corpus over a document embeds. The
    evidence builders produce exactly these, so runs with and without the evidence
    viewer retrieve over identical text."""
    chunks = extract_chunks_tokens_with_spans(
        text or "", max_chunk_tokens=max_chunk_tokens, encoding_name=embedding_model
    )
    refs_span = _references_span(text or "")
    segments = [_display_text(chunk.text) for chunk in chunks]
    metadata = [{"in_references": True} if _in_span(chunk.start, refs_span) else {} for chunk in chunks]
    return segments, metadata


def _chunk_id(prefix: str, index: int) -> str:
    return f"{prefix.upper()}_{index:04d}"


def _references_span(text: str) -> tuple[int, int] | None:
    """Char span ``[start, end)`` of the references/bibliography section, or None.
    Chunks starting inside it get flagged ``in_references`` so retrieval and the
    targeted verification search can skip citation lists (a cited title is not
    evidence that an element was reported). Matches in the first 30% of the
    document are ignored (prose mentions, tables of contents). The section ends at
    the next merged-document separator, so supplementary material appended after
    a paper's reference list is not mistaken for references."""
    from .documents import DOCUMENT_SEPARATOR_RE, REFERENCE_PATTERN

    haystack = text or ""
    match = REFERENCE_PATTERN.search(haystack)
    if not match or match.start() <= len(haystack) * 0.3:
        return None
    start = match.start()
    separator = DOCUMENT_SEPARATOR_RE.search(haystack, match.end())
    return start, separator.start() if separator else len(haystack)


def _in_span(offset: int, span: tuple[int, int] | None) -> bool:
    return span is not None and span[0] <= offset < span[1]


def _content_type_for_path(path: str) -> str:
    guessed, _encoding = mimetypes.guess_type(path)
    return guessed or "application/octet-stream"


def _build_text_pdf_render(text: str, *, title: str) -> bytes | None:
    if fitz is None or not (text or "").strip():
        return None
    doc = fitz.open()
    try:
        page_width = 612
        page_height = 792
        margin = 54
        line_rect = fitz.Rect(margin, margin, page_width - margin, page_height - margin)
        font_size = 10.5
        lineheight = 1.35

        def _insert(page, body: str):
            return page.insert_textbox(
                line_rect, body, fontsize=font_size, fontname="helv",
                lineheight=lineheight, color=(0, 0, 0),
            )

        def _fits(body: str) -> bool:
            # Probe the fit on a THROWAWAY page. page.clean_contents() does NOT erase
            # already-drawn text, so probing on the real page would overlay ~log2(N)
            # copies of the text → the re-extracted PDF is many times larger than the
            # input with sentences duplicated. Drawing each probe on a scratch page we
            # immediately delete keeps the real pages clean.
            scratch = doc.new_page(width=page_width, height=page_height)
            try:
                overflow = _insert(scratch, body)
                return not (isinstance(overflow, (int, float)) and overflow < 0)
            finally:
                doc.delete_page(len(doc) - 1)

        remaining = text.strip()
        guard = 0
        while remaining and guard < 10000:
            guard += 1
            # Binary-search the LARGEST prefix that fits without clipping (probing on
            # scratch pages), then draw exactly that prefix ONCE on a fresh real page.
            # insert_textbox returns >=0 when the text fits and a negative value when it
            # overflows.
            lo, hi, best = 1, len(remaining), 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if _fits(remaining[:mid]):
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            # Snap the cut to a paragraph/line/word boundary when one is reasonably
            # close, so we don't split mid-word.
            if best < len(remaining):
                window = remaining[:best]
                for sep in ("\n\n", "\n", " "):
                    pos = window.rfind(sep)
                    if pos >= int(best * 0.5):
                        best = pos + len(sep)
                        break
            page = doc.new_page(width=page_width, height=page_height)
            page.insert_text((margin, 28), title[:120], fontsize=9, color=(0.28, 0.31, 0.42))
            _insert(page, remaining[:best].strip())
            remaining = remaining[best:].lstrip()
        return doc.tobytes()
    finally:
        doc.close()


def _build_rendered_text_pdf_evidence_source(
    *,
    source_id: str,
    label: str,
    text: str,
    chunk_prefix: str,
    metadata: dict[str, Any] | None,
    max_chunk_tokens: int,
    embedding_model: str,
) -> dict[str, Any] | None:
    pdf_bytes = _build_text_pdf_render(text, title=label)
    if pdf_bytes is None:
        return None
    # Chunk the CANONICAL text — not the synthetic-PDF roundtrip — so the judge's
    # retrieval corpus, the manifest chunks, and the text pane all share ONE
    # segmentation with offsets into the same string (the roundtrip injects a page
    # header per page and rewraps every line, which used to leak into the corpus
    # on evidence-enabled runs but not on bare CLI runs). The synthetic PDF is
    # kept only as the Doc view: its page geometry + page text ride along in
    # render_data, and the viewer locates quotes there via its render-text search
    # fallback instead of pre-computed rects.
    payload = build_text_evidence_source(
        source_id=source_id,
        label=f"{label} Text Render",
        text=text,
        chunk_prefix=chunk_prefix,
        kind="pdf",
        metadata={"rendered_from_text": True, **(metadata or {})},
        max_chunk_tokens=max_chunk_tokens,
        embedding_model=embedding_model,
    )
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as handle:
        handle.write(pdf_bytes)
        handle.flush()
        doc = fitz.open(handle.name)
        try:
            roundtrip_text, pages = _pdf_page_text_and_offsets(doc)
            page_count = len(doc)
        finally:
            doc.close()
    page_entries = [
        {
            "page_number": page["page_number"],
            "width": page["width"],
            "height": page["height"],
            "start": page["start"],
            "end": page["end"],
        }
        for page in pages
    ]
    payload["source"]["render_mode"] = "pdf"
    payload["source"]["page_count"] = page_count
    payload["source"]["pages"] = page_entries
    payload["source"]["raw_filename"] = f"{source_id}-text-render.pdf"
    payload["raw_bytes"] = pdf_bytes
    payload["raw_content_type"] = "application/pdf"
    payload["render_data"] = {
        "kind": "pdf",
        "render_mode": "pdf",
        # Doc-view page mapping needs the roundtrip text (offsets align with the
        # page start/end entries); the TEXT pane reads display_text (canonical).
        "text": roundtrip_text,
        "display_text": _display_text(text),
        "pages": page_entries,
        "metadata": metadata or {},
    }
    return payload


def _plain_text_render_data(text: str, *, kind: str = "text", metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "kind": kind,
        "text": text or "",
        # For pure-text sources the display text IS the text; the field exists so
        # the viewer can uniformly prefer display_text over (possibly raw) text.
        "display_text": text or "",
        "metadata": metadata or {},
    }


def build_text_evidence_source(
    *,
    source_id: str,
    label: str,
    text: str,
    chunk_prefix: str,
    kind: str = "text",
    metadata: dict[str, Any] | None = None,
    raw_bytes: bytes | None = None,
    raw_content_type: str | None = None,
    raw_filename: str | None = None,
    max_chunk_tokens: int = 300,
    embedding_model: str = "text-embedding-3-large",
    reflow: bool = True,
) -> dict[str, Any]:
    # Prose reflows hard line-wraps for readability; structured JSON-row sources keep
    # their newlines (each row is "path\nvalue"), so they opt out with reflow=False.
    disp = _display_text if reflow else normalize_text
    chunks = extract_chunks_tokens_with_spans(
        text or "",
        max_chunk_tokens=max_chunk_tokens,
        encoding_name=embedding_model,
    )
    segments = [disp(chunk.text) for chunk in chunks]
    refs_span = _references_span(text or "")
    chunk_metadata: list[dict[str, Any]] = []
    manifest_chunks: dict[str, dict[str, Any]] = {}
    for index, chunk in enumerate(chunks, start=1):
        cid = _chunk_id(chunk_prefix, index)
        location = {
            "kind": "text",
            "start": chunk.start,
            "end": chunk.end,
        }
        item = {
            "id": cid,
            "source_id": source_id,
            "source_label": label,
            "source_kind": kind,
            "text": disp(chunk.text),
            "locations": [location],
            "relevance_scores_by_dimension": {},
        }
        if _in_span(chunk.start, refs_span):
            item["in_references"] = True
        manifest_chunks[cid] = item
        chunk_metadata.append(item)

    source = {
        "id": source_id,
        "label": label,
        "kind": kind,
        "render_mode": "text",
        "raw_filename": raw_filename,
        "metadata": metadata or {},
    }
    return {
        "source": source,
        "text": text or "",
        "segments": segments,
        "metadata": chunk_metadata,
        "chunks": manifest_chunks,
        "raw_bytes": raw_bytes,
        "raw_content_type": raw_content_type,
        "render_data": _plain_text_render_data(disp(text or ""), kind=kind, metadata=metadata),
    }


def _pdf_page_text_and_offsets(doc) -> tuple[str, list[dict[str, Any]]]:
    pages: list[dict[str, Any]] = []
    parts: list[str] = []
    offset = 0
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        page_text = page.get_text("text") or ""
        rect = page.rect
        if parts:
            parts.append("\n\n")
            offset += 2
        start = offset
        parts.append(page_text)
        offset += len(page_text)
        pages.append(
            {
                "page_number": page_index + 1,
                "start": start,
                "end": offset,
                "width": float(rect.width),
                "height": float(rect.height),
            }
        )
    return "".join(parts), pages


def _text_slice_for_page(text: str, chunk_start: int, chunk_end: int, page_info: dict[str, Any]) -> str:
    start = max(chunk_start, int(page_info["start"]))
    end = min(chunk_end, int(page_info["end"]))
    if start >= end:
        return ""
    return text[start:end].strip()


# Ligatures PyMuPDF may emit in extracted text; expanded before token comparison.
_LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "ft",
    "ﬆ": "st",
}
_TOKEN_STRIP_CHARS = string.punctuation + "“”‘’«»…·—–‐‑­"


def _norm_match_token(token: str) -> str:
    text = token
    for src, dst in _LIGATURES.items():
        text = text.replace(src, dst)
    text = text.replace("­", "")
    return text.strip(_TOKEN_STRIP_CHARS).lower()


def _word_sequence_rects(page, text: str) -> list[dict[str, float]]:
    """Locate `text` on the page by matching its word sequence against the
    page's extracted words, returning per-line bounding rects.

    Unlike page.search_for(), this is immune to line wraps, hyphenation, and
    ligature rendering, and it covers the entire matched span instead of a
    truncated prefix.
    """
    needle = [t for t in (_norm_match_token(w) for w in (text or "").split()) if t]
    if not needle:
        return []
    try:
        words = page.get_text("words")  # (x0, y0, x1, y1, word, block, line, word_no)
    except Exception:
        return []
    tokens: list[tuple[str, int]] = []
    for index, word in enumerate(words):
        norm = _norm_match_token(word[4])
        if norm:
            tokens.append((norm, index))
    if not tokens:
        return []

    values = [t[0] for t in tokens]
    n, m = len(values), len(needle)
    budget_total = max(1, m // 10)

    def walk(start: int, budget: int) -> tuple[int, int]:
        """Walk needle vs page tokens from `start` with a substitution budget.
        Returns (matched_count, consumed_page_tokens)."""
        matched = 0
        offset = 0
        for expected in needle:
            position = start + offset
            if position >= n:
                break
            if values[position] == expected:
                matched += 1
            elif budget > 0:
                budget -= 1
            else:
                break
            offset += 1
        return matched, offset

    best_start = -1
    best_consumed = 0
    best_matched = 0
    for start in range(n):
        if values[start] != needle[0]:
            continue
        matched, consumed = walk(start, budget_total)
        if matched > best_matched:
            best_matched, best_start, best_consumed = matched, start, consumed
            if matched == m:
                break
    if best_start < 0 or best_matched < max(1, int(m * 0.8)):
        return []

    # Merge the consumed words into one rect per (block, line).
    groups: dict[tuple[int, int], list[float]] = {}
    order: list[tuple[int, int]] = []
    for token_index in range(best_start, best_start + best_consumed):
        word = words[tokens[token_index][1]]
        key = (int(word[5]), int(word[6]))
        if key not in groups:
            groups[key] = [float(word[0]), float(word[1]), float(word[2]), float(word[3])]
            order.append(key)
        else:
            box = groups[key]
            box[0] = min(box[0], float(word[0]))
            box[1] = min(box[1], float(word[1]))
            box[2] = max(box[2], float(word[2]))
            box[3] = max(box[3], float(word[3]))
    return [
        {"x0": groups[key][0], "y0": groups[key][1], "x1": groups[key][2], "y1": groups[key][3]}
        for key in order
    ]


def _search_page_rects(page, text: str) -> list[dict[str, float]]:
    # Primary: word-sequence matching (full coverage, layout-robust).
    rects = _word_sequence_rects(page, text)
    if rects:
        return rects

    # Fallback: PyMuPDF geometric search, dehyphenated, then truncated needles.
    candidate = " ".join((text or "").split())
    if not candidate:
        return []
    attempts = [candidate]
    if len(candidate) > 240:
        attempts.append(candidate[:240].rsplit(" ", 1)[0] or candidate[:240])
    if len(candidate) > 120:
        attempts.append(candidate[:120].rsplit(" ", 1)[0] or candidate[:120])
    flags = getattr(fitz, "TEXT_DEHYPHENATE", 0) if fitz is not None else 0
    for attempt in attempts:
        try:
            rects = page.search_for(attempt, flags=flags)
        except TypeError:
            try:
                rects = page.search_for(attempt)
            except Exception:
                rects = []
        except Exception:
            rects = []
        if rects:
            return [
                {
                    "x0": float(rect.x0),
                    "y0": float(rect.y0),
                    "x1": float(rect.x1),
                    "y1": float(rect.y1),
                }
                for rect in rects
            ]
    return []


class _DocumentWords:
    """Every word of a PDF in reading order, normalised for matching, so a chunk of
    ANOTHER text (the parser's canonical output) can be placed on the pages: the
    chunk's words are aligned to this stream, tolerating words the PDF has and the
    canonical text lacks (running heads, page numbers, footnote marks at page breaks)
    and a minority of mismatched words. Built once per document."""

    _LOOKAHEAD = 15  # PDF-only words skipped between two matched chunk words
    _MAX_CANDIDATES = 40

    def __init__(self, doc) -> None:
        self.tokens: list[str] = []
        self.words: list[tuple[int, tuple]] = []  # (page_index, PyMuPDF word tuple)
        self.positions: dict[str, list[int]] = {}
        self.page_sizes: list[tuple[float, float]] = []
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            self.page_sizes.append((float(page.rect.width), float(page.rect.height)))
            try:
                words = page.get_text("words")
            except Exception:
                words = []
            for word in words:
                norm = _norm_match_token(word[4])
                if norm:
                    self.positions.setdefault(norm, []).append(len(self.tokens))
                    self.tokens.append(norm)
                    self.words.append((page_index, word))

    def _walk(self, needle: list[str], start: int) -> list[int]:
        matched: list[int] = []
        pos, misses = start, 0
        max_misses = int(len(needle) * 0.3) + 1
        n = len(self.tokens)
        for token in needle:
            limit = min(n, pos + self._LOOKAHEAD)
            j = pos
            while j < limit and self.tokens[j] != token:
                j += 1
            if j < limit:
                matched.append(j)
                pos = j + 1
            else:
                misses += 1
                if misses > max_misses:
                    break
        return matched

    def align(self, text: str, near: int = 0) -> list[int] | None:
        """Indices of the PDF words matching ``text`` (>= 70% of its words), searching
        candidate starts nearest to ``near`` first; None when it can't be placed."""
        needle = [t for t in (_norm_match_token(w) for w in (text or "").split()) if t]
        if not needle:
            return None
        best: list[int] = []
        for skip in range(min(3, len(needle))):  # tolerate a mismatched first word or two
            starts = sorted(
                self.positions.get(needle[skip], []),
                key=lambda i: (i < near, abs(i - near)),
            )[: self._MAX_CANDIDATES]
            for start in starts:
                matched = self._walk(needle[skip:], start)
                if len(matched) > len(best):
                    best = matched
                    if len(best) >= 0.95 * len(needle):
                        break
            if len(best) >= 0.7 * len(needle):
                break
        return best if len(best) >= max(1, int(0.7 * len(needle))) else None

    def locations(self, indices: list[int]) -> list[dict[str, Any]]:
        """Per-page rect locations (one rect per text line) for matched word indices."""
        pages: dict[int, dict[tuple[int, int], list[float]]] = {}
        for index in indices:
            page_index, word = self.words[index]
            lines = pages.setdefault(page_index, {})
            key = (int(word[5]), int(word[6]))
            box = lines.get(key)
            if box is None:
                lines[key] = [float(word[0]), float(word[1]), float(word[2]), float(word[3])]
            else:
                box[0], box[1] = min(box[0], float(word[0])), min(box[1], float(word[1]))
                box[2], box[3] = max(box[2], float(word[2])), max(box[3], float(word[3]))
        out = []
        for page_index in sorted(pages):
            width, height = self.page_sizes[page_index]
            out.append(
                {
                    "kind": "pdf",
                    "page": page_index + 1,
                    "page_width": width,
                    "page_height": height,
                    "rects": [
                        {"x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3]} for b in pages[page_index].values()
                    ],
                }
            )
        return out


def _pdf_locations_for_chunk(doc, full_text: str, chunk_start: int, chunk_end: int, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    locations: list[dict[str, Any]] = []
    for page_info in pages:
        if chunk_end < page_info["start"] or chunk_start > page_info["end"]:
            continue
        page_text = _text_slice_for_page(full_text, chunk_start, chunk_end, page_info)
        if not page_text:
            continue
        page = doc.load_page(int(page_info["page_number"]) - 1)
        rects = _search_page_rects(page, page_text)
        if rects:
            locations.append(
                {
                    "kind": "pdf",
                    "page": page_info["page_number"],
                    "page_width": page_info["width"],
                    "page_height": page_info["height"],
                    "rects": rects,
                }
            )
    return locations


def build_pdf_evidence_source(
    *,
    source_id: str,
    label: str,
    pdf_path: str,
    chunk_prefix: str,
    fallback_text: str | None = None,
    metadata: dict[str, Any] | None = None,
    max_chunk_tokens: int = 300,
    embedding_model: str = "text-embedding-3-large",
) -> dict[str, Any]:
    path = Path(pdf_path)
    raw_bytes = path.read_bytes() if path.exists() else None
    raw_filename = path.name or f"{source_id}.pdf"
    base_source = {
        "id": source_id,
        "label": label,
        "kind": "pdf",
        "raw_filename": raw_filename,
        "metadata": metadata or {},
    }

    if fitz is None or raw_bytes is None:
        return build_text_evidence_source(
            source_id=source_id,
            label=label,
            text=fallback_text or "",
            chunk_prefix=chunk_prefix,
            kind="text",
            metadata={"fallback_reason": "PDF rendering unavailable", **(metadata or {})},
            raw_bytes=raw_bytes,
            raw_content_type="application/pdf" if raw_bytes else None,
            raw_filename=raw_filename if raw_bytes else None,
            max_chunk_tokens=max_chunk_tokens,
            embedding_model=embedding_model,
        )

    try:
        doc = fitz.open(str(path))
    except Exception:
        return build_text_evidence_source(
            source_id=source_id,
            label=label,
            text=fallback_text or "",
            chunk_prefix=chunk_prefix,
            kind="text",
            metadata={"fallback_reason": "PDF could not be opened for highlighting", **(metadata or {})},
            raw_bytes=raw_bytes,
            raw_content_type="application/pdf",
            raw_filename=raw_filename,
            max_chunk_tokens=max_chunk_tokens,
            embedding_model=embedding_model,
        )
    try:
        pdf_text, pages = _pdf_page_text_and_offsets(doc)
        use_pdf_text = bool((pdf_text or "").strip())
        # The CANONICAL text is the selected parser's output (``fallback_text``), so
        # the judge, retrieval, and every run path read the same text; the PDF's own
        # text layer supplies page geometry only. Without parser text (direct
        # callers), the text layer is canonical as before.
        canonical_is_pdf = not (fallback_text or "").strip()
        source_text = pdf_text if canonical_is_pdf else (fallback_text or "")
        render_mode = "pdf" if use_pdf_text else "text"
        word_index = _DocumentWords(doc) if (use_pdf_text and not canonical_is_pdf) else None
        cursor = 0
        chunks = extract_chunks_tokens_with_spans(
            source_text,
            max_chunk_tokens=max_chunk_tokens,
            encoding_name=embedding_model,
        )
        segments = [_display_text(chunk.text) for chunk in chunks]
        refs_span = _references_span(source_text)
        chunk_metadata: list[dict[str, Any]] = []
        manifest_chunks: dict[str, dict[str, Any]] = {}
        for index, chunk in enumerate(chunks, start=1):
            cid = _chunk_id(chunk_prefix, index)
            if word_index is not None:
                # Parser text: place the chunk by aligning its words to the PDF's. An
                # unplaceable chunk carries no offsets (they'd index a different text);
                # the viewer then finds it by searching the page text.
                matched = word_index.align(chunk.text, near=cursor)
                locations = word_index.locations(matched) if matched else []
                if matched:
                    cursor = matched[0]
            elif use_pdf_text:
                locations = _pdf_locations_for_chunk(doc, source_text, chunk.start, chunk.end, pages)
                if not locations:
                    locations = [{"kind": "text", "start": chunk.start, "end": chunk.end}]
            else:
                locations = [{"kind": "text", "start": chunk.start, "end": chunk.end}]
            item = {
                "id": cid,
                "source_id": source_id,
                "source_label": label,
                "source_kind": "pdf",
                "text": _display_text(chunk.text),
                "locations": locations,
                "relevance_scores_by_dimension": {},
            }
            if _in_span(chunk.start, refs_span):
                item["in_references"] = True
            manifest_chunks[cid] = item
            chunk_metadata.append(item)

        source = {
            **base_source,
            "render_mode": render_mode,
            "page_count": len(doc),
            # start/end are char offsets into the source text, letting the
            # viewer map any text span to its page when rects are missing.
            "pages": [
                {
                    "page_number": page["page_number"],
                    "width": page["width"],
                    "height": page["height"],
                    "start": page["start"],
                    "end": page["end"],
                }
                for page in pages
            ],
        }
        render_data = {
            "kind": "pdf",
            "render_mode": render_mode,
            # In PDF mode the viewer maps search/rects via the RAW source_text
            # offsets, so `text` stays raw there; the text-mode PANE reads
            # `display_text` (reflowed — PDF hard line-wraps collapsed, paragraph
            # breaks kept), which matches the reflowed chunk text the locator
            # matches against. Text-mode fallback keeps both fields reflowed.
            # (Offsets/pages index the PDF text layer; the text pane shows the
            # canonical text the judge read.)
            "text": _display_text(source_text) if render_mode == "text" else pdf_text,
            "display_text": _display_text(source_text),
            "pages": source["pages"],
            "metadata": metadata or {},
        }
        return {
            "source": source,
            "text": source_text,
            "segments": segments,
            "metadata": chunk_metadata,
            "chunks": manifest_chunks,
            "raw_bytes": raw_bytes,
            "raw_content_type": "application/pdf",
            "render_data": render_data,
        }
    finally:
        doc.close()


def _flatten_json_rows(data: dict[str, Any], parent: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, value in data.items():
        path = f"{parent} → {key}" if parent else str(key)
        if isinstance(value, dict):
            rows.extend(_flatten_json_rows(value, path))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                list_path = f"{path}[{index}]"
                if isinstance(item, dict):
                    rows.extend(_flatten_json_rows(item, list_path))
                elif item not in [None, ""]:
                    rows.append({"path": list_path, "value": str(item)})
        elif value not in [None, ""]:
            rows.append({"path": path, "value": str(value)})
    return rows


def build_json_evidence_source(
    *,
    source_id: str,
    label: str,
    data: dict[str, Any],
    chunk_prefix: str,
    metadata: dict[str, Any] | None = None,
    max_chunk_tokens: int = 300,
    embedding_model: str = "text-embedding-3-large",
) -> dict[str, Any]:
    rows = _flatten_json_rows(data)
    text_parts: list[str] = []
    offset = 0
    render_rows: list[dict[str, Any]] = []
    for row in rows:
        line = f"{row['path']}\n{row['value']}"
        if text_parts:
            text_parts.append("\n\n")
            offset += 2
        start = offset
        text_parts.append(line)
        offset += len(line)
        render_rows.append(
            {
                "path": row["path"],
                "value": row["value"],
                "start": start,
                "end": offset,
            }
        )
    text = "".join(text_parts)
    payload = build_text_evidence_source(
        source_id=source_id,
        label=label,
        text=text,
        chunk_prefix=chunk_prefix,
        kind="json",
        metadata=metadata,
        raw_bytes=text.encode("utf-8"),
        raw_content_type="text/plain; charset=utf-8",
        raw_filename=f"{source_id}.txt",
        max_chunk_tokens=max_chunk_tokens,
        embedding_model=embedding_model,
        reflow=False,   # JSON rows ("path\nvalue") rely on their newlines — don't reflow
    )
    payload["source"]["kind"] = "json"
    payload["source"]["render_mode"] = "json"
    payload["render_data"] = {
        "kind": "json",
        "text": text,
        "rows": render_rows,
        "metadata": metadata or {},
    }
    for chunk in payload["chunks"].values():
        chunk["source_kind"] = "json"
        for location in chunk["locations"]:
            location["kind"] = "json"
    return payload


def build_file_evidence_source(
    *,
    source_id: str,
    label: str,
    file_path: str,
    file_ext: str,
    text: str,
    chunk_prefix: str,
    metadata: dict[str, Any] | None = None,
    max_chunk_tokens: int = 300,
    embedding_model: str = "text-embedding-3-large",
) -> dict[str, Any]:
    ext = (file_ext or Path(file_path).suffix).lower()
    if ext == ".pdf":
        return build_pdf_evidence_source(
            source_id=source_id,
            label=label,
            pdf_path=file_path,
            chunk_prefix=chunk_prefix,
            fallback_text=text,
            metadata=metadata,
            max_chunk_tokens=max_chunk_tokens,
            embedding_model=embedding_model,
        )

    if ext in {".docx", ".txt", ".html", ".htm", ".xml"}:
        rendered_pdf = _build_rendered_text_pdf_evidence_source(
            source_id=source_id,
            label=label,
            text=text,
            chunk_prefix=chunk_prefix,
            metadata={
                "original_extension": ext,
                "original_filename": Path(file_path).name,
                **(metadata or {}),
            },
            max_chunk_tokens=max_chunk_tokens,
            embedding_model=embedding_model,
        )
        if rendered_pdf is not None:
            return rendered_pdf

    path = Path(file_path)
    raw_bytes = path.read_bytes() if path.exists() else None
    return build_text_evidence_source(
        source_id=source_id,
        label=label,
        text=text,
        chunk_prefix=chunk_prefix,
        kind="text",
        metadata=metadata,
        raw_bytes=raw_bytes,
        raw_content_type=_content_type_for_path(str(path)) if raw_bytes is not None else None,
        raw_filename=path.name if raw_bytes is not None else None,
        max_chunk_tokens=max_chunk_tokens,
        embedding_model=embedding_model,
    )
