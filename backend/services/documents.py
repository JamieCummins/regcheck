from __future__ import annotations

import re
import uuid
from html.parser import HTMLParser
from pathlib import Path

import docx
import fitz

from .text_normalize import decode_bytes

__all__ = [
    "extract_text_from_docx",
    "extract_text_from_pdf",
    "extract_text_from_html",
    "convert_txt_to_pdf",
    "clean_document_text",
    "read_file",
    "read_file_as_pdf",
]


UPLOADS_DIR = Path("uploads")


def extract_text_from_docx(file_path: str) -> str:
    """Extract plain text from a DOCX document."""
    document = docx.Document(file_path)
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from all pages of a PDF document."""
    document = fitz.open(file_path)
    text = ""
    for page in document:
        text += page.get_text()
    return text


def convert_txt_to_pdf(txt_path: str) -> str:
    """Convert a TXT file into a PDF stored near the source file with a unique name.

    Rendered with PyMuPDF rather than a Latin-1-only core-font engine, so the
    scientific characters that pervade real text — em dashes, curly quotes,
    Greek letters, accented names, ≤/×/μ — survive instead of raising an
    encoding error. This is the same rendering path used for DOCX/HTML uploads.
    """
    text = _read_txt(txt_path).strip()

    page_width, page_height, margin = 612, 792, 54
    font_size = 11.0
    rect = fitz.Rect(margin, margin, page_width - margin, page_height - margin)

    doc = fitz.open()
    try:
        remaining = text
        if not remaining:
            # Always emit at least one (blank) page so downstream PDF handling
            # has a valid document.
            doc.new_page(width=page_width, height=page_height)
        guard = 0
        while remaining and guard < 10000:
            guard += 1
            page = doc.new_page(width=page_width, height=page_height)
            # Binary-search the largest prefix that fits, then advance by exactly that
            # — a fixed char slice could overflow a page (text with many short lines),
            # clipping + dropping the overflow → missing content + near-blank pages.
            lo, hi, best = 1, len(remaining), 1
            while lo <= hi:
                mid = (lo + hi) // 2
                page.clean_contents()
                overflow = page.insert_textbox(
                    rect, remaining[:mid], fontsize=font_size, fontname="helv",
                    lineheight=1.35, color=(0, 0, 0),
                )
                if isinstance(overflow, (int, float)) and overflow < 0:
                    hi = mid - 1
                else:
                    best = mid
                    lo = mid + 1
            if best < len(remaining):
                window = remaining[:best]
                for sep in ("\n\n", "\n", " "):
                    pos = window.rfind(sep)
                    if pos >= int(best * 0.5):
                        best = pos + len(sep)
                        break
            page.clean_contents()
            page.insert_textbox(
                rect, remaining[:best].strip(), fontsize=font_size,
                fontname="helv", lineheight=1.35, color=(0, 0, 0),
            )
            remaining = remaining[best:].lstrip()

        stem = Path(txt_path).stem or "converted_txt"
        target_dir = Path(txt_path).parent if Path(txt_path).parent else UPLOADS_DIR
        output_path = target_dir / f"{stem}_{uuid.uuid4().hex}.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(output_path))
        return str(output_path)
    finally:
        doc.close()


class _HTMLTextExtractor(HTMLParser):
    """Collect readable text from HTML, skipping script/style and inserting line
    breaks on block-level tags so paragraph structure survives."""

    _SKIP = {"script", "style", "head", "noscript", "template"}
    _BLOCK = {
        "p", "br", "div", "li", "ul", "ol", "tr", "table", "section", "article",
        "header", "footer", "blockquote", "pre", "h1", "h2", "h3", "h4", "h5", "h6",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def extract_text_from_html(file_path: str) -> str:
    """Extract readable text from an HTML document (stdlib parser; no JS execution)."""
    markup = decode_bytes(Path(file_path).read_bytes())  # charset-robust (UTF-8 -> cp1252)
    extractor = _HTMLTextExtractor()
    extractor.feed(markup)
    return extractor.text()


REFERENCE_PATTERN = re.compile(
    r"(?:^|\n)([A-Z\s]*\bReferences\b|Bibliography|Cited Works)[\s]*\n",
    re.IGNORECASE,
)
INTRODUCTION_PATTERN = re.compile(
    r"(?:^|\n)([A-Z\s]*\bIntroduction\b)[\s]*\n", re.IGNORECASE
)


def remove_references(document_text: str) -> str:
    """Remove trailing references sections from a document."""
    match = REFERENCE_PATTERN.search(document_text)
    if match:
        return document_text[: match.start()]
    return document_text


def clean_document_text(document_text: str) -> str:
    """Trim boilerplate sections from parsed documents."""
    document_text = _normalize_whitespace(document_text)
    match = INTRODUCTION_PATTERN.search(document_text)
    if match:
        document_text = document_text[match.start() :]
    return remove_references(document_text)


def _read_txt(path: str) -> str:
    return decode_bytes(Path(path).read_bytes())  # charset-robust (UTF-8 -> cp1252)


def read_file(file_path: str, file_extension: str) -> str:
    """Read a document and normalise its text content."""
    file_extension = file_extension.lower()
    if file_extension == ".txt":
        text = _read_txt(file_path)
    elif file_extension == ".docx":
        text = extract_text_from_docx(file_path)
    elif file_extension == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif file_extension in (".html", ".htm"):
        text = extract_text_from_html(file_path)
    else:
        raise ValueError(
            f"Unsupported file type '{file_extension}'. Please upload a PDF, DOCX, TXT, or HTML file."
        )
    return clean_document_text(text)


def read_file_as_pdf(filename: str, file_extension: str) -> str:
    """Return a path to a PDF representation of the provided document."""
    file_extension = file_extension.lower()
    if file_extension == ".txt":
        return convert_txt_to_pdf(filename)
    if file_extension in (".docx", ".html", ".htm"):
        # Text is extracted separately and re-rendered to PDF pages by the
        # evidence layer (build_file_evidence_source); keep the original file
        # so its bytes remain available for the "open original" fallback.
        return filename
    if file_extension == ".pdf":
        return filename
    raise ValueError(
        f"Unsupported file type '{file_extension}'. Please upload a PDF, DOCX, TXT, or HTML file."
    )


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace to reduce mid-quote line breaks without losing content."""
    if not isinstance(text, str):
        return text
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Drop standalone page numbers (lines that are only digits)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    # Collapse consecutive blank lines to a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    # Keep at most double newlines to preserve some paragraph structure
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
