import fitz
import pytest

from backend.services.documents import (
    convert_txt_to_pdf,
    extract_text_from_html,
    extract_text_from_xml,
    read_file,
    read_file_as_pdf,
)


def _write(path, content: str) -> str:
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_extract_text_from_html_strips_tags_and_scripts(tmp_path):
    html = """
    <html><head><title>T</title><style>.x{color:red}</style></head>
    <body>
      <h1>Hypotheses</h1>
      <p>We predict <b>X</b> &gt; Y.</p>
      <script>console.log('ignore me')</script>
      <ul><li>One</li><li>Two</li></ul>
    </body></html>
    """
    text = extract_text_from_html(_write(tmp_path / "p.html", html))
    flat = " ".join(text.split())
    assert "Hypotheses" in flat
    assert "We predict X > Y." in flat
    assert "One" in flat and "Two" in flat
    assert "console.log" not in flat  # <script> contents skipped
    assert "color:red" not in flat    # <style> contents skipped


def test_read_file_handles_html(tmp_path):
    path = _write(
        tmp_path / "p.html",
        "<html><body><h2>Introduction</h2><p>Body text here.</p></body></html>",
    )
    out = read_file(path, ".html")
    assert "Body text here." in out


def test_read_file_as_pdf_keeps_html_original(tmp_path):
    # HTML (like DOCX) is returned as-is; the evidence layer re-renders the
    # extracted text into PDF pages, so the original bytes stay available.
    path = _write(tmp_path / "p.html", "<p>hi</p>")
    assert read_file_as_pdf(path, ".html") == path


def test_extract_text_from_xml_jats_structure(tmp_path):
    # JATS-style article XML: structural tags become paragraph boundaries,
    # inline formatting tags (<italic>, <xref>) stay within the sentence.
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <article>
      <front><article-meta><title-group>
        <article-title>A preregistered study</article-title>
      </title-group></article-meta></front>
      <body>
        <sec><title>Hypotheses</title>
          <p>We predict <italic>X</italic> &gt; Y <xref ref-type="bibr">(Smith, 2020)</xref>.</p>
        </sec>
      </body>
    </article>
    """
    text = extract_text_from_xml(_write(tmp_path / "p.xml", xml))
    assert "A preregistered study" in text
    # Inline tags must not split the sentence apart.
    assert "We predict X > Y (Smith, 2020)." in " ".join(text.split())
    # Structural tags must produce a boundary between title and paragraph.
    assert "Hypotheses\n" in text.replace("\n\n", "\n")


def test_extract_text_from_xml_handles_namespaces(tmp_path):
    xml = '<a:doc xmlns:a="urn:x"><a:p>Namespaced body text.</a:p></a:doc>'
    text = extract_text_from_xml(_write(tmp_path / "n.xml", xml))
    assert "Namespaced body text." in text


def test_extract_text_from_xml_malformed_falls_back_to_soup(tmp_path):
    # Not well-formed (unclosed tag) → tolerant HTML-soup fallback still
    # recovers the text rather than raising.
    xml = "<article><p>Recovered content<p>Second paragraph</article>"
    text = extract_text_from_xml(_write(tmp_path / "bad.xml", xml))
    assert "Recovered content" in text
    assert "Second paragraph" in text


def test_read_file_handles_xml(tmp_path):
    path = _write(
        tmp_path / "p.xml",
        "<article><sec><title>Introduction</title><p>Body text here.</p></sec></article>",
    )
    out = read_file(path, ".xml")
    assert "Body text here." in out


def test_read_file_as_pdf_keeps_xml_original(tmp_path):
    # XML (like DOCX/HTML) is returned as-is; the evidence layer re-renders the
    # extracted text into PDF pages.
    path = _write(tmp_path / "p.xml", "<doc><p>hi</p></doc>")
    assert read_file_as_pdf(path, ".xml") == path


def test_remove_references_keeps_content_after_early_heading():
    # An OSF registration whose description cites literature under an early
    # "References" heading (real example: osf.io/v734e). The form content after
    # it is the bulk of the document and must NOT be discarded.
    body = "\n\n".join(f"Q{i}: registered answer number {i} with substantial detail." for i in range(2, 24))
    doc = f"Description of the study.\n\nReferences\nSmith (2020). A cited work.\n\n{body}"
    from backend.services.documents import remove_references

    assert remove_references(doc) == doc


def test_remove_references_cuts_trailing_section():
    body = "\n\n".join(f"Section {i}: methods and analysis details." for i in range(1, 15))
    refs = "\nReferences\nSmith (2020). A cited work.\nJones (2021). Another.\n"
    doc = body + refs
    from backend.services.documents import remove_references

    out = remove_references(doc)
    assert "Smith (2020)" not in out
    assert "Section 14" in out


def test_remove_references_cuts_at_last_heading_not_first():
    # Early prose mention of a References heading plus a genuine trailing
    # section: only the trailing one is stripped.
    body = "\n\n".join(f"Item {i}: registered content." for i in range(1, 20))
    doc = f"Intro.\nReferences\nEarly citation list mention.\n\n{body}\nReferences\nSmith (2020).\n"
    from backend.services.documents import remove_references

    out = remove_references(doc)
    assert "Item 19" in out
    assert "Early citation list mention." in out
    assert "Smith (2020)" not in out


def test_read_file_rejects_unknown_extension(tmp_path):
    path = _write(tmp_path / "p.xyz", "data")
    with pytest.raises(ValueError, match="Unsupported file type"):
        read_file(path, ".xyz")


def test_convert_txt_to_pdf_handles_unicode(tmp_path):
    # Scientific text routinely contains non-Latin-1 characters that the old
    # core-font renderer could not encode. This must produce a valid PDF with
    # extractable text instead of raising.
    content = (
        "Effect of treatment—a pre–post design.\n\n"
        "We found α = .05, μ ≤ 0.2, n × 2 “blinded” raters; "
        "Müller et al. report β weights.\n"
    )
    txt = _write(tmp_path / "doc.txt", content)
    out = convert_txt_to_pdf(txt)

    assert out.endswith(".pdf")
    doc = fitz.open(out)
    try:
        assert doc.page_count >= 1
        rendered = "".join(page.get_text() for page in doc)
    finally:
        doc.close()
    # ASCII content always survives; this also proves no encoding crash occurred.
    assert "Effect of treatment" in rendered
    assert "blinded" in rendered
