import fitz
import pytest

from backend.services.documents import (
    convert_txt_to_pdf,
    extract_text_from_html,
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
