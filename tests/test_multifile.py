from __future__ import annotations

import asyncio
import io

from fastapi import UploadFile

from backend.routes.comparisons import _coalesce_uploads, _non_empty_uploads


def _run(coro):
    # Use a private loop and DON'T touch the global current loop, so legacy tests that
    # still call asyncio.get_event_loop() aren't left with a closed/None loop.
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _upload(name: str, content: str) -> UploadFile:
    data = content.encode("utf-8")
    return UploadFile(file=io.BytesIO(data), filename=name, size=len(data))


def test_non_empty_uploads_filters_placeholders():
    real = _upload("a.txt", "x")
    blank = _upload("", "")
    assert _non_empty_uploads(None) == []
    assert _non_empty_uploads([blank]) == []
    assert _non_empty_uploads(real) == [real]          # single value, not a list
    assert _non_empty_uploads([real, blank]) == [real]


def test_coalesce_single_file_is_returned_untouched(tmp_path):
    f = _upload("paper.pdf", "ignored")
    out = _run(_coalesce_uploads([f], upload_dir=tmp_path, kind="paper"))
    assert out is f  # single upload keeps its native parser/PDF path


def test_coalesce_none_or_empty_returns_none(tmp_path):
    assert _run(_coalesce_uploads([], upload_dir=tmp_path, kind="paper")) is None
    assert _run(_coalesce_uploads(None, upload_dir=tmp_path, kind="paper")) is None


def test_coalesce_multiple_files_concatenated_with_separators(tmp_path):
    async def run():
        files = [_upload("a.txt", "First document body."), _upload("b.txt", "Second document body.")]
        out = await _coalesce_uploads(files, upload_dir=tmp_path, kind="paper")
        return out, (await out.read()).decode("utf-8")

    out, text = _run(run())
    assert out.filename == "paper-combined.txt"
    assert "===== Document 1: a.txt =====" in text
    assert "First document body." in text
    assert "===== Document 2: b.txt =====" in text
    assert "Second document body." in text
    # the combined upload is consumable by the normal storage path (read drains it)
    assert text.index("First document body.") < text.index("Second document body.")
