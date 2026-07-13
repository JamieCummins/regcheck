from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

TEST_SCRIPT = Path(__file__).resolve().parent / "js" / "quote_locator_test.mjs"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_quote_locator_js_suite():
    """Run the node test suite for the front-end quote locator
    (static/js/quote-locator.js) — the client tier of evidence tracing."""
    result = subprocess.run(
        ["node", str(TEST_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"locator tests failed:\n{result.stdout}\n{result.stderr}"
