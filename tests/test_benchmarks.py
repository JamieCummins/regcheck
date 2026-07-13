"""Structural integrity of the frozen verdict benchmark (no LLM calls here —
the live run is benchmarks/run_probes.py). Guards against a probe being edited
or deleted without its manifest entry, and vice versa."""
from __future__ import annotations

import json
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1] / "benchmarks"


def _manifest():
    return json.loads((BENCH / "manifest.json").read_text(encoding="utf-8"))


def test_manifest_and_probe_files_agree():
    manifest = _manifest()
    names = {p["name"] for p in manifest["probes"]}
    assert len(names) == len(manifest["probes"]), "duplicate probe names"
    on_disk = {d.name for d in (BENCH / "probes").iterdir() if d.is_dir()}
    assert names == on_disk, f"manifest/probes mismatch: {names ^ on_disk}"


def test_probe_entries_are_complete():
    for probe in _manifest()["probes"]:
        assert probe["expected"] in {"yes", "no", "missing"}, probe["name"]
        assert probe["rule"].startswith("Rule "), probe["name"]
        assert probe["dimension"].strip() and probe["dimension_definition"].strip()
        probe_dir = BENCH / "probes" / probe["name"]
        for fname in ("registration.txt", "paper.txt"):
            text = (probe_dir / fname).read_text(encoding="utf-8")
            assert len(text.split()) >= 80, f"{probe['name']}/{fname} suspiciously short"


def test_verdict_expectations_cover_all_three_outcomes():
    expected = [p["expected"] for p in _manifest()["probes"]]
    assert {"yes", "no", "missing"} <= set(expected)
