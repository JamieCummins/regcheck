"""Live cost tracking, chain-of-thought capture, and the CLI batch runner."""
from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("DEEPSEEK_API_KEY", "test")
os.environ.setdefault("CLAUDE_API_KEY", "test")

from backend.services import comparisons  # noqa: E402
from backend.services import cost_tracking as ct  # noqa: E402


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── tracker maths ──────────────────────────────────────────────────────────────


def test_tracker_prices_known_models():
    t = ct.CostTracker()
    t.record_llm("gpt-5.5-2026-05-preview", 1_000_000, 100_000)  # prefix match
    t.record_embedding("text-embedding-3-large", 2_000_000)
    snap = t.snapshot()
    assert snap["input_tokens"] == 1_000_000
    assert snap["output_tokens"] == 100_000
    assert snap["embedding_tokens"] == 2_000_000
    assert snap["llm_usd"] == pytest.approx(1.25 + 1.0, abs=0.01)
    assert snap["embedding_usd"] == pytest.approx(0.26, abs=0.01)
    assert snap["estimate_complete"] is True


def test_tracker_flags_unpriced_models():
    t = ct.CostTracker()
    t.record_llm("some-unknown-model", 1000, 1000)
    snap = t.snapshot()
    assert snap["input_tokens"] == 1000
    assert snap["estimate_complete"] is False


def test_pricing_env_override(monkeypatch):
    monkeypatch.setenv("COST_PRICING_JSON", '{"some-unknown-model": [1.0, 2.0]}')
    t = ct.CostTracker()
    t.record_llm("some-unknown-model", 1_000_000, 1_000_000)
    snap = t.snapshot()
    assert snap["llm_usd"] == pytest.approx(3.0)
    assert snap["estimate_complete"] is True


def test_record_llm_usage_accepts_both_shapes():
    t = ct.start_run()
    ct.record_llm_usage("gpt-5.5", SimpleNamespace(prompt_tokens=10, completion_tokens=5))
    ct.record_llm_usage("claude-opus-4-8", SimpleNamespace(input_tokens=7, output_tokens=3))
    snap = t.snapshot()
    assert snap["input_tokens"] == 17
    assert snap["output_tokens"] == 8
    assert snap["llm_calls"] == 2


# ── chain-of-thought capture ───────────────────────────────────────────────────


def test_inline_think_block_is_stashed_and_stripped():
    from backend.services.llm import _strip_deepseek_reasoning

    ct.pop_reasoning()
    out = _strip_deepseek_reasoning("<think>step by step reasoning</think>The answer")
    assert out == "The answer"
    assert ct.pop_reasoning() == "step by step reasoning"
    assert ct.pop_reasoning() == ""  # popped once, then empty


def test_judge_attaches_chain_of_thought(monkeypatch):
    def _dispatch(messages, **_kw):
        ct.stash_reasoning("model reasoned about sample sizes here")
        return json.dumps(
            {
                "dimension": "Sample size",
                "paper_content_quotes": "",
                "paper_content_summary": "",
                "registration_content_quotes": "",
                "registration_content_summary": "",
                "deviation_judgement": "no",
                "deviation_information": "matches",
                "unlocated_in_paper": "",
                "unlocated_in_registration": "",
            }
        )

    monkeypatch.setattr(comparisons, "_dispatch_judgement", _dispatch)
    item = comparisons._judge_dimension_once(
        [{"role": "user", "content": "x"}],
        client_choice="openai",
        dimension_query="Sample size",
        paper_top=[],
        prereg_top=[],
        reasoning_effort=None,
    )
    assert item.chain_of_thought == "model reasoned about sample sizes here"
    # History replay must never include the chain of thought.
    assert "chain_of_thought" not in json.dumps(item.model_dump(exclude={"chain_of_thought"}))


def test_openai_response_schema_excludes_chain_of_thought():
    fields = comparisons._ComparisonResponseSchema.model_fields
    assert "chain_of_thought" not in fields
    assert "deviation_judgement" in fields


# ── orchestrator cost attachment ───────────────────────────────────────────────


def test_quality_orchestrator_attaches_cost(tmp_path):
    from backend.services import registration_quality as rq

    reg = tmp_path / "r.txt"
    reg.write_text("We will recruit 300 participants.", encoding="utf-8")

    def fake_runner(text, client, dimension, **kwargs):
        tracker = ct.current()
        assert tracker is not None  # runner executes inside the run's context
        tracker.record_llm("gpt-5.5", 1000, 200)
        return comparisons.ComparisonResult(
            items=[comparisons.ComparisonItem(dimension=dimension, deviation_judgement="partial")]
        )

    result = _run(
        rq.registration_quality_assessment(
            str(reg), ".txt", "openai", "pymupdf",
            selected_dimensions=[{"dimension": "Sample size", "definition": "n"}],
            assessment_runner=fake_runner,
        )
    )
    assert result.cost is not None
    assert result.cost["input_tokens"] == 1000
    assert result.cost["total_usd"] > 0


# ── CLI batch runner ───────────────────────────────────────────────────────────


def test_batch_runs_rows_and_writes_summary(tmp_path, monkeypatch):
    from backend import cli

    manifest = tmp_path / "jobs.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["type", "preregistration", "paper", "label"])
        writer.writeheader()
        writer.writerow({"type": "quality", "preregistration": "a.docx", "label": "regA"})
        writer.writerow({"type": "general", "preregistration": "b.docx", "paper": "b_paper.pdf", "label": "pairB"})
        writer.writerow({"type": "quality", "preregistration": "broken.docx", "label": "regC"})

    seen = []

    async def fake_quality(ns):
        seen.append(("quality", ns.preregistration))
        if "broken" in (ns.preregistration or ""):
            raise ValueError("boom")
        return {"items": [{"dimension": "d"}], "cost": {"total_usd": 0.02}}

    async def fake_general(ns):
        seen.append(("general", ns.paper))
        return {"items": [{"dimension": "d"}], "cost": {"total_usd": 0.10}}

    monkeypatch.setattr(cli, "_run_quality", fake_quality)
    monkeypatch.setattr(cli, "_run_general", fake_general)

    out_dir = tmp_path / "out"
    args = argparse.Namespace(
        manifest=str(manifest),
        output_dir=str(out_dir),
        client="openai",
        parser_choice="pymupdf",
        append_previous_output=False,
        reasoning_effort="medium",
        output_format="json",
        stop_on_error=False,
        embedding_model=None,
    )
    summary = _run(cli._run_batch(args))

    assert summary["total"] == 3
    assert summary["succeeded"] == 2
    assert summary["failed"] == ["regC"]
    assert summary["estimated_total_usd"] == pytest.approx(0.12)
    assert (out_dir / "regA.json").exists()
    assert (out_dir / "pairB.json").exists()
    assert not (out_dir / "regC.json").exists()
    written = json.loads((out_dir / "batch_summary.json").read_text())
    assert written["failed"] == ["regC"]
    assert [s[0] for s in seen] == ["quality", "general", "quality"]


def test_batch_stop_on_error(tmp_path, monkeypatch):
    from backend import cli

    manifest = tmp_path / "jobs.csv"
    manifest.write_text("type,preregistration,label\nquality,x.docx,one\nquality,y.docx,two\n", encoding="utf-8")

    async def fake_quality(ns):
        raise ValueError("nope")

    monkeypatch.setattr(cli, "_run_quality", fake_quality)
    args = argparse.Namespace(
        manifest=str(manifest),
        output_dir=str(tmp_path / "out"),
        client="openai",
        parser_choice="pymupdf",
        append_previous_output=False,
        reasoning_effort="medium",
        output_format="json",
        stop_on_error=True,
        embedding_model=None,
    )
    summary = _run(cli._run_batch(args))
    assert summary["total"] == 1  # aborted after the first failure
    assert summary["failed"] == ["one"]


def test_cli_csv_output_includes_chain_of_thought(tmp_path, capsys):
    from backend import cli

    payload = {
        "items": [
            {
                "dimension": "d",
                "deviation_judgement": "no",
                "deviation_information": "r",
                "chain_of_thought": "why I think so",
            }
        ]
    }
    out = tmp_path / "x.csv"
    cli._write_output(payload, str(out), "csv")
    text = out.read_text()
    reader = csv.DictReader(io.StringIO(text))
    row = next(reader)
    assert row["chain_of_thought"] == "why I think so"
