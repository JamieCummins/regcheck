"""Label-based study isolation: segmentation, label handling, deterministic filter."""
from __future__ import annotations

import asyncio
import json

import pytest

from backend.services import comparisons
from backend.services import study_labels as sl

# A PDF-text-layer-like paper: hard-wrapped lines, headings on their own lines, and
# no blank lines between paragraphs.
PAPER = """Two Experiments on Testing
Jane Doe and John Roe
Introduction
Background theory is general and applies to all
of the research reported here. In Experiment 2 we expected larger effects than before.
Experiment 1
Participants were 100 students recruited through Prolific. They completed task A online
and were paid two pounds.
Results
The effect was significant in Experiment 1, t(98) = 2.5.
Experiment 2
Our sampling approach was the same as in Experiment 1. Two hundred adults took part
and completed task B.
General Discussion
Both experiments support the theory in general terms.
References
Smith, J. (2020). A cited paper. Journal of Tests."""


def _pid(paras, startswith):
    return next(p.pid for p in paras if p.text.startswith(startswith))


def _labels(paras):
    p = lambda s: _pid(paras, s)  # noqa: E731
    intro_body = p("Background theory")
    return {
        "studies": [{"id": "S1", "label": "Experiment 1"}, {"id": "S2", "label": "Experiment 2"}],
        "paragraphs": [
            {"from": 1, "to": p("Introduction") - 1, "label": "front"},
            {"from": p("Introduction"), "to": intro_body, "label": "shared"},
            {"from": p("Experiment 1"), "to": p("The effect"), "label": ["S1"]},
            # Boundary heading wrongly put in the previous study's range: the guard fixes it.
            {"from": p("Experiment 2"), "to": p("Experiment 2"), "label": ["S1"]},
            {"from": p("Our sampling"), "to": p("Our sampling"), "label": ["S2"]},
            {"from": p("General Discussion"), "to": p("Both experiments"), "label": "shared"},
            {"from": p("References"), "to": len(paras), "label": "front"},
        ],
        "sentences": [{"id": f"{intro_body}.2", "label": ["S2"]}],
        "cross_refs": [{
            "citing": f"{p('Our sampling')}.1", "study": "S2", "referenced_study": "S1",
            "aspect": "sampling", "referenced": [f"{p('Participants were 100')}.1"],
        }],
    }


def test_segment_handles_wrapped_text_without_blank_lines():
    paras = sl.segment(PAPER)
    texts = [p.text for p in paras]
    assert "Introduction" in texts and "Experiment 1" in texts and "General Discussion" in texts
    intro = paras[_pid(paras, "Background theory") - 1]
    assert "applies to all of the research" in intro.text          # hard wrap joined
    assert len(intro.sentences) == 2
    assert next(p for p in paras if p.text == "Experiment 2").block


def test_segment_splits_long_paragraphs_and_keeps_tables():
    long = " ".join(f"Sentence number {i} is here." for i in range(20))
    paras = sl.segment(long + "\n\nWave | N\nTime 4 (2019) | 633")
    assert [len(p.sentences) for p in paras[:3]] == [8, 8, 4]
    assert paras[-1].block and paras[-1].text == "Wave | N\nTime 4 (2019) | 633"


def test_isolate_keeps_shared_and_target_with_scoped_cross_refs():
    paras = sl.segment(PAPER)
    out, stats = sl.isolate(paras, _labels(paras), "S2")
    assert "Background theory" in out
    assert "In Experiment 2 we expected larger effects" in out       # override: S2's sentence
    assert "Experiment 2" in out.splitlines()                        # heading guard
    assert "Two hundred adults took part" in out
    assert "Both experiments support the theory" in out
    assert "t(98) = 2.5" not in out                                  # other study's result
    assert "Jane Doe" not in out and "Smith, J." not in out          # front matter + refs
    assert "[From Experiment 1, cited for: sampling] Participants were 100 students" in out
    assert "They completed task A" not in out                        # only the cited aspect
    assert stats["imported_ref_sentences"] == 1

    out1, _ = sl.isolate(paras, _labels(paras), "S1")
    assert "In Experiment 2 we expected" not in out1
    assert "t(98) = 2.5" in out1 and "Two hundred adults" not in out1


@pytest.mark.parametrize(
    "user,expected",
    [("2", "S2"), ("Study 2", "S2"), ("experiment 2", "S2"), ("S1", "S1"), ("3", None), ("", None)],
)
def test_resolve_target(user, expected):
    labels = {"studies": [{"id": "S1", "label": "Experiment 1"}, {"id": "S2", "label": "Experiment 2"}]}
    assert sl.resolve_target(labels, user) == expected


def test_resolve_target_single_study_paper():
    assert sl.resolve_target({"studies": [{"id": "S1", "label": "The study"}]}, "Study 1") == "S1"


def test_validate_labels():
    paras = sl.segment(PAPER)
    assert sl.validate_labels(paras, _labels(paras)) == []
    assert sl.validate_labels(paras, {"studies": []}) == ["no studies identified"]
    partial = {"studies": [{"id": "S1"}], "paragraphs": [{"from": 1, "to": 2, "label": "front"}]}
    assert "paragraphs labelled" in sl.validate_labels(paras, partial)[0]
    assert sl.validate_labels(paras, ["not", "a", "dict"])


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_label_isolation_end_to_end_and_label_cache(tmp_path, monkeypatch):
    paras = sl.segment(PAPER)
    calls = []

    def fake_llm(messages, **kw):
        calls.append(messages)
        return "```json\n" + json.dumps(_labels(paras)) + "\n```"

    monkeypatch.setattr(comparisons, "_isolation_llm_text", fake_llm)
    monkeypatch.setattr(comparisons, "_isolation_min_chars", lambda n: 50)
    outcome: dict = {}
    out = _run(comparisons.extract_experiment_specific_paper_text(
        PAPER, "Study 2", client_choice="claude", cache_dir=str(tmp_path), outcome=outcome))
    assert outcome == {"method": "labels"}
    assert "Two hundred adults took part" in out and "t(98) = 2.5" not in out
    assert "<P1>" in calls[0][1]["content"]                          # numbered units sent
    # One label set serves every study: another target reuses the cached labels.
    out1 = _run(comparisons.extract_experiment_specific_paper_text(
        PAPER, "1", client_choice="claude", cache_dir=str(tmp_path)))
    assert len(calls) == 1
    assert "t(98) = 2.5" in out1
    assert list(tmp_path.glob("study_labels_*.json"))


def test_label_isolation_falls_back_to_generative(monkeypatch):
    monkeypatch.setattr(comparisons, "_isolation_llm_text", lambda messages, **kw: "not json at all")
    monkeypatch.setattr(comparisons, "_isolation_min_chars", lambda n: 10)

    async def generative(*a, **k):
        return "## Experiment 2\nGenerative isolation text for the study."

    monkeypatch.setattr(comparisons, "_generative_study_text", generative)
    outcome: dict = {}
    out = _run(comparisons.extract_experiment_specific_paper_text(
        PAPER, "2", client_choice="claude", outcome=outcome))
    assert out.startswith("## Experiment 2")
    assert outcome["method"] == "generative"
    assert "not valid JSON" in outcome["fallback_reason"]


def test_unknown_target_falls_back(monkeypatch):
    paras = sl.segment(PAPER)
    monkeypatch.setattr(comparisons, "_isolation_llm_text", lambda m, **k: json.dumps(_labels(paras)))

    async def generative(*a, **k):
        return "generative"

    monkeypatch.setattr(comparisons, "_generative_study_text", generative)
    outcome: dict = {}
    _run(comparisons.extract_experiment_specific_paper_text(
        PAPER, "Study 7", client_choice="claude", outcome=outcome))
    assert outcome["method"] == "generative"
    assert "did not match a labelled study" in outcome["fallback_reason"]


def test_isolation_method_env(monkeypatch):
    monkeypatch.delenv("ISOLATION_METHOD", raising=False)
    assert comparisons._isolation_method() == "labels"
    monkeypatch.setenv("ISOLATION_METHOD", "generative")
    assert comparisons._isolation_method() == "generative"
    monkeypatch.setenv("ISOLATION_METHOD", "bogus")
    assert comparisons._isolation_method() == "labels"


def test_target_doctrine_states_cross_ref_precedence():
    block = comparisons._target_study_doctrine("2")
    assert "[From <another study>, cited for: <aspect>]" in block
    assert "own text takes precedence" in block


def test_orchestrator_records_isolation_method(tmp_path, monkeypatch):
    import numpy as np
    import backend.services.embeddings as emb

    monkeypatch.setattr(
        emb, "openai_embed_segments",
        lambda s, model=None, **k: np.ones((max(1, len(list(s))), 8), dtype=np.float32),
    )
    paras = sl.segment(PAPER)
    monkeypatch.setattr(comparisons, "_isolation_llm_text", lambda m, **k: json.dumps(_labels(paras)))
    monkeypatch.setattr(comparisons, "_isolation_min_chars", lambda n: 50)
    (tmp_path / "p.txt").write_text("Preregistration of Experiment 2.")
    (tmp_path / "paper.txt").write_text(PAPER)
    seen = []

    def runner(p, pa, c, d, **kw):
        seen.append(pa)
        return comparisons.ComparisonResult(items=[comparisons.ComparisonItem(dimension=d)])

    res = _run(comparisons.general_preregistration_comparison(
        str(tmp_path / "p.txt"), ".txt", str(tmp_path / "paper.txt"), ".txt", "claude", "pymupdf",
        selected_dimensions=[{"dimension": "N", "definition": "n"}], comparison_runner=runner,
        multiple_experiments="yes", experiment_number="2", evidence_out={}))
    assert res.multi_study_isolation == "ok"
    assert res.multi_study_isolation_method == "labels"
    assert "Two hundred adults" in seen[0] and "t(98) = 2.5" not in seen[0]
