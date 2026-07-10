"""Contract tests for the Registered Report carried-forward-text engine (Track A).

The engine is deliberately deterministic and lexical: block alignment via exact +
fuzzy matching, verbatim-move detection, and word-level diffs. No API calls."""

from backend.services.rr_alignment import align_stages

S1 = """Introduction. Cognitive flexibility is central to adaptive behaviour.

Hypotheses. We predict that response times will be slower on switch trials than on repeat trials.

Method. Participants will complete 240 trials in a colour-shape switching paradigm.

Participants. We will recruit 200 adults via Prolific. Data collection will take two weeks.

Analysis plan. We will fit a linear mixed-effects model predicting log response time."""


def test_identical_stages_produce_no_changes():
    out = align_stages(S1, S1)
    assert out["changes"] == []
    assert out["stats"]["identical"] == 5
    assert out["stats"]["modified"] == out["stats"]["deleted"] == out["stats"]["inserted"] == 0


def test_reflow_robustness_hard_wrapped_text_is_not_a_change():
    # The same content arriving with PDF-style hard line wraps must align as identical.
    wrapped = S1.replace(
        "Cognitive flexibility is central to adaptive behaviour.",
        "Cognitive flexibility is central\nto adaptive behaviour.",
    )
    out = align_stages(S1, wrapped)
    assert out["changes"] == []


def test_tense_conversion_is_a_modified_block_with_word_diff():
    s2 = S1.replace(
        "Participants will complete 240 trials",
        "Participants completed 240 trials",
    )
    out = align_stages(S1, s2)
    assert out["stats"]["modified"] == 1
    change = out["changes"][0]
    assert change["kind"] == "modified"
    assert "will complete" in change["removed_text"]
    assert "completed" in change["added_text"]
    # The unchanged part of the block is not reported as removed/added.
    assert "colour-shape" not in change["removed_text"]


def test_appended_results_section_reports_inserted_blocks():
    s2 = S1 + "\n\nResults. The registered model showed the predicted switch cost (b = 0.083, p < .001).\n\nDiscussion. These findings confirm the preregistered prediction."
    out = align_stages(S1, s2)
    kinds = [c["kind"] for c in out["changes"]]
    assert kinds.count("inserted") == 2
    assert out["stats"]["identical"] == 5


def test_deleted_caveat_reports_deleted_block():
    s2 = S1.replace(
        "\n\nParticipants. We will recruit 200 adults via Prolific. Data collection will take two weeks.",
        "",
    )
    out = align_stages(S1, s2)
    assert out["stats"]["deleted"] == 1
    assert out["changes"][0]["kind"] == "deleted"
    assert "200 adults" in out["changes"][0]["stage1_text"]


def test_verbatim_move_is_not_reported_as_a_change():
    # Move the Analysis plan block before Method, verbatim: fixity is intact.
    blocks = S1.split("\n\n")
    reordered = "\n\n".join([blocks[0], blocks[1], blocks[4], blocks[2], blocks[3]])
    out = align_stages(S1, reordered)
    assert out["changes"] == []
    assert out["stats"]["moved"] >= 1


def test_moderate_rewrite_pairs_as_modified_not_delete_insert():
    s2 = S1.replace(
        "We predict that response times will be slower on switch trials than on repeat trials.",
        "We tentatively explored whether response times might differ between switch and repeat trials.",
    )
    out = align_stages(S1, s2)
    assert out["stats"]["modified"] == 1
    assert out["stats"]["deleted"] == 0 and out["stats"]["inserted"] == 0
    change = out["changes"][0]
    assert change["kind"] == "modified"
    assert 0 < change["similarity"] < 1


def test_wholesale_rewrite_falls_back_to_delete_plus_insert():
    s2 = S1.replace(
        "Hypotheses. We predict that response times will be slower on switch trials than on repeat trials.",
        "Open questions. Prior evidence is mixed and no directional expectation can be justified a priori.",
    )
    out = align_stages(S1, s2)
    kinds = sorted(c["kind"] for c in out["changes"])
    assert kinds == ["deleted", "inserted"]


def test_stats_summarise_carried_forward_fidelity():
    s2 = S1.replace("will fit", "fitted") + "\n\nResults. b = 0.083."
    out = align_stages(S1, s2)
    st = out["stats"]
    assert st["stage1_blocks"] == 5 and st["stage2_blocks"] == 6
    assert st["identical"] == 4 and st["modified"] == 1 and st["inserted"] == 1
    assert 0 < st["carried_forward_identical_pct"] < 100


def test_changes_carry_pending_classification_fields():
    s2 = S1.replace("will fit", "fitted")
    out = align_stages(S1, s2)
    change = out["changes"][0]
    assert change["classification"] == "pending"
    assert change["category"] == ""


def test_classifier_merges_labels_and_survives_garbage(monkeypatch):
    import backend.services.rr_alignment as rra

    changes = [
        {"kind": "modified", "removed_text": "will fit", "added_text": "fitted",
         "classification": "pending", "category": "", "note": ""},
        {"kind": "inserted", "removed_text": "", "added_text": "Results. b = 0.083.",
         "classification": "pending", "category": "", "note": ""},
        {"kind": "modified", "removed_text": "we predict", "added_text": "we explored",
         "classification": "pending", "category": "", "note": ""},
    ]

    reply = """Here you go:
    [{"id": 1, "classification": "licensed", "category": "tense", "note": "Future to past."},
     {"id": 2, "classification": "licensed", "category": "results_discussion", "note": "Appended results."},
     {"id": 3, "classification": "substantive", "category": "emphasis_hedging", "note": "Prediction demoted."},
     {"id": 99, "classification": "licensed", "category": "x", "note": "out of range"},
     {"id": 2, "classification": "banana", "category": "x", "note": "invalid label ignored"}]"""
    monkeypatch.setattr(rra, "_classifier_completion", lambda prompt, **_kw: reply)
    out = rra.classify_changes(changes)
    assert [c["classification"] for c in out] == ["licensed", "licensed", "substantive"]
    assert out[0]["category"] == "tense"
    assert out[2]["note"] == "Prediction demoted."


def test_classifier_failure_leaves_changes_pending(monkeypatch):
    import backend.services.rr_alignment as rra

    changes = [{"kind": "modified", "removed_text": "a", "added_text": "b",
                "classification": "pending", "category": "", "note": ""}]

    def _boom(prompt, **_kw):
        raise RuntimeError("provider down")

    monkeypatch.setattr(rra, "_classifier_completion", _boom)
    out = rra.classify_changes(changes)
    assert out[0]["classification"] == "pending"
