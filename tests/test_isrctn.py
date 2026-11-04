from datetime import date

import pytest

from backend.services import isrctn

NS = "http://www.67bricks.com/isrctn"


def _record(
    *,
    hypothesis="",
    primary="",
    primary_structured="",
    secondary="",
    inclusion="",
    exclusion="",
    intervention="Drug A versus placebo",
    pes="",
    target="300",
    registered="2020-01-10",
    start="2020-03-01",
    last_updated="2023-06-01",
) -> bytes:
    def el(tag, text):
        return f"<{tag}>{text}</{tag}>" if text else f"<{tag}/>"

    return f"""<allTrials xmlns="{NS}"><fullTrial>
<trial lastUpdated="{last_updated}T10:00:00Z" version="7" publicIdentifierCanonical="ISRCTN12345678">
  <isrctn dateAssigned="{registered}T09:00:00Z">12345678</isrctn>
  <trialDescription>
    <title>A trial</title>
    <scientificTitle>A randomised trial of drug A</scientificTitle>
    {el("studyHypothesis", hypothesis)}
    {el("plainEnglishSummary", pes)}
    <primaryOutcomes>{primary_structured}</primaryOutcomes>
    {el("primaryOutcome", primary)}
    <secondaryOutcomes/>
    {el("secondaryOutcome", secondary)}
  </trialDescription>
  <trialDesign><studyDesign>Randomised controlled trial</studyDesign>
    <primaryStudyDesign>Interventional</primaryStudyDesign></trialDesign>
  <participants>
    {el("inclusion", inclusion)}
    {el("exclusion", exclusion)}
    <targetEnrolment>{target}</targetEnrolment>
    <recruitmentStart>{start}T00:00:00.000Z</recruitmentStart>
    <recruitmentEnd>2022-01-01T00:00:00.000Z</recruitmentEnd>
  </participants>
  <conditions><condition><description>Depression</description></condition></conditions>
  <interventions><intervention><description>{intervention}</description>
    <interventionType>Drug</interventionType></intervention></interventions>
</trial></fullTrial></allTrials>""".encode()


def _run(**kwargs):
    xml = _record(**kwargs)
    nested, meta = isrctn.extract_nested_trial_with_metadata("ISRCTN12345678", xml_fetcher=lambda _: xml)
    fields = {f["field"]: f for f in meta["fields"]}
    return nested, meta, fields


CHAIN = """Current primary outcomes as of 03/11/2022:
Depression (PHQ-9) at 6, 12 and 24 weeks

Previous primary outcomes as of 13/03/2020:
Depression (PHQ-9) at 6 and 12 weeks

Previous primary outcomes:
Depression (BDI-II) at 12 weeks"""


def test_find_isrctn_id_variants():
    assert isrctn.find_isrctn_id("ISRCTN17573805") == "ISRCTN17573805"
    assert isrctn.find_isrctn_id("https://www.isrctn.com/ISRCTN17573805") == "ISRCTN17573805"
    assert isrctn.find_isrctn_id("doi: 10.1186/isrctn17573805") == "ISRCTN17573805"
    assert isrctn.find_isrctn_id("ISRCTN 17573805") == "ISRCTN17573805"
    assert isrctn.find_isrctn_id("NCT01234567") is None
    assert isrctn.find_isrctn_id("ISRCTN1757380") is None  # 7 digits
    with pytest.raises(ValueError):
        isrctn.extract_isrctn_id("not an id")


def test_split_versions_reads_the_chain_newest_first():
    blocks, regular = isrctn.split_versions(CHAIN)
    assert regular
    assert [b.kind for b in blocks] == ["current", "older", "older"]
    assert [b.date for b in blocks] == [date(2022, 11, 3), date(2020, 3, 13), None]
    assert blocks[2].text == "Depression (BDI-II) at 12 weeks"


def test_content_lines_are_not_headings():
    text = (
        "1. Previous abdominal surgery: excluded\n"
        "2. The previous sponsor for this trial (from 01/09/2005 to 31/01/2008) was:\n"
        "MRC"
    )
    blocks, regular = isrctn.split_versions(text)
    assert regular and len(blocks) == 1 and blocks[0].kind == "unlabeled"


def test_middle_version_in_force_at_start_is_confirmed():
    # Start 2021-01-01 falls between the 13/03/2020 and 03/11/2022 changes.
    nested, meta, fields = _run(primary=CHAIN, start="2021-01-01")
    assert nested["Outcomes"]["Primary outcome"] == "Depression (PHQ-9) at 6 and 12 weeks"
    assert fields["Primary outcome"]["tier"] == "confirmed"
    assert meta["prospective"] is True and meta["source"] == "reconstructed"


def test_original_version_in_force_before_first_change():
    nested, _meta, fields = _run(primary=CHAIN, start="2020-02-01")
    assert nested["Outcomes"]["Primary outcome"] == "Depression (BDI-II) at 12 weeks"
    assert fields["Primary outcome"]["tier"] == "confirmed"


def test_record_not_edited_after_start_is_confirmed_current():
    nested, meta, fields = _run(primary=CHAIN, start="2023-01-01", last_updated="2022-12-01")
    assert nested["Outcomes"]["Primary outcome"].startswith("Depression (PHQ-9) at 6, 12 and 24")
    assert {f["tier"] for f in fields.values()} == {"confirmed"}
    assert meta["source"] == "latest"


def test_unmarked_fields_are_probable_and_scalars_unverified():
    _nested, _meta, fields = _run(hypothesis="Drug A reduces depression", start="2020-03-01")
    assert fields["Study hypothesis"]["tier"] == "probable"
    assert fields["Target sample size"]["tier"] == "unverified"


def test_until_dates_bound_the_previous_version():
    text = (
        "Current exclusion criteria as of 12/02/2013:\nAged under 18\n\n"
        "Previous exclusion criteria as of 10/03/2010 and until 12/02/2013:\nAged under 16"
    )
    nested, _meta, fields = _run(exclusion=text, registered="2009-12-04", start="2010-10-20", last_updated="2019-05-29")
    assert nested["Participants"]["Exclusion criteria"] == "Aged under 16"
    assert fields["Exclusion criteria"]["tier"] == "confirmed"


def test_earliest_kept_version_after_start_is_unrecoverable():
    text = (
        "Current inclusion criteria as of 25/06/2025:\nAge 50 or over\n\n"
        "Previous inclusion criteria from 19/06/2014 to 25/06/2025:\nAdults at high risk"
    )
    _nested, _meta, fields = _run(inclusion=text, registered="2009-01-23", start="2009-03-01", last_updated="2026-01-29")
    assert fields["Inclusion criteria"]["tier"] == "unrecoverable"


def test_structured_outcomes_with_previous_block_try_both_readings():
    structured = (
        "<outcomeMeasure><variable>Depressive symptoms</variable><method>PHQ-9</method>"
        "<timepoints>3 and 6 months</timepoints></outcomeMeasure>"
    )
    legacy = "Previous primary outcome as of 27/04/2021:\nDepressive symptoms and loneliness at 3 and 6 months"
    # Start before the dated change. Read as "replaced on 27/04", the previous
    # text was in force; read as "began on 27/04", nothing kept covers the start
    # (the registration-time text would be missing), so that reading is set aside.
    nested, _meta, fields = _run(
        primary=legacy, primary_structured=structured, registered="2021-01-05", start="2021-02-01",
        last_updated="2021-05-01",
    )
    assert nested["Outcomes"]["Primary outcome"].startswith("Depressive symptoms and loneliness")
    assert fields["Primary outcome"]["tier"] == "confirmed"
    # Start after it: the readings disagree (the structured list may have
    # replaced it on 27/04 or later) -> unverified.
    _nested, _meta, fields = _run(
        primary=legacy, primary_structured=structured, registered="2021-01-05", start="2021-04-30",
        last_updated="2021-05-01",
    )
    assert fields["Primary outcome"]["tier"] == "unverified"


def test_added_text_after_start_is_removed():
    text = "1. Inability to consent\n2. Urgent delivery\nAdded 27/01/2021:\n3. Retired healthcare professionals"
    nested, _meta, fields = _run(exclusion=text, start="2020-03-01")
    assert "Retired" not in nested["Participants"]["Exclusion criteria"]
    assert fields["Exclusion criteria"]["tier"] == "probable"
    assert fields["Exclusion criteria"]["removed_additions"] == ["2021-01-27"]
    # Added before the start: kept.
    nested, _meta, _fields = _run(exclusion=text.replace("27/01/2021", "27/01/2020"), start="2020-03-01")
    assert "Retired" in nested["Participants"]["Exclusion criteria"]


def test_update_note_after_start_is_removed_but_unverified():
    text = "Couples are randomised 1:1.\n\nUpdated 13/03/2021:\nThe intervention runs throughout pregnancy."
    nested, _meta, fields = _run(intervention=text, start="2020-03-01")
    assert nested["Interventions"]["Intervention"] == "Couples are randomised 1:1."
    assert fields["Intervention"]["tier"] == "unverified"


def test_inline_update_after_start_is_reverted():
    pes = (
        "When is the study starting and how long is it expected to run for?\n"
        "July 2019 to April 2021 (updated 08/07/2020, previously: June 2021)\n\nWho is funding the study?\nNIHR"
    )
    nested, _meta, fields = _run(pes=pes, start="2020-03-01")
    text = nested["Description"]["Plain English summary"]
    assert "July 2019 to April 2021" not in text and "June 2021" in text
    assert "previously" not in text
    # Update before the start: the new value stands, note dropped.
    nested, _meta, _fields = _run(pes=pes.replace("08/07/2020", "08/01/2020"), start="2020-03-01")
    assert "July 2019 to April 2021\n" in nested["Description"]["Plain English summary"] + "\n"


def test_retrospective_registration_uses_text_as_first_registered():
    nested, meta, fields = _run(primary=CHAIN, registered="2020-01-10", start="2019-06-01")
    assert meta["prospective"] is False
    assert nested["Outcomes"]["Primary outcome"] == "Depression (BDI-II) at 12 weeks"
    assert fields["Primary outcome"]["tier"] == "retrospective"
    assert meta["summary"]["retrospective"] == len(meta["fields"])


def test_irregular_layout_is_unrecoverable_with_full_text():
    text = (
        "Current inclusion criteria as of 12/02/2013:\nA\n\n"
        "Previous inclusion criteria as of 23/03/2010:\nB\n\n"
        "Current information as of 10/03/2010:\nC"
    )
    nested, _meta, fields = _run(inclusion=text, start="2010-10-20", registered="2009-12-04")
    assert fields["Inclusion criteria"]["tier"] == "unrecoverable"
    assert "Current information as of 10/03/2010" in nested["Participants"]["Inclusion criteria"]


def test_nested_shape_and_metadata():
    nested, meta, _fields = _run(hypothesis="H1", primary=CHAIN, start="2021-01-01")
    assert nested["Identification"]["ISRCTN"] == "ISRCTN12345678"
    assert nested["Participants"]["Recruitment start"] == "2021-01-01"
    assert nested["Interventions"]["Intervention type"] == "Drug"
    assert meta["registry"] == "isrctn"
    assert meta["start_date"] == "2021-01-01" and meta["registration_date"] == "2020-01-10"
    assert sum(meta["summary"].values()) == len(meta["fields"])
    # Untracked descriptive fields (title, design category, ...) aren't graded.
    assert "Title" not in {f["field"] for f in meta["fields"]}


def test_fetch_rejects_non_xml_reply(monkeypatch):
    class Resp:
        status_code = 500
        headers = {"content-type": "text/html"}
        content = b"<html>"

    monkeypatch.setattr(isrctn.requests, "get", lambda *a, **k: Resp())
    with pytest.raises(ValueError, match="no public record"):
        isrctn.fetch_isrctn_xml("ISRCTN00000001")


@pytest.mark.parametrize(
    "value",
    ["NCT01234567", "https://clinicaltrials.gov/study/NCT01234567", "ISRCTN17573805",
     "https://www.isrctn.com/ISRCTN17573805"],
)
def test_route_accepts_either_registry_id(value):
    from backend.routes.comparisons import _validate_trial_registration_id

    _validate_trial_registration_id(value)


@pytest.mark.parametrize("value", ["", "   ", "12345678", "DRKS00012345"])
def test_route_rejects_other_ids(value):
    from fastapi import HTTPException

    from backend.routes.comparisons import _validate_trial_registration_id

    with pytest.raises(HTTPException) as exc_info:
        _validate_trial_registration_id(value)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_clinical_comparison_reads_isrctn_reconstruction(tmp_path, monkeypatch):
    from backend.services import comparisons
    from backend.services.comparisons import ComparisonResult

    xml = _record(primary=CHAIN, start="2021-01-01")
    real = isrctn.extract_nested_trial_with_metadata
    monkeypatch.setattr(
        comparisons.isrctn, "extract_nested_trial_with_metadata",
        lambda i: real(i, xml_fetcher=lambda _: xml),
    )

    def nct_must_not_run(_text):
        raise AssertionError("an ISRCTN id must not go through the CT.gov path")

    seen = {}

    def fake_run(prereg, paper, client, dimension, **kwargs):
        seen["prereg"] = prereg
        return ComparisonResult(items=[])

    paper = tmp_path / "paper.txt"
    paper.write_text("Paper text.")
    await comparisons.clinical_trial_comparison(
        "https://www.isrctn.com/ISRCTN12345678", str(paper), ".txt", "openai", parser_choice="pymupdf",
        selected_dimensions=[{"dimension": "Primary outcome", "definition": "d"}],
        nct_extractor=nct_must_not_run, comparison_runner=fake_run,
    )
    # The version in force at recruitment start, not the current text.
    assert "Depression (PHQ-9) at 6 and 12 weeks" in seen["prereg"]
    assert "24 weeks" not in seen["prereg"]


def test_recovery_disclosure_keeps_only_viewer_fields():
    from backend.services.comparisons import _recovery_disclosure

    _nested, meta, _fields = _run(primary=CHAIN, start="2021-01-01")
    out = _recovery_disclosure(meta)
    assert out["registry"] == "isrctn" and out["start_date"] == "2021-01-01"
    assert set(out["fields"][0]) == {"field", "tier", "reason"}
