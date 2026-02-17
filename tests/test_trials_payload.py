import os

import pytest

from backend.services.trials import extract_nct_id, extract_nested_trial_with_metadata


def _build_prereg_text(nested_trial: dict[str, dict[str, str]]) -> str:
    return "\n\n".join(
        f"{dimension}\n\n" + "\n".join(f"{sub}\n{text}" for sub, text in subcomponents.items())
        for dimension, subcomponents in nested_trial.items()
    )


def test_print_clinical_prereg_payload():
    nct_raw = os.environ.get("REGCHECK_NCT_ID")
    if not nct_raw:
        pytest.skip("Set REGCHECK_NCT_ID to an NCT identifier to run this test.")

    nct_id = extract_nct_id(nct_raw)
    nested_trial, metadata = extract_nested_trial_with_metadata(nct_id)
    prereg_text = _build_prereg_text(nested_trial)

    print("=== regcheck clinical prereg payload ===")
    print("nct_id:", nct_id)
    print("metadata:", metadata)
    print("payload:")
    print(prereg_text)

    assert prereg_text.strip()
