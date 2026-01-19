from __future__ import annotations

import re
from typing import Any, Dict

import requests


def extract_nct_id(text: str) -> str:
    match = re.search(r"(NCT\d{8})", text, re.IGNORECASE)
    if not match:
        raise ValueError("Unable to parse NCT ID from input")
    return match.group(1).upper()


def fetch_trial_json(nct_id: str) -> Dict[str, Any]:
    url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def flatten_json(data: Dict[str, Any], parent_key: str = "", sep: str = " → ") -> Dict[str, str]:
    items: Dict[str, str] = {}
    for key, value in data.items():
        cleaned_key = key.replace("Module", "").replace("module", "")
        new_key = f"{parent_key}{sep}{cleaned_key}" if parent_key else cleaned_key

        if isinstance(value, dict):
            items.update(flatten_json(value, new_key, sep=sep))
        elif isinstance(value, list):
            if all(isinstance(item, dict) for item in value):
                for index, item in enumerate(value):
                    items.update(flatten_json(item, f"{new_key}[{index}]", sep=sep))
            else:
                items[new_key] = "\n".join(map(str, value))
        elif value not in [None, ""]:
            items[new_key] = str(value)
    return items


def nested_flatten_json(
    data: Dict[str, Any], parent_key: str = "", sep: str = " → "
) -> Dict[str, Dict[str, str]]:
    nested: Dict[str, Dict[str, str]] = {}

    def recurse(obj: Any, current_path: list[str]) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                cleaned_key = key.replace("Module", "").replace("module", "")
                new_path = current_path + [cleaned_key]
                recurse(value, new_path)
        elif isinstance(obj, list):
            for index, item in enumerate(obj):
                recurse(item, current_path + [f"[{index}]"])
        elif obj not in [None, ""]:
            if len(current_path) >= 2:
                dimension = current_path[0]
                subcomponent = sep.join(current_path[1:])
                nested.setdefault(dimension, {})[subcomponent] = str(obj)
            elif len(current_path) == 1:
                dimension = current_path[0]
                nested.setdefault(dimension, {})[""] = str(obj)

    recurse(data, [])
    return nested


def extract_flattened_trial(nct_id: str) -> Dict[str, str]:
    trial_json = fetch_trial_json(nct_id)
    protocol_section = trial_json.get("protocolSection", {})
    return flatten_json(protocol_section)


def extract_nested_trial(nct_id: str) -> Dict[str, Dict[str, str]]:
    trial_json = fetch_trial_json(nct_id)
    protocol_section = trial_json.get("protocolSection", {})
    return nested_flatten_json(protocol_section)
