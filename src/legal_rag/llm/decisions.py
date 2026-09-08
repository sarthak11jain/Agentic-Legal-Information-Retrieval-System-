"""Parsing and safety policy for LLM citation-filter decisions."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping


_DECISION_PATTERN = re.compile(
    r'"label"\s*:\s*"(?P<label>[A-D])".*?"decision"\s*:\s*"(?P<decision>KEEP|REMOVE)"',
    flags=re.S,
)


def parse_decisions_fallback(raw_response: str) -> dict[str, str]:
    """Extract valid labelled decisions from malformed JSON-like output."""

    return {
        match.group("label"): match.group("decision")
        for match in _DECISION_PATTERN.finditer(raw_response)
    }


def parse_decisions(raw_response: str) -> tuple[dict[str, str], str]:
    """Parse model output and return decisions plus a parse diagnostic."""

    text = raw_response.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\[.*\]", text, flags=re.S)
    payload = match.group(0) if match else text
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as error:
        fallback = parse_decisions_fallback(text)
        if fallback:
            return fallback, f"json_decode_error_regex_fallback: {error}"
        return {}, f"json_decode_error: {error}"

    if not isinstance(parsed, list):
        fallback = parse_decisions_fallback(text)
        if fallback:
            return fallback, "json_root_not_list_regex_fallback"
        return {}, "json_root_not_list"

    decisions: dict[str, str] = {}
    for item in parsed:
        if not isinstance(item, Mapping):
            continue
        label = str(item.get("label", "")).strip().upper()
        decision = str(item.get("decision", "")).strip().upper()
        if label and decision in {"KEEP", "REMOVE"}:
            decisions[label] = decision
    return decisions, ""


def confirm_removals(
    first_pass: Mapping[str, str],
    confirmation: Mapping[str, str],
) -> dict[str, str]:
    """Apply the conservative policy: remove only after two REMOVE votes."""

    return {
        label: "REMOVE" if decision == "REMOVE" and confirmation.get(label) == "REMOVE" else "KEEP"
        for label, decision in first_pass.items()
    }
