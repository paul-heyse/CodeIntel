"""Normalization helpers for core TypedDicts."""

from __future__ import annotations

import pytest

from codeintel.core.types import (
    normalize_pytest_entry,
    normalize_scip_document,
    validate_pytest_entry,
    validate_scip_document,
)


def test_normalize_scip_document_filters_invalid_occurrences() -> None:
    """Valid SCIP docs should survive normalization while invalid occurrences are dropped."""
    raw = {
        "relative_path": "pkg\\file.py",
        "occurrences": [
            {"symbol": "valid#def", "symbol_roles": "1"},
            {"symbol_roles": 2},
        ],
    }
    doc = normalize_scip_document(raw)
    if doc is None:
        pytest.fail("Normalized SCIP document should not be None")
    validate_scip_document(doc)
    if "relative_path" not in doc:
        pytest.fail("relative_path missing after normalization")
    if doc["relative_path"] != "pkg/file.py":
        pytest.fail("Path was not normalized to forward slashes")
    occurrences = doc.get("occurrences", [])
    if len(occurrences) != 1:
        pytest.fail("Invalid occurrence should have been filtered out")
    first_occurrence = occurrences[0]
    if "symbol_roles" not in first_occurrence:
        pytest.fail("symbol_roles missing after normalization")
    if first_occurrence["symbol_roles"] != 1:
        pytest.fail("Symbol roles should normalize to integer 1")


def test_normalize_pytest_entry_sanitizes_keywords_and_durations() -> None:
    """Pytest entries should normalize keywords and numeric durations."""
    expected_duration = 1.5
    expected_call_duration = 0.25
    raw = {
        "nodeid": "tests/test_sample.py::test_ok",
        "keywords": {"slow": True, "skip": False},
        "duration": str(expected_duration),
        "call": {"duration": str(expected_call_duration)},
    }
    entry = normalize_pytest_entry(raw)
    if entry is None:
        pytest.fail("Normalized pytest entry should not be None")
    validate_pytest_entry(entry)
    if "keywords" not in entry:
        pytest.fail("keywords missing after normalization")
    if entry["keywords"] != ["slow"]:
        pytest.fail("Keywords were not normalized to a sorted list")
    if "duration" not in entry:
        pytest.fail("duration missing after normalization")
    if entry["duration"] != expected_duration:
        pytest.fail("Duration was not coerced to float seconds")
    call = entry.get("call")
    if call is None:
        pytest.fail("Call entry should be present")
    call_duration = call.get("duration")
    if call_duration != expected_call_duration:
        pytest.fail("Call duration was not coerced to float seconds")
