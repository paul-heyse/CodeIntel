"""Test core types from codeintel.core.types.

This module tests:
- ScipRange, ScipOccurrence, ScipDocument TypedDicts
- _normalize_path, _normalize_occurrence, normalize_scip_document
- validate_scip_document
- PytestTestEntry, PytestCallEntry TypedDicts
- normalize_pytest_entry, validate_pytest_entry
"""

from __future__ import annotations

import pytest

from codeintel.core.types import (
    PytestTestEntry,
    ScipDocument,
    _normalize_occurrence,
    _normalize_path,
    normalize_pytest_entry,
    normalize_scip_document,
    validate_pytest_entry,
    validate_scip_document,
)

# =============================================================================
# _normalize_path Tests
# =============================================================================


def test_normalize_path_forward_slashes() -> None:
    """Verify forward slashes are preserved."""
    result = _normalize_path("path/to/file.py")
    assert result == "path/to/file.py"


def test_normalize_path_backslashes_converted() -> None:
    """Verify backslashes are converted to forward slashes."""
    result = _normalize_path("path\\to\\file.py")
    assert result == "path/to/file.py"


def test_normalize_path_mixed_slashes() -> None:
    """Verify mixed slashes are normalized."""
    result = _normalize_path("path\\to/file.py")
    assert result == "path/to/file.py"


def test_normalize_path_non_string_returns_none() -> None:
    """Verify non-string inputs return None."""
    assert _normalize_path(None) is None
    assert _normalize_path(123) is None
    assert _normalize_path([]) is None


def test_normalize_path_empty_string() -> None:
    """Verify empty string returns empty string."""
    result = _normalize_path("")
    assert result == ""


# =============================================================================
# _normalize_occurrence Tests
# =============================================================================


def test_normalize_occurrence_valid() -> None:
    """Verify valid occurrence is normalized."""
    raw = {
        "symbol": "test#symbol",
        "symbol_roles": 1,
        "range": {
            "start_line": 10,
            "start_character": 5,
            "end_line": 10,
            "end_character": 15,
        },
    }

    result = _normalize_occurrence(raw)

    assert result is not None
    assert result.get("symbol") == "test#symbol"
    assert result.get("symbol_roles") == 1
    result_range = result.get("range")
    assert result_range is not None
    assert result_range.get("start_line") == 10


def test_normalize_occurrence_missing_symbol() -> None:
    """Verify occurrence without symbol returns None."""
    raw = {"symbol_roles": 1}

    result = _normalize_occurrence(raw)

    assert result is None


def test_normalize_occurrence_empty_symbol() -> None:
    """Verify occurrence with empty symbol returns None."""
    raw = {"symbol": ""}

    result = _normalize_occurrence(raw)

    assert result is None


def test_normalize_occurrence_invalid_symbol_roles() -> None:
    """Verify invalid symbol_roles is handled."""
    raw = {"symbol": "test#sym", "symbol_roles": "invalid"}

    result = _normalize_occurrence(raw)

    assert result is not None
    assert result.get("symbol_roles") is None


def test_normalize_occurrence_non_mapping() -> None:
    """Verify non-mapping input returns None."""
    result = _normalize_occurrence("not a mapping")
    assert result is None

    result = _normalize_occurrence(None)
    assert result is None


def test_normalize_occurrence_without_range() -> None:
    """Verify occurrence without range is valid."""
    raw = {"symbol": "test#sym"}

    result = _normalize_occurrence(raw)

    assert result is not None
    assert result.get("symbol") == "test#sym"
    assert "range" not in result


# =============================================================================
# normalize_scip_document Tests
# =============================================================================


def test_normalize_scip_document_valid() -> None:
    """Verify valid document is normalized."""
    raw = {
        "relative_path": "src/file.py",
        "occurrences": [{"symbol": "test#sym", "symbol_roles": 1}],
    }

    result = normalize_scip_document(raw)

    assert result is not None
    assert result.get("relative_path") == "src/file.py"
    result_occurrences = result.get("occurrences")
    assert result_occurrences is not None
    assert len(result_occurrences) == 1


def test_normalize_scip_document_missing_path() -> None:
    """Verify document without path returns None."""
    raw = {"occurrences": [{"symbol": "test#sym"}]}

    result = normalize_scip_document(raw)

    assert result is None


def test_normalize_scip_document_empty_occurrences() -> None:
    """Verify document with empty occurrences returns None."""
    raw = {"relative_path": "file.py", "occurrences": []}

    result = normalize_scip_document(raw)

    assert result is None


def test_normalize_scip_document_filters_invalid_occurrences() -> None:
    """Verify invalid occurrences are filtered out."""
    raw = {
        "relative_path": "file.py",
        "occurrences": [
            {"symbol": "valid#sym"},
            {"no_symbol": "invalid"},
            {"symbol": "another#sym"},
        ],
    }

    result = normalize_scip_document(raw)

    assert result is not None
    result_occurrences = result.get("occurrences")
    assert result_occurrences is not None
    assert len(result_occurrences) == 2


def test_normalize_scip_document_normalizes_path() -> None:
    """Verify path is normalized."""
    raw = {
        "relative_path": "src\\file.py",
        "occurrences": [{"symbol": "test#sym"}],
    }

    result = normalize_scip_document(raw)

    assert result is not None
    assert result.get("relative_path") == "src/file.py"


# =============================================================================
# validate_scip_document Tests
# =============================================================================


def test_validate_scip_document_valid() -> None:
    """Verify valid document passes validation."""
    doc: ScipDocument = {
        "relative_path": "file.py",
        "occurrences": [{"symbol": "test#sym"}],
    }

    validate_scip_document(doc)  # Should not raise


def test_validate_scip_document_missing_path() -> None:
    """Verify document without path fails validation."""
    doc: ScipDocument = {"occurrences": [{"symbol": "test#sym"}]}  # type: ignore[typeddict-item]

    with pytest.raises(ValueError, match="missing relative_path"):
        validate_scip_document(doc)


def test_validate_scip_document_empty_path() -> None:
    """Verify document with empty path fails validation."""
    doc: ScipDocument = {"relative_path": "", "occurrences": []}

    with pytest.raises(ValueError, match="missing relative_path"):
        validate_scip_document(doc)


def test_validate_scip_document_missing_occurrence_symbol() -> None:
    """Verify occurrence without symbol fails validation."""
    doc: ScipDocument = {
        "relative_path": "file.py",
        "occurrences": [{"symbol_roles": 1}],  # type: ignore[typeddict-item]
    }

    with pytest.raises(ValueError, match="missing symbol"):
        validate_scip_document(doc)


# =============================================================================
# normalize_pytest_entry Tests
# =============================================================================


def test_normalize_pytest_entry_valid() -> None:
    """Verify valid entry is normalized."""
    raw = {
        "nodeid": "tests/test_sample.py::test_func",
        "outcome": "passed",
        "status": "passed",
        "keywords": {"slow": True, "fast": False},
        "duration": 1.5,
        "call": {"duration": 1.0},
    }

    result = normalize_pytest_entry(raw)

    assert result is not None
    assert result.get("nodeid") == "tests/test_sample.py::test_func"
    assert result.get("outcome") == "passed"
    assert result.get("keywords") == ["slow"]  # Only truthy values, sorted
    assert result.get("duration") == 1.5
    result_call = result.get("call")
    assert result_call is not None
    assert result_call.get("duration") == 1.0


def test_normalize_pytest_entry_missing_nodeid() -> None:
    """Verify entry without nodeid returns None."""
    raw = {"outcome": "passed"}

    result = normalize_pytest_entry(raw)

    assert result is None


def test_normalize_pytest_entry_empty_nodeid() -> None:
    """Verify entry with empty nodeid returns None."""
    raw = {"nodeid": "", "outcome": "passed"}

    result = normalize_pytest_entry(raw)

    assert result is None


def test_normalize_pytest_entry_keywords_as_list() -> None:
    """Verify keywords list is preserved."""
    raw = {
        "nodeid": "test.py::test",
        "keywords": ["alpha", "beta", "gamma"],
    }

    result = normalize_pytest_entry(raw)

    assert result is not None
    assert result.get("keywords") == ["alpha", "beta", "gamma"]


def test_normalize_pytest_entry_duration_as_string() -> None:
    """Verify string duration is converted."""
    raw = {"nodeid": "test.py::test", "duration": "1.5"}

    result = normalize_pytest_entry(raw)

    assert result is not None
    assert result.get("duration") == 1.5


def test_normalize_pytest_entry_invalid_duration() -> None:
    """Verify invalid duration results in None."""
    raw = {"nodeid": "test.py::test", "duration": "invalid"}

    result = normalize_pytest_entry(raw)

    assert result is not None
    assert result.get("duration") is None


def test_normalize_pytest_entry_defaults() -> None:
    """Verify defaults are applied."""
    raw = {"nodeid": "test.py::test"}

    result = normalize_pytest_entry(raw)

    assert result is not None
    assert result.get("outcome") == "unknown"
    assert result.get("status") == "unknown"
    assert result.get("keywords") == []


def test_normalize_pytest_entry_call_without_duration() -> None:
    """Verify call without duration is handled."""
    raw = {"nodeid": "test.py::test", "call": {}}

    result = normalize_pytest_entry(raw)

    assert result is not None
    assert result.get("call") is not None


# =============================================================================
# validate_pytest_entry Tests
# =============================================================================


def test_validate_pytest_entry_valid() -> None:
    """Verify valid entry passes validation."""
    entry: PytestTestEntry = {
        "nodeid": "test.py::test_func",
        "outcome": "passed",
    }

    validate_pytest_entry(entry)  # Should not raise


def test_validate_pytest_entry_missing_nodeid() -> None:
    """Verify entry without nodeid fails validation."""
    entry: PytestTestEntry = {"outcome": "passed"}  # type: ignore[typeddict-item]

    with pytest.raises(ValueError, match="missing nodeid"):
        validate_pytest_entry(entry)


def test_validate_pytest_entry_empty_nodeid() -> None:
    """Verify entry with empty nodeid fails validation."""
    entry: PytestTestEntry = {"nodeid": "", "outcome": "passed"}

    with pytest.raises(ValueError, match="missing nodeid"):
        validate_pytest_entry(entry)
