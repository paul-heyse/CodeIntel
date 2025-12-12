"""Snapshot helpers for Hamilton Phase 2 tests.

Provides JSON normalization and comparison utilities for testing CLI outputs
and plan results. Dynamic fields like timestamps and durations are removed
to enable deterministic snapshot comparisons.
"""

from __future__ import annotations

import json

# Fields that vary between runs and should be removed for comparison
DYNAMIC_KEYS: frozenset[str] = frozenset({
    "run_id",
    "duration_ms",
    "duration_seconds",
    "started_at",
    "completed_at",
    "recorded_at",
    "computed_at",
    "total_duration_ms",
})


def normalize(obj: object) -> object:
    """Remove dynamic fields from an object for snapshot comparison.

    Recursively processes dicts and lists, removing any keys that are
    in DYNAMIC_KEYS to enable deterministic comparisons.

    Parameters
    ----------
    obj
        Object to normalize (dict, list, or scalar).

    Returns
    -------
    object
        Normalized object with dynamic fields removed.

    Examples
    --------
    >>> normalize({"target": "modules", "duration_ms": 123.4})
    {'target': 'modules'}
    >>> normalize([{"run_id": "abc", "status": "ok"}])
    [{'status': 'ok'}]
    """
    if isinstance(obj, dict):
        return {k: normalize(v) for k, v in obj.items() if k not in DYNAMIC_KEYS}
    if isinstance(obj, list):
        return [normalize(x) for x in obj]
    return obj


def normalize_json(json_text: str) -> dict[str, object]:
    """Parse and normalize JSON text.

    Parameters
    ----------
    json_text
        JSON string to parse and normalize.

    Returns
    -------
    dict[str, object]
        Normalized dictionary.
    """
    data = json.loads(json_text)
    result = normalize(data)
    if not isinstance(result, dict):
        return {}
    return result


def assert_json_snapshot(actual_text: str, expected: dict[str, object]) -> None:
    """Assert JSON text matches expected dict after normalization.

    Parameters
    ----------
    actual_text
        JSON string from CLI or function output.
    expected
        Expected dictionary (should already be normalized).

    Raises
    ------
    AssertionError
        If normalized actual doesn't match expected.
    """
    actual = normalize_json(actual_text)
    if actual != expected:
        msg = f"JSON snapshot mismatch:\nActual: {actual}\nExpected: {expected}"
        raise AssertionError(msg)


def assert_json_contains(actual_text: str, expected_subset: dict[str, object]) -> None:
    """Assert JSON text contains expected fields after normalization.

    Parameters
    ----------
    actual_text
        JSON string from CLI or function output.
    expected_subset
        Expected fields that must be present.

    Raises
    ------
    AssertionError
        If normalized actual doesn't contain all expected fields.
    """
    actual = normalize_json(actual_text)
    for key, value in expected_subset.items():
        if key not in actual:
            msg = f"Missing key '{key}' in JSON output"
            raise AssertionError(msg)
        if actual[key] != value:
            msg = f"Value mismatch for '{key}': got {actual[key]}, expected {value}"
            raise AssertionError(msg)


__all__ = [
    "DYNAMIC_KEYS",
    "assert_json_contains",
    "assert_json_snapshot",
    "normalize",
    "normalize_json",
]
