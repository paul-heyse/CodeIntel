"""Unit tests for docs view normalization utilities."""

from __future__ import annotations

import pytest

from codeintel.serving.mcp.view_utils import (
    normalize_entrypoints_payload,
    normalize_entrypoints_row,
)


def test_normalize_entrypoints_payload_from_json_string() -> None:
    """JSON strings should parse into a list of dictionaries."""
    raw = '[{"path": "/a"}, {"path": "/b", "methods": ["GET", "POST"]}]'
    normalized = normalize_entrypoints_payload(raw)
    expected = [
        {"path": "/a"},
        {"path": "/b", "methods": ["GET", "POST"]},
    ]
    if normalized != expected:
        pytest.fail(f"Unexpected normalization: {normalized}")


def test_normalize_entrypoints_payload_from_mixed_dicts() -> None:
    """Mixed dict payloads should coerce keys/values to strings where needed."""
    raw = [{"path": "/a", "methods": ["GET"]}, {"index": 0, "active": True}, "skip"]
    normalized = normalize_entrypoints_payload(raw)
    expected = [
        {"path": "/a", "methods": ["GET"]},
        {"index": "0", "active": "True"},
    ]
    if normalized != expected:
        pytest.fail(f"Unexpected normalization: {normalized}")


def test_normalize_entrypoints_payload_from_single_dict() -> None:
    """Single dict payloads should wrap into a list."""
    normalized = normalize_entrypoints_payload({"path": "/api", "methods": ["POST"]})
    expected = [{"path": "/api", "methods": ["POST"]}]
    if normalized != expected:
        pytest.fail(f"Unexpected normalization: {normalized}")


def test_normalize_entrypoints_row_updates_mapping_in_place() -> None:
    """Row mutation should replace entrypoints_json with normalized list."""
    row = {"entrypoints_json": {"path": "/x", "methods": ["PUT", "DELETE"]}}
    normalize_entrypoints_row(row)
    expected = [{"path": "/x", "methods": ["PUT", "DELETE"]}]
    if row["entrypoints_json"] != expected:
        pytest.fail(f"Row not normalized: {row['entrypoints_json']}")


def test_normalize_entrypoints_row_handles_none() -> None:
    """None entrypoints should become an empty list."""
    row = {"entrypoints_json": None}
    normalize_entrypoints_row(row)
    if row["entrypoints_json"] != []:
        pytest.fail(f"Expected empty entrypoints list, got {row['entrypoints_json']}")
