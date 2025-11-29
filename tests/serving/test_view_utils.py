"""Unit tests for docs view normalization helpers."""

from __future__ import annotations

import pytest

from codeintel.serving.mcp.view_utils import (
    normalize_entrypoints_row,
    normalize_entrypoints_rows,
)


def test_normalize_entrypoints_row_parses_json_string() -> None:
    """Stringified JSON should be coerced to a list."""
    row = {"entrypoints_json": '["foo", "bar"]'}
    normalize_entrypoints_row(row)
    if row["entrypoints_json"] != ["foo", "bar"]:
        pytest.fail("Expected JSON string to parse into list")


def test_normalize_entrypoints_rows_handles_none_and_scalar() -> None:
    """None becomes empty list and scalar becomes singleton list."""
    rows = [
        {"entrypoints_json": None},
        {"entrypoints_json": {"path": "/api"}},
    ]
    normalize_entrypoints_rows(rows)
    if rows[0]["entrypoints_json"] != []:
        pytest.fail("Expected None to normalize to empty list")
    if rows[1]["entrypoints_json"] != [{"path": "/api"}]:
        pytest.fail("Expected scalar to normalize into singleton list")
