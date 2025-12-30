"""Tests for table key helper utilities."""

from __future__ import annotations

import pytest

from codeintel.storage.helpers.table_key import (
    TableKeyValidationError,
    parse_table_key,
    try_parse_table_key,
)
from tests._helpers.assertions.expectation_assertions import expect_is_none


def test_try_parse_table_key_returns_none_for_unqualified() -> None:
    """try_parse_table_key should return None for unqualified keys."""
    parsed = try_parse_table_key("unqualified")
    expect_is_none(parsed)


def test_try_parse_table_key_returns_none_for_invalid() -> None:
    """try_parse_table_key should return None for invalid keys."""
    parsed = try_parse_table_key("bad-key.format")
    expect_is_none(parsed)


def test_parse_table_key_raises_for_unqualified() -> None:
    """parse_table_key should raise for unqualified keys."""
    with pytest.raises(TableKeyValidationError):
        parse_table_key("missing_schema")
