"""Unit tests for SQL builder validation."""

from __future__ import annotations

import pytest

from codeintel.storage.sql import quote_identifier, quote_table_key, safe_macro_call


def test_quote_identifier_validates_simple_names() -> None:
    """Identifiers that match the regex should be quoted."""
    if quote_identifier("foo") != '"foo"':
        pytest.fail("Expected quoted identifier for foo")
    if quote_identifier("Foo_1") != '"Foo_1"':
        pytest.fail("Expected quoted identifier for Foo_1")


@pytest.mark.parametrize("value", ["", "1foo", "foo-bar", "foo;drop", "foo bar"])
def test_quote_identifier_rejects_invalid(value: str) -> None:
    """Unsafe identifiers should raise."""
    with pytest.raises(ValueError, match="Unsafe identifier"):
        quote_identifier(value)


def test_quote_table_key_validates_schema_and_table() -> None:
    """Table keys must include schema and be quoted."""
    if quote_table_key("schema.table") != '"schema"."table"':
        pytest.fail("Expected quoted table key")


@pytest.mark.parametrize("value", ["", "notable", "foo..bar"])
def test_quote_table_key_rejects_bad_keys(value: str) -> None:
    """Bad table keys should raise."""
    with pytest.raises(ValueError, match="Table key must include schema"):
        quote_table_key(value)


def test_safe_macro_call_validates_allowlist() -> None:
    """safe_macro_call should validate allowlist and preserve args."""
    sql, args = safe_macro_call("metadata.dataset_rows", [1, 2], allowed={"metadata.dataset_rows"})
    if "metadata.dataset_rows" not in sql:
        pytest.fail("Macro name should appear in SQL")
    if args != [1, 2]:
        pytest.fail("Args should be preserved")
    with pytest.raises(ValueError, match="not allowlisted"):
        safe_macro_call("metadata.bad_macro", [], allowed={"metadata.dataset_rows"})
