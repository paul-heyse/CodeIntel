"""Tests for ingest macro registration helpers."""

from __future__ import annotations

import duckdb

from codeintel.storage.macros import (
    assert_ingest_macros_present,
    clear_macro_cache_for_connection,
    ensure_ingest_macros,
    list_ingest_macros,
)
from codeintel.storage.metadata import INGEST_MACROS


def test_ensure_ingest_macros_registers_all_macros() -> None:
    """Verify ensure_ingest_macros registers all required macros."""
    con = duckdb.connect(":memory:")
    try:
        ensure_ingest_macros(con)

        macros = list_ingest_macros(con)
        macro_set = {macro.lower() for macro in INGEST_MACROS.values()}

        # All required macros should be registered
        assert macro_set.issubset(macros)
    finally:
        con.close()


def test_ensure_ingest_macros_is_idempotent() -> None:
    """Verify ensure_ingest_macros can be called multiple times safely."""
    con = duckdb.connect(":memory:")
    try:
        # First call
        ensure_ingest_macros(con)
        macros_first = list_ingest_macros(con)

        # Second call (should use cache)
        ensure_ingest_macros(con)
        macros_second = list_ingest_macros(con)

        # Results should be the same
        assert macros_first == macros_second
    finally:
        con.close()


def test_list_ingest_macros_returns_set() -> None:
    """Verify list_ingest_macros returns a set of macro names."""
    con = duckdb.connect(":memory:")
    try:
        ensure_ingest_macros(con)
        macros = list_ingest_macros(con)

        assert isinstance(macros, set)
        assert len(macros) > 0
    finally:
        con.close()


def test_clear_macro_cache_for_connection_with_connection() -> None:
    """Verify clear_macro_cache_for_connection clears cache for a connection."""
    con = duckdb.connect(":memory:")
    try:
        ensure_ingest_macros(con)

        # Clear cache
        clear_macro_cache_for_connection(con)

        # Should still work after cache clear
        ensure_ingest_macros(con)
        macros = list_ingest_macros(con)
        macro_set = {macro.lower() for macro in INGEST_MACROS.values()}
        assert macro_set.issubset(macros)
    finally:
        con.close()


def test_clear_macro_cache_for_connection_with_int_key() -> None:
    """Verify clear_macro_cache_for_connection accepts integer key."""
    con = duckdb.connect(":memory:")
    try:
        ensure_ingest_macros(con)

        # Clear cache using id
        cache_key = id(con)
        clear_macro_cache_for_connection(cache_key)

        # Should still work
        macros = list_ingest_macros(con)
        assert len(macros) > 0
    finally:
        con.close()


def test_assert_ingest_macros_present_succeeds_when_macros_exist() -> None:
    """Verify assert_ingest_macros_present passes when all macros exist."""
    con = duckdb.connect(":memory:")
    try:
        ensure_ingest_macros(con)

        # Should not raise
        assert_ingest_macros_present(con)
    finally:
        con.close()


def test_assert_ingest_macros_present_registers_missing_macros() -> None:
    """Verify assert_ingest_macros_present registers macros if missing."""
    con = duckdb.connect(":memory:")
    try:
        # Create metadata schema without full macro registration
        con.execute("CREATE SCHEMA IF NOT EXISTS metadata")

        # assert_ingest_macros_present should register the missing macros
        assert_ingest_macros_present(con)

        # Verify macros are now present
        macros = list_ingest_macros(con)
        macro_set = {macro.lower() for macro in INGEST_MACROS.values()}
        assert macro_set.issubset(macros)
    finally:
        con.close()


def test_registered_macros_handles_prefixed_function_names() -> None:
    """Verify _registered_macros parses function names with catalog/schema prefixes."""
    con = duckdb.connect(":memory:")
    try:
        # Create a macro with a prefixed name
        con.execute("CREATE SCHEMA IF NOT EXISTS test_schema")
        con.execute(
            """
            CREATE OR REPLACE MACRO test_schema.prefixed_macro(x) AS x + 1
            """
        )

        macros = list_ingest_macros(con)

        # Should have both qualified and unqualified versions
        assert "test_schema.prefixed_macro" in macros
        assert "prefixed_macro" in macros
    finally:
        con.close()


def test_macros_contain_ingest_prefix() -> None:
    """Verify all ingest macros follow the metadata.ingest_ naming convention."""
    for table_key, macro_name in INGEST_MACROS.items():
        assert macro_name.startswith("metadata.ingest_"), (
            f"Macro {macro_name} for {table_key} doesn't follow naming convention"
        )
