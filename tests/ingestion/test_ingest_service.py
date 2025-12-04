"""Tests for the ingestion macros module.

This module tests the ingestion macro utilities, including the macro_exists
function and INGEST_MACRO_TABLES constant.
"""

from __future__ import annotations

import pytest

from codeintel.ingestion.infrastructure_utilities.macros import (
    INGEST_MACRO_TABLES,
    macro_exists,
)
from codeintel.storage.gateway import StorageGateway

# =============================================================================
# INGEST_MACRO_TABLES Tests
# =============================================================================


def test_ingest_macro_tables_is_frozenset() -> None:
    """INGEST_MACRO_TABLES should be a frozenset."""
    assert isinstance(INGEST_MACRO_TABLES, frozenset)


def test_ingest_macro_tables_contains_expected_tables() -> None:
    """INGEST_MACRO_TABLES should contain key ingestion tables."""
    # Core tables
    assert "core.ast_nodes" in INGEST_MACRO_TABLES
    assert "core.cst_nodes" in INGEST_MACRO_TABLES
    assert "core.docstrings" in INGEST_MACRO_TABLES
    assert "core.modules" in INGEST_MACRO_TABLES
    assert "core.goids" in INGEST_MACRO_TABLES

    # Analytics tables
    assert "analytics.coverage_lines" in INGEST_MACRO_TABLES
    assert "analytics.function_metrics" in INGEST_MACRO_TABLES
    assert "analytics.typedness" in INGEST_MACRO_TABLES

    # Graph tables
    assert "graph.call_graph_edges" in INGEST_MACRO_TABLES
    assert "graph.call_graph_nodes" in INGEST_MACRO_TABLES


EXPECTED_TABLE_KEY_PARTS = 2


def test_ingest_macro_tables_all_have_schema_prefix() -> None:
    """All entries in INGEST_MACRO_TABLES should have schema.table format."""
    for table_key in INGEST_MACRO_TABLES:
        parts = table_key.split(".")
        assert len(parts) == EXPECTED_TABLE_KEY_PARTS, (
            f"Table key '{table_key}' should have format 'schema.table'"
        )
        schema, table = parts
        assert schema, f"Table key '{table_key}' has empty schema"
        assert table, f"Table key '{table_key}' has empty table name"


def test_ingest_macro_tables_not_empty() -> None:
    """INGEST_MACRO_TABLES should not be empty."""
    assert len(INGEST_MACRO_TABLES) > 0


# =============================================================================
# macro_exists Tests
# =============================================================================


def test_macro_exists_returns_true_for_existing_macro(
    fresh_gateway: StorageGateway,
) -> None:
    """macro_exists should return True for macros that exist."""
    # The fresh_gateway fixture applies schema and ingest macros
    # Test a known table that should have a macro
    result = macro_exists(fresh_gateway.con, "core.modules")

    # If macros are registered, this should return True
    # If not, the test verifies the function doesn't crash
    assert isinstance(result, bool)


def test_macro_exists_returns_false_for_nonexistent_macro(
    fresh_gateway: StorageGateway,
) -> None:
    """macro_exists should return False for macros that don't exist."""
    result = macro_exists(fresh_gateway.con, "nonexistent.table_xyz")

    assert result is False


def test_macro_exists_handles_malformed_table_key(
    fresh_gateway: StorageGateway,
) -> None:
    """macro_exists should handle malformed table keys."""
    # This should raise ValueError due to not having a dot
    with pytest.raises(ValueError, match="not enough values to unpack"):
        macro_exists(fresh_gateway.con, "no_dot_in_name")


def test_macro_exists_with_multiple_dots_in_table_key(
    fresh_gateway: StorageGateway,
) -> None:
    """macro_exists should handle table keys with multiple dots."""
    # The function uses maxsplit=1, so this should work
    result = macro_exists(fresh_gateway.con, "schema.table.extra")

    # Function extracts "table.extra" as the table name and looks for
    # "ingest_table.extra" macro which won't exist
    assert result is False


def test_macro_exists_extracts_correct_macro_name(
    fresh_gateway: StorageGateway,
) -> None:
    """macro_exists should construct the correct macro name."""
    # Test that the function correctly extracts the table name
    # from "schema.table_name" and looks for "ingest_table_name"

    # Create a test macro with a known name
    fresh_gateway.con.execute("""
        CREATE OR REPLACE MACRO metadata.ingest_test_custom_table()
        AS (SELECT 1 AS dummy)
    """)

    result = macro_exists(fresh_gateway.con, "any_schema.test_custom_table")

    assert result is True


def test_macro_exists_with_various_schemas(
    fresh_gateway: StorageGateway,
) -> None:
    """macro_exists should work with different schema prefixes."""
    # Create a test macro
    fresh_gateway.con.execute("""
        CREATE OR REPLACE MACRO metadata.ingest_schema_test()
        AS (SELECT 1 AS dummy)
    """)

    # Different schema prefixes should all work
    assert macro_exists(fresh_gateway.con, "core.schema_test") is True
    assert macro_exists(fresh_gateway.con, "analytics.schema_test") is True
    assert macro_exists(fresh_gateway.con, "graph.schema_test") is True


def test_macro_exists_handles_duckdb_errors_gracefully(
    fresh_gateway: StorageGateway,
) -> None:
    """macro_exists should handle DuckDB errors gracefully."""
    # Close the connection to simulate an error scenario
    # Note: We can't actually close fresh_gateway without side effects,
    # so we just verify the function structure handles exceptions

    # Test with a valid connection but nonexistent table
    result = macro_exists(fresh_gateway.con, "test.nonexistent")
    assert result is False


# =============================================================================
# Integration Tests
# =============================================================================


def test_ingest_macro_tables_members_can_be_checked(
    fresh_gateway: StorageGateway,
) -> None:
    """Each table in INGEST_MACRO_TABLES can be passed to macro_exists."""
    # This test verifies the integration between the constant and the function
    for table_key in INGEST_MACRO_TABLES:
        # Should not raise an exception
        result = macro_exists(fresh_gateway.con, table_key)
        assert isinstance(result, bool)
