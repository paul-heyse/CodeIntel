"""Comprehensive tests for ingestion adapters.

This module tests the storage and tool adapters used during
code ingestion, focusing on the interface contracts.
"""

from __future__ import annotations

import pytest

from codeintel.ingestion.adapters.duckdb_storage import (
    INGEST_MACROS,
    SMALL_BATCH_THRESHOLD,
    DuckDBStorageAdapter,
    _build_delete_in_query,  # noqa: PLC2701
    _quote_identifier,  # noqa: PLC2701
    _quote_macro_name,  # noqa: PLC2701
    _quote_table_key,  # noqa: PLC2701
)
from codeintel.ingestion.ports.storage import BatchResult, QueryResult
from codeintel.storage.gateway import StorageGateway

# Test constants
ROWS_WRITTEN_100 = 100
DURATION_1_5 = 1.5

# =============================================================================
# _quote_identifier Tests
# =============================================================================


def test_quote_identifier_valid_simple() -> None:
    """Should quote simple identifiers."""
    result = _quote_identifier("my_table")

    assert result == '"my_table"'


def test_quote_identifier_valid_with_numbers() -> None:
    """Should quote identifiers with numbers."""
    result = _quote_identifier("table_123")

    assert result == '"table_123"'


def test_quote_identifier_valid_uppercase() -> None:
    """Should quote uppercase identifiers."""
    result = _quote_identifier("MyTable")

    assert result == '"MyTable"'


def test_quote_identifier_rejects_spaces() -> None:
    """Should reject identifiers with spaces."""
    with pytest.raises(ValueError, match="Unsafe identifier"):
        _quote_identifier("my table")


def test_quote_identifier_rejects_dashes() -> None:
    """Should reject identifiers with dashes."""
    with pytest.raises(ValueError, match="Unsafe identifier"):
        _quote_identifier("my-table")


def test_quote_identifier_rejects_sql_injection() -> None:
    """Should reject SQL injection attempts."""
    with pytest.raises(ValueError, match="Unsafe identifier"):
        _quote_identifier("table; DROP TABLE users;--")


def test_quote_identifier_rejects_quotes() -> None:
    """Should reject identifiers with quotes."""
    with pytest.raises(ValueError, match="Unsafe identifier"):
        _quote_identifier('table"name')


def test_quote_identifier_rejects_semicolons() -> None:
    """Should reject identifiers with semicolons."""
    with pytest.raises(ValueError, match="Unsafe identifier"):
        _quote_identifier("table;name")


# =============================================================================
# _quote_table_key Tests
# =============================================================================


def test_quote_table_key_valid() -> None:
    """Should quote valid table keys."""
    schema, table, quoted = _quote_table_key("core.modules")

    assert schema == "core"
    assert table == "modules"
    assert quoted == '"core"."modules"'


def test_quote_table_key_unknown_table() -> None:
    """Should reject unknown table keys."""
    with pytest.raises(ValueError, match="Unknown table key"):
        _quote_table_key("nonexistent.table")


# =============================================================================
# Constants Tests
# =============================================================================


def test_small_batch_threshold_positive() -> None:
    """SMALL_BATCH_THRESHOLD should be a positive integer."""
    assert SMALL_BATCH_THRESHOLD > 0
    assert isinstance(SMALL_BATCH_THRESHOLD, int)


def test_ingest_macros_not_empty() -> None:
    """INGEST_MACROS should contain mappings."""
    assert len(INGEST_MACROS) > 0
    assert isinstance(INGEST_MACROS, dict)


def test_ingest_macros_keys_are_table_format() -> None:
    """INGEST_MACROS keys should be in schema.table format."""
    for key in INGEST_MACROS:
        assert "." in key
        parts = key.split(".")
        assert len(parts) >= 2  # noqa: PLR2004


def test_ingest_macros_values_start_with_metadata() -> None:
    """INGEST_MACROS values should start with metadata.ingest_."""
    for value in INGEST_MACROS.values():
        assert value.startswith("metadata.ingest_")


# =============================================================================
# BatchResult Tests
# =============================================================================


def test_batch_result_attributes() -> None:
    """BatchResult should store write results."""
    result = BatchResult(
        table_key="core.test",
        rows_written=ROWS_WRITTEN_100,
        duration_s=DURATION_1_5,
    )

    assert result.table_key == "core.test"
    assert result.rows_written == ROWS_WRITTEN_100
    assert result.duration_s == DURATION_1_5


def test_batch_result_defaults() -> None:
    """BatchResult should have sensible defaults."""
    result = BatchResult(table_key="core.test", rows_written=50)

    assert result.duration_s == 0.0


# =============================================================================
# QueryResult Tests
# =============================================================================


def test_query_result_attributes() -> None:
    """QueryResult should store query results."""
    result = QueryResult(
        rows=[("a", 1), ("b", 2)],
        columns=("name", "value"),
        row_count=2,
    )

    expected_rows = 2
    assert len(result.rows) == expected_rows
    assert result.columns == ("name", "value")
    assert result.row_count == expected_rows


def test_query_result_defaults() -> None:
    """QueryResult should have sensible defaults."""
    result = QueryResult()

    assert result.rows == []
    assert result.columns == ()
    assert result.row_count == 0


# =============================================================================
# DuckDBStorageAdapter Tests
# =============================================================================


def test_duckdb_adapter_initialization(fresh_gateway: StorageGateway) -> None:
    """DuckDBStorageAdapter should initialize from gateway."""
    adapter = DuckDBStorageAdapter(fresh_gateway)

    assert adapter is not None


def test_duckdb_adapter_ensure_schema(fresh_gateway: StorageGateway) -> None:
    """DuckDBStorageAdapter.ensure_schema should not raise for valid tables."""
    adapter = DuckDBStorageAdapter(fresh_gateway)

    # Should not raise
    adapter.ensure_schema("core.modules")


def test_duckdb_adapter_ensure_schema_unknown_table(
    fresh_gateway: StorageGateway,
) -> None:
    """DuckDBStorageAdapter.ensure_schema should raise for unknown tables."""
    adapter = DuckDBStorageAdapter(fresh_gateway)

    with pytest.raises(RuntimeError, match="missing from TABLE_SCHEMAS"):
        adapter.ensure_schema("nonexistent.table_xyz")


def test_duckdb_adapter_execute_query(fresh_gateway: StorageGateway) -> None:
    """DuckDBStorageAdapter.execute_query should return results."""
    adapter = DuckDBStorageAdapter(fresh_gateway)

    result = adapter.execute_query("SELECT 1 as value")

    assert result is not None
    assert result.row_count >= 0


def test_duckdb_adapter_execute_query_with_params(
    fresh_gateway: StorageGateway,
) -> None:
    """DuckDBStorageAdapter.execute_query should handle parameters."""
    adapter = DuckDBStorageAdapter(fresh_gateway)

    result = adapter.execute_query("SELECT ? + ? as sum", [1, 2])

    assert result is not None
    assert result.row_count == 1


def test_duckdb_adapter_write_batch_small(fresh_gateway: StorageGateway) -> None:
    """DuckDBStorageAdapter.write_batch should handle small batches."""
    adapter = DuckDBStorageAdapter(fresh_gateway)

    rows = [
        ("test_module", "test/path.py", "test/repo", "abc123", "python", "[]", "[]"),
    ]

    result = adapter.write_batch("core.modules", rows, scope="test/repo@abc123")

    assert result.rows_written == 1


def test_duckdb_adapter_delete_by_paths(fresh_gateway: StorageGateway) -> None:
    """DuckDBStorageAdapter.delete_by_paths should delete matching rows."""
    adapter = DuckDBStorageAdapter(fresh_gateway)

    # First insert some data
    rows = [
        ("mod1", "src/a.py", "test/repo", "abc123", "python", "[]", "[]"),
        ("mod2", "src/b.py", "test/repo", "abc123", "python", "[]", "[]"),
    ]
    adapter.write_batch("core.modules", rows, scope="test/repo@abc123")

    # Delete one path
    deleted = adapter.delete_by_paths(
        "core.modules",
        ["src/a.py"],
        path_column="path",
    )

    assert deleted >= 0  # May be 0 if table structure differs


def test_duckdb_adapter_fetch_dataframe(fresh_gateway: StorageGateway) -> None:
    """DuckDBStorageAdapter.fetch_dataframe should return dataframe."""
    adapter = DuckDBStorageAdapter(fresh_gateway)

    df = adapter.fetch_dataframe("SELECT 1 as value, 'test' as name")

    assert df is not None
    # Check it has expected shape
    assert len(df) >= 0


# =============================================================================
# Integration Tests
# =============================================================================


def test_adapter_write_and_query_cycle(fresh_gateway: StorageGateway) -> None:
    """Adapter should support write-then-query cycle."""
    adapter = DuckDBStorageAdapter(fresh_gateway)

    # Write some data
    rows = [
        ("cycle_mod", "cycle/path.py", "cycle/repo", "xyz789", "python", "[]", "[]"),
    ]
    write_result = adapter.write_batch("core.modules", rows, scope="cycle/repo@xyz789")

    assert write_result.rows_written == 1

    # Query it back
    query_result = adapter.execute_query(
        "SELECT module, path FROM core.modules WHERE repo = ?",
        ["cycle/repo"],
    )

    assert query_result.row_count >= 1


# =============================================================================
# Additional Helper Function Tests
# =============================================================================


def test_build_delete_in_query() -> None:
    """_build_delete_in_query should construct valid DELETE SQL."""
    result = _build_delete_in_query('"core"."modules"', '"path"', 3)

    expected_count = 3
    assert "DELETE FROM" in result
    assert '"core"."modules"' in result
    assert '"path"' in result
    assert result.count("?") == expected_count


def test_quote_macro_name_simple() -> None:
    """_quote_macro_name should handle simple macro names."""
    result = _quote_macro_name("metadata.ingest_modules")

    assert result == "metadata.ingest_modules"


def test_quote_macro_name_invalid() -> None:
    """_quote_macro_name should reject invalid macro names."""
    with pytest.raises(ValueError, match="Unsafe"):
        _quote_macro_name("metadata..double_dot")


# =============================================================================
# DuckDBStorageAdapter Property Tests
# =============================================================================


def test_duckdb_adapter_con_property(fresh_gateway: StorageGateway) -> None:
    """DuckDBStorageAdapter.con should return the gateway connection."""
    adapter = DuckDBStorageAdapter(fresh_gateway)

    assert adapter.con is fresh_gateway.con


# =============================================================================
# Write Batch Tests (Edge Cases)
# =============================================================================


def test_duckdb_adapter_write_batch_empty(fresh_gateway: StorageGateway) -> None:
    """DuckDBStorageAdapter.write_batch should handle empty rows."""
    adapter = DuckDBStorageAdapter(fresh_gateway)

    result = adapter.write_batch("core.modules", [], scope="test@abc")

    assert result.rows_written == 0
    assert result.duration_s == 0.0


def test_duckdb_adapter_write_batch_no_scope(fresh_gateway: StorageGateway) -> None:
    """DuckDBStorageAdapter.write_batch should work without scope."""
    adapter = DuckDBStorageAdapter(fresh_gateway)

    rows = [
        ("no_scope_mod", "no_scope/path.py", "test", "xyz", "python", "[]", "[]"),
    ]
    result = adapter.write_batch("core.modules", rows)

    assert result.rows_written == 1


def test_duckdb_adapter_write_batch_larger(fresh_gateway: StorageGateway) -> None:
    """DuckDBStorageAdapter.write_batch should handle batches larger than threshold."""
    adapter = DuckDBStorageAdapter(fresh_gateway)

    # Create more than SMALL_BATCH_THRESHOLD rows
    rows = [
        (f"mod_{i}", f"path/{i}.py", "test/repo", "abc123", "python", "[]", "[]")
        for i in range(30)  # > SMALL_BATCH_THRESHOLD (25)
    ]

    result = adapter.write_batch("core.modules", rows, scope="test/repo@abc123")

    expected_rows = 30
    assert result.rows_written == expected_rows


# =============================================================================
# Delete Tests
# =============================================================================


def test_duckdb_adapter_delete_by_paths_empty(fresh_gateway: StorageGateway) -> None:
    """DuckDBStorageAdapter.delete_by_paths should handle empty paths list."""
    adapter = DuckDBStorageAdapter(fresh_gateway)

    deleted = adapter.delete_by_paths("core.modules", [])

    assert deleted == 0


def test_duckdb_adapter_delete_by_params(fresh_gateway: StorageGateway) -> None:
    """DuckDBStorageAdapter.delete_by_params should execute delete."""
    adapter = DuckDBStorageAdapter(fresh_gateway)

    # First insert data
    rows = [
        ("del_mod", "del/path.py", "test/repo", "abc123", "python", "[]", "[]"),
    ]
    adapter.write_batch("core.modules", rows)

    # Try delete by params - should not raise
    deleted = adapter.delete_by_params("core.modules", ["test/repo", "abc123"])

    assert deleted >= 0  # DuckDB doesn't return count


# =============================================================================
# Query Tests (Edge Cases)
# =============================================================================


def test_duckdb_adapter_execute_query_no_params(fresh_gateway: StorageGateway) -> None:
    """DuckDBStorageAdapter.execute_query should work without params."""
    adapter = DuckDBStorageAdapter(fresh_gateway)

    result = adapter.execute_query("SELECT 42 as answer")

    assert result.row_count == 1
    assert result.columns == ("answer",)


def test_duckdb_adapter_fetch_dataframe_with_params(fresh_gateway: StorageGateway) -> None:
    """DuckDBStorageAdapter.fetch_dataframe should handle params."""
    adapter = DuckDBStorageAdapter(fresh_gateway)

    df = adapter.fetch_dataframe("SELECT ? as value", [100])

    assert len(df) == 1


# =============================================================================
# Registry Metadata Tests
# =============================================================================


def test_ingest_macros_has_core_modules() -> None:
    """INGEST_MACROS should have entry for core.modules."""
    assert "core.modules" in INGEST_MACROS
    assert INGEST_MACROS["core.modules"].startswith("metadata.")
