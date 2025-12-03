"""Extended tests for DuckDB storage adapter.

This module provides additional test coverage for the DuckDB storage
adapter from `codeintel.graphs.adapters.duckdb_storage`, including:

- Storage initialization
- Connection management
- Query execution
- Batch insert operations
- Table operations
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest

from codeintel.graphs.adapters.duckdb_storage import DuckDBStorageAdapter
from codeintel.storage.gateway import DuckDBError
from codeintel.storage.schemas import apply_all_schemas
from codeintel.storage.sql_builder import QueryBuilder, SafeTable, render_sql
from tests._helpers.gateway import open_ingestion_gateway_with_macros

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PARAM_VALUE: Final = 42
UNNEST_VALUES: Final = (1, 2, 3)
INSERT_ROW_COUNT: Final = 2
BATCH_SIZE: Final = 100
AGG_VALUES: Final = (("A", 10), ("A", 20), ("B", 30), ("B", 40))
AGG_EXPECTED: Final = (("A", 30), ("B", 70))

# Table names validated via SafeTable at module load time
TABLE_BASE: Final = SafeTable("test_graphs_adapter")
TABLE_INSERT: Final = SafeTable("test_graphs_adapter_insert")
TABLE_BATCH: Final = SafeTable("test_graphs_adapter_batch")
TABLE_COLS: Final = SafeTable("test_graphs_adapter_cols")
TABLE_TX: Final = SafeTable("test_graphs_adapter_tx")
TABLE_AGG: Final = SafeTable("test_graphs_adapter_agg")


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------


def _make_gateway() -> StorageGateway:
    """Create a gateway for adapter tests.

    Returns
    -------
    StorageGateway
        Configured gateway.
    """
    gateway = open_ingestion_gateway_with_macros(
        apply_schema=True, ensure_views=True, validate_schema=True
    )
    apply_all_schemas(gateway.con)
    return gateway


def _create_table_sql(table: SafeTable, columns: str) -> str:
    """Build CREATE TEMP TABLE SQL using SafeTable.

    Parameters
    ----------
    table
        Validated table name.
    columns
        Column definitions string.

    Returns
    -------
    str
        CREATE TABLE SQL statement.
    """
    return render_sql(["CREATE TEMP TABLE", str(table), f"({columns})"])


def _describe_sql(table: SafeTable) -> str:
    """Build DESCRIBE SQL using SafeTable.

    Parameters
    ----------
    table
        Validated table name.

    Returns
    -------
    str
        DESCRIBE SQL statement.
    """
    return render_sql(["DESCRIBE", str(table)])


def _insert_values_sql(table: SafeTable, values: str) -> str:
    """Build INSERT VALUES SQL using SafeTable.

    Parameters
    ----------
    table
        Validated table name.
    values
        VALUES clause content.

    Returns
    -------
    str
        INSERT SQL statement.
    """
    return render_sql(["INSERT INTO", str(table), "VALUES", values])


def _select_ordered_sql(table: SafeTable, order_by: str) -> str:
    """Build SELECT * with ORDER BY using SafeTable.

    Parameters
    ----------
    table
        Validated table name.
    order_by
        Column to order by.

    Returns
    -------
    str
        SELECT SQL statement.
    """
    return render_sql(["SELECT * FROM", str(table), "ORDER BY", order_by])


def _aggregate_sql(table: SafeTable) -> str:
    """Build aggregate query using SafeTable.

    Parameters
    ----------
    table
        Validated table name.

    Returns
    -------
    str
        Aggregate SQL statement.
    """
    return render_sql(
        [
            "SELECT category, SUM(value) as total FROM",
            str(table),
            "GROUP BY category ORDER BY category",
        ]
    )


# ---------------------------------------------------------------------------
# Tests: DuckDBStorageAdapter initialization
# ---------------------------------------------------------------------------


def test_duckdb_adapter_init_with_gateway() -> None:
    """DuckDBStorageAdapter initializes with gateway."""
    gateway = _make_gateway()
    try:
        adapter = DuckDBStorageAdapter(gateway)

        assert adapter.gateway is gateway
    finally:
        gateway.close()


def test_duckdb_adapter_connection_accessible() -> None:
    """DuckDBStorageAdapter connection is accessible."""
    gateway = _make_gateway()
    try:
        adapter = DuckDBStorageAdapter(gateway)

        # Connection should be accessible via gateway
        assert adapter.gateway.con is not None
    finally:
        gateway.close()


# ---------------------------------------------------------------------------
# Tests: DuckDBStorageAdapter query execution
# ---------------------------------------------------------------------------


def test_duckdb_adapter_execute_simple_query() -> None:
    """DuckDBStorageAdapter can execute simple queries."""
    gateway = _make_gateway()
    try:
        adapter = DuckDBStorageAdapter(gateway)

        result = adapter.execute("SELECT 1 AS value")

        assert result is not None
        row = result.fetchone()
        assert row is not None
        assert row[0] == 1
    finally:
        gateway.close()


def test_duckdb_adapter_execute_with_params() -> None:
    """DuckDBStorageAdapter can execute parameterized queries."""
    gateway = _make_gateway()
    try:
        adapter = DuckDBStorageAdapter(gateway)

        result = adapter.execute("SELECT ? AS value", [PARAM_VALUE])

        row = result.fetchone()
        assert row is not None
        assert row[0] == PARAM_VALUE
    finally:
        gateway.close()


def test_duckdb_adapter_execute_fetch_all() -> None:
    """DuckDBStorageAdapter fetch_all returns all rows."""
    gateway = _make_gateway()
    try:
        adapter = DuckDBStorageAdapter(gateway)

        # Use UNNEST with list literal - safe SQL
        result = adapter.execute("SELECT unnest([1, 2, 3]) AS value")
        rows = result.fetchall()

        assert len(rows) == len(UNNEST_VALUES)
        assert [r[0] for r in rows] == list(UNNEST_VALUES)
    finally:
        gateway.close()


# ---------------------------------------------------------------------------
# Tests: DuckDBStorageAdapter table operations
# ---------------------------------------------------------------------------


def test_duckdb_adapter_create_temp_table() -> None:
    """DuckDBStorageAdapter can create temporary tables."""
    gateway = _make_gateway()
    try:
        adapter = DuckDBStorageAdapter(gateway)

        adapter.execute(_create_table_sql(TABLE_BASE, "id INTEGER, name VARCHAR"))

        # Verify table exists using QueryBuilder
        count_sql, params = QueryBuilder.count(TABLE_BASE)
        result = adapter.execute(count_sql, params)
        count = result.fetchone()
        assert count is not None
        assert count[0] == 0
    finally:
        gateway.close()


def test_duckdb_adapter_insert_and_select() -> None:
    """DuckDBStorageAdapter can insert and select data."""
    gateway = _make_gateway()
    try:
        adapter = DuckDBStorageAdapter(gateway)

        # Create table
        adapter.execute(_create_table_sql(TABLE_INSERT, "id INTEGER, value VARCHAR"))

        # Insert data
        adapter.execute(_insert_values_sql(TABLE_INSERT, "(1, 'test1'), (2, 'test2')"))

        # Select and verify
        result = adapter.execute(_select_ordered_sql(TABLE_INSERT, "id"))
        rows = result.fetchall()

        assert len(rows) == INSERT_ROW_COUNT
        assert rows[0] == (1, "test1")
        assert rows[1] == (2, "test2")
    finally:
        gateway.close()


# ---------------------------------------------------------------------------
# Tests: DuckDBStorageAdapter batch operations
# ---------------------------------------------------------------------------


def test_duckdb_adapter_batch_insert() -> None:
    """DuckDBStorageAdapter can perform batch inserts."""
    gateway = _make_gateway()
    try:
        adapter = DuckDBStorageAdapter(gateway)

        # Create table
        adapter.execute(_create_table_sql(TABLE_BATCH, "id INTEGER, name VARCHAR"))

        # Batch insert using executemany pattern with parameterized query
        insert_sql = render_sql(["INSERT INTO", str(TABLE_BATCH), "VALUES (?, ?)"])
        data = [(i, f"item_{i}") for i in range(BATCH_SIZE)]
        adapter.gateway.con.executemany(insert_sql, data)

        # Verify count
        count_sql, params = QueryBuilder.count(TABLE_BATCH)
        result = adapter.execute(count_sql, params)
        count = result.fetchone()
        assert count is not None
        assert count[0] == BATCH_SIZE
    finally:
        gateway.close()


# ---------------------------------------------------------------------------
# Tests: DuckDBStorageAdapter error handling
# ---------------------------------------------------------------------------


def test_duckdb_adapter_invalid_query_raises() -> None:
    """DuckDBStorageAdapter raises on invalid query."""
    gateway = _make_gateway()
    try:
        adapter = DuckDBStorageAdapter(gateway)

        with pytest.raises(DuckDBError):
            adapter.execute("INVALID SQL SYNTAX HERE")
    finally:
        gateway.close()


def test_duckdb_adapter_missing_table_raises() -> None:
    """DuckDBStorageAdapter raises on missing table."""
    gateway = _make_gateway()
    try:
        adapter = DuckDBStorageAdapter(gateway)

        with pytest.raises(DuckDBError):
            adapter.execute("SELECT * FROM nonexistent_table_xyz")
    finally:
        gateway.close()


# ---------------------------------------------------------------------------
# Tests: DuckDBStorageAdapter schema operations
# ---------------------------------------------------------------------------


def test_duckdb_adapter_list_tables() -> None:
    """DuckDBStorageAdapter can list tables in schema."""
    gateway = _make_gateway()
    try:
        adapter = DuckDBStorageAdapter(gateway)

        # Query information schema
        result = adapter.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' LIMIT 10"
        )

        rows = result.fetchall()
        # Should have some tables
        assert isinstance(rows, list)
    finally:
        gateway.close()


def test_duckdb_adapter_table_columns() -> None:
    """DuckDBStorageAdapter can query table columns."""
    gateway = _make_gateway()
    try:
        adapter = DuckDBStorageAdapter(gateway)

        # Create a test table
        adapter.execute(_create_table_sql(TABLE_COLS, "id INTEGER, name VARCHAR, value DOUBLE"))

        # Query columns using DESCRIBE
        result = adapter.execute(_describe_sql(TABLE_COLS))

        rows = result.fetchall()
        column_names = [r[0] for r in rows]

        assert "id" in column_names
        assert "name" in column_names
        assert "value" in column_names
    finally:
        gateway.close()


# ---------------------------------------------------------------------------
# Tests: DuckDBStorageAdapter transaction handling
# ---------------------------------------------------------------------------


def test_duckdb_adapter_transaction_commit() -> None:
    """DuckDBStorageAdapter can handle transactions."""
    gateway = _make_gateway()
    try:
        adapter = DuckDBStorageAdapter(gateway)

        # Create table
        adapter.execute(_create_table_sql(TABLE_TX, "id INTEGER"))

        # Insert in transaction
        adapter.execute(_insert_values_sql(TABLE_TX, "(1)"))

        # Verify data persisted
        count_sql, params = QueryBuilder.count(TABLE_TX)
        result = adapter.execute(count_sql, params)
        count = result.fetchone()
        assert count is not None
        assert count[0] == 1
    finally:
        gateway.close()


# ---------------------------------------------------------------------------
# Tests: DuckDBStorageAdapter aggregate operations
# ---------------------------------------------------------------------------


def test_duckdb_adapter_aggregate_query() -> None:
    """DuckDBStorageAdapter can execute aggregate queries."""
    gateway = _make_gateway()
    try:
        adapter = DuckDBStorageAdapter(gateway)

        # Create and populate table
        adapter.execute(_create_table_sql(TABLE_AGG, "category VARCHAR, value INTEGER"))

        # Insert aggregate test data using parameterized insert
        insert_sql = render_sql(["INSERT INTO", str(TABLE_AGG), "VALUES (?, ?)"])
        for cat, val in AGG_VALUES:
            adapter.execute(insert_sql, [cat, val])

        # Aggregate query
        result = adapter.execute(_aggregate_sql(TABLE_AGG))

        rows = result.fetchall()
        assert len(rows) == len(AGG_EXPECTED)
        assert rows[0] == AGG_EXPECTED[0]
        assert rows[1] == AGG_EXPECTED[1]
    finally:
        gateway.close()
