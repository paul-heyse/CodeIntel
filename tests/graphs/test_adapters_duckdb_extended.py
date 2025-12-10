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

from typing import Final

import pytest

from codeintel.graphs.adapters.duckdb_storage import DuckDBStorageAdapter
from codeintel.storage.gateway import DuckDBError, StorageGateway
from codeintel.storage.sql import QueryBuilder, SafeTable, render_sql
from tests._helpers.assertions import (
    expect_in,
    expect_is_instance,
    expect_is_not_none,
    expect_row_value,
    expect_rows_equal,
    expect_true,
)

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


def test_duckdb_adapter_init_with_gateway(graph_gateway: StorageGateway) -> None:
    """DuckDBStorageAdapter initializes with gateway."""
    adapter = DuckDBStorageAdapter(graph_gateway)

    expect_true(adapter.gateway is graph_gateway)


def test_duckdb_adapter_connection_accessible(graph_gateway: StorageGateway) -> None:
    """DuckDBStorageAdapter connection is accessible."""
    adapter = DuckDBStorageAdapter(graph_gateway)

    # Connection should be accessible via gateway
    expect_is_not_none(adapter.gateway.con)


# ---------------------------------------------------------------------------
# Tests: DuckDBStorageAdapter query execution
# ---------------------------------------------------------------------------


def test_duckdb_adapter_execute_simple_query(graph_gateway: StorageGateway) -> None:
    """DuckDBStorageAdapter can execute simple queries."""
    adapter = DuckDBStorageAdapter(graph_gateway)

    result = adapter.execute("SELECT 1 AS value")

    expect_is_not_none(result)
    row = result.fetchone()
    expect_row_value(row, 0, 1, message="Expected a row from SELECT")


def test_duckdb_adapter_execute_with_params(graph_gateway: StorageGateway) -> None:
    """DuckDBStorageAdapter can execute parameterized queries."""
    adapter = DuckDBStorageAdapter(graph_gateway)

    result = adapter.execute("SELECT ? AS value", [PARAM_VALUE])

    row = result.fetchone()
    expect_row_value(row, 0, PARAM_VALUE, message="Expected a row from parameterized query")


def test_duckdb_adapter_execute_fetch_all(graph_gateway: StorageGateway) -> None:
    """DuckDBStorageAdapter fetch_all returns all rows."""
    adapter = DuckDBStorageAdapter(graph_gateway)

    # Use UNNEST with list literal - safe SQL
    result = adapter.execute("SELECT unnest([1, 2, 3]) AS value")
    rows = result.fetchall()

    expect_rows_equal(rows, [(value,) for value in UNNEST_VALUES])


# ---------------------------------------------------------------------------
# Tests: DuckDBStorageAdapter table operations
# ---------------------------------------------------------------------------


def test_duckdb_adapter_create_temp_table(graph_gateway: StorageGateway) -> None:
    """DuckDBStorageAdapter can create temporary tables."""
    adapter = DuckDBStorageAdapter(graph_gateway)

    adapter.execute(_create_table_sql(TABLE_BASE, "id INTEGER, name VARCHAR"))

    # Verify table exists using QueryBuilder
    count_sql, params = QueryBuilder.count(TABLE_BASE)
    result = adapter.execute(count_sql, params)
    count = result.fetchone()
    expect_row_value(count, 0, 0, message="Expected count row from QueryBuilder.count")


def test_duckdb_adapter_insert_and_select(graph_gateway: StorageGateway) -> None:
    """DuckDBStorageAdapter can insert and select data."""
    adapter = DuckDBStorageAdapter(graph_gateway)

    # Create table
    adapter.execute(_create_table_sql(TABLE_INSERT, "id INTEGER, value VARCHAR"))

    # Insert data
    adapter.execute(_insert_values_sql(TABLE_INSERT, "(1, 'test1'), (2, 'test2')"))

    # Select and verify
    result = adapter.execute(_select_ordered_sql(TABLE_INSERT, "id"))
    rows = result.fetchall()

    expect_rows_equal(rows, [(1, "test1"), (2, "test2")])


# ---------------------------------------------------------------------------
# Tests: DuckDBStorageAdapter batch operations
# ---------------------------------------------------------------------------


def test_duckdb_adapter_batch_insert(graph_gateway: StorageGateway) -> None:
    """DuckDBStorageAdapter can perform batch inserts."""
    adapter = DuckDBStorageAdapter(graph_gateway)

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
    expect_row_value(count, 0, BATCH_SIZE, message="Expected count row for batch insert")


# ---------------------------------------------------------------------------
# Tests: DuckDBStorageAdapter error handling
# ---------------------------------------------------------------------------


def test_duckdb_adapter_invalid_query_raises(graph_gateway: StorageGateway) -> None:
    """DuckDBStorageAdapter raises on invalid query."""
    adapter = DuckDBStorageAdapter(graph_gateway)

    with pytest.raises(DuckDBError):
        adapter.execute("INVALID SQL SYNTAX HERE")


def test_duckdb_adapter_missing_table_raises(graph_gateway: StorageGateway) -> None:
    """DuckDBStorageAdapter raises on missing table."""
    adapter = DuckDBStorageAdapter(graph_gateway)

    with pytest.raises(DuckDBError):
        adapter.execute("SELECT * FROM nonexistent_table_xyz")


# ---------------------------------------------------------------------------
# Tests: DuckDBStorageAdapter schema operations
# ---------------------------------------------------------------------------


def test_duckdb_adapter_list_tables(graph_gateway: StorageGateway) -> None:
    """DuckDBStorageAdapter can list tables in schema."""
    adapter = DuckDBStorageAdapter(graph_gateway)

    # Query information schema
    result = adapter.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' LIMIT 10"
    )

    rows = result.fetchall()
    expect_is_instance(rows, list)


def test_duckdb_adapter_table_columns(graph_gateway: StorageGateway) -> None:
    """DuckDBStorageAdapter can query table columns."""
    adapter = DuckDBStorageAdapter(graph_gateway)

    # Create a test table
    adapter.execute(_create_table_sql(TABLE_COLS, "id INTEGER, name VARCHAR, value DOUBLE"))

    # Query columns using DESCRIBE
    result = adapter.execute(_describe_sql(TABLE_COLS))

    rows = result.fetchall()
    column_names = [r[0] for r in rows]

    expect_in("id", column_names)
    expect_in("name", column_names)
    expect_in("value", column_names)


# ---------------------------------------------------------------------------
# Tests: DuckDBStorageAdapter transaction handling
# ---------------------------------------------------------------------------


def test_duckdb_adapter_transaction_commit(graph_gateway: StorageGateway) -> None:
    """DuckDBStorageAdapter can handle transactions."""
    adapter = DuckDBStorageAdapter(graph_gateway)

    # Create table
    adapter.execute(_create_table_sql(TABLE_TX, "id INTEGER"))

    # Insert in transaction
    adapter.execute(_insert_values_sql(TABLE_TX, "(1)"))

    # Verify data persisted
    count_sql, params = QueryBuilder.count(TABLE_TX)
    result = adapter.execute(count_sql, params)
    count = result.fetchone()
    expect_row_value(count, 0, 1, message="Expected count row after transaction")


# ---------------------------------------------------------------------------
# Tests: DuckDBStorageAdapter aggregate operations
# ---------------------------------------------------------------------------


def test_duckdb_adapter_aggregate_query(graph_gateway: StorageGateway) -> None:
    """DuckDBStorageAdapter can execute aggregate queries."""
    adapter = DuckDBStorageAdapter(graph_gateway)

    # Create and populate table
    adapter.execute(_create_table_sql(TABLE_AGG, "category VARCHAR, value INTEGER"))

    # Insert aggregate test data using parameterized insert
    insert_sql = render_sql(["INSERT INTO", str(TABLE_AGG), "VALUES (?, ?)"])
    for cat, val in AGG_VALUES:
        adapter.execute(insert_sql, [cat, val])

    # Aggregate query
    result = adapter.execute(_aggregate_sql(TABLE_AGG))

    rows = result.fetchall()
    expect_rows_equal(rows, AGG_EXPECTED)
