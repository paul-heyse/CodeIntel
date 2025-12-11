"""Tests for IbisGateway.write() unified write API."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from codeintel.storage.ibis_adapter import OnConflict, WriteResult
from tests._helpers.gateway import GatewayFactory

if TYPE_CHECKING:
    from codeintel.storage.gateway.protocol import StorageGateway


@pytest.fixture
def write_gateway(tmp_path: Path) -> StorageGateway:
    """Create a gateway with test tables for write operations.

    Returns
    -------
    StorageGateway
        Gateway with analytics.write_test table created.
    """
    gateway = GatewayFactory().file_backed(tmp_path / "write.duckdb").with_schema().open()

    # Create test tables
    gateway.con.execute("CREATE SCHEMA IF NOT EXISTS analytics")
    gateway.con.execute(
        """
        CREATE TABLE IF NOT EXISTS analytics.write_test (
            id INTEGER,
            name VARCHAR,
            value DOUBLE
        )
        """
    )
    gateway.con.execute(
        """
        CREATE TABLE IF NOT EXISTS analytics.upsert_test (
            id INTEGER PRIMARY KEY,
            name VARCHAR,
            value DOUBLE
        )
        """
    )
    return gateway


def test_write_dataframe_basic(write_gateway: StorageGateway) -> None:
    """Verify DataFrame can be written to table."""
    df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"], "value": [1.0, 2.0, 3.0]})

    result = write_gateway.ibis.write("analytics.write_test", df)

    expected_rows = 3
    assert isinstance(result, WriteResult)
    assert result.table_key == "analytics.write_test"
    assert result.rows_affected == expected_rows
    assert result.method == "insert_values"

    # Verify data was written
    read_df = write_gateway.ibis.table("analytics.write_test").to_pandas()
    assert len(read_df) == expected_rows
    assert list(read_df["id"]) == [1, 2, 3]


def test_write_dataframe_with_columns(write_gateway: StorageGateway) -> None:
    """Verify DataFrame can be written with explicit columns."""
    df = pd.DataFrame({"a": [4, 5], "b": ["x", "y"], "c": [4.0, 5.0]})

    result = write_gateway.ibis.write(
        "analytics.write_test",
        df,
        columns=["id", "name", "value"],
    )

    expected_rows = 2
    assert result.rows_affected == expected_rows

    read_df = write_gateway.ibis.table("analytics.write_test").to_pandas()
    assert len(read_df) == expected_rows


def test_write_tuples_basic(write_gateway: StorageGateway) -> None:
    """Verify tuples can be written to table."""
    rows = [(1, "a", 1.0), (2, "b", 2.0)]

    result = write_gateway.ibis.write(
        "analytics.write_test",
        rows,
        columns=["id", "name", "value"],
    )

    expected_rows = 2
    assert result.rows_affected == expected_rows
    assert result.method == "insert_values"

    read_df = write_gateway.ibis.table("analytics.write_test").to_pandas()
    assert len(read_df) == expected_rows


def test_write_tuples_empty(write_gateway: StorageGateway) -> None:
    """Verify empty tuple list returns zero rows."""
    result = write_gateway.ibis.write(
        "analytics.write_test",
        [],
        columns=["id", "name", "value"],
    )

    assert result.rows_affected == 0
    assert result.method == "noop"


def test_write_ibis_expression_basic(write_gateway: StorageGateway) -> None:
    """Verify Ibis expression can be written via INSERT...SELECT."""
    # Seed source data
    df = pd.DataFrame({"id": [10, 20], "name": ["x", "y"], "value": [10.0, 20.0]})
    write_gateway.ibis.write("analytics.write_test", df)

    # Create Ibis expression that transforms data
    source = write_gateway.ibis.table("analytics.write_test")
    transformed = source.mutate(value=source.value * 2)

    # Write to a new insert (same table for simplicity)
    result = write_gateway.ibis.write("analytics.write_test", transformed)

    assert result.method == "insert_select"
    # rows_affected is -1 for INSERT...SELECT (not known without counting)
    assert result.rows_affected == -1

    # Verify we now have 4 rows (2 original + 2 from transform)
    expected_rows = 4
    read_df = write_gateway.ibis.table("analytics.write_test").to_pandas()
    assert len(read_df) == expected_rows


def test_upsert_insert_new_rows(write_gateway: StorageGateway) -> None:
    """Verify UPSERT inserts new rows when no conflict."""
    rows = [(1, "a", 1.0), (2, "b", 2.0)]

    result = write_gateway.ibis.write(
        "analytics.upsert_test",
        rows,
        columns=["id", "name", "value"],
        on_conflict=OnConflict(conflict_columns=["id"]),
    )

    expected_rows = 2
    assert result.method == "upsert"
    assert result.rows_affected == expected_rows

    read_df = write_gateway.ibis.table("analytics.upsert_test").to_pandas()
    assert len(read_df) == expected_rows


def test_upsert_updates_on_conflict(write_gateway: StorageGateway) -> None:
    """Verify UPSERT updates existing rows on conflict."""
    # Insert initial data
    write_gateway.ibis.write(
        "analytics.upsert_test",
        [(1, "original", 1.0)],
        columns=["id", "name", "value"],
    )

    # Upsert with conflict on id
    expected_value = 99.0
    write_gateway.ibis.write(
        "analytics.upsert_test",
        [(1, "updated", expected_value)],
        columns=["id", "name", "value"],
        on_conflict=OnConflict(conflict_columns=["id"]),
    )

    # Verify update occurred
    read_df = write_gateway.ibis.table("analytics.upsert_test").to_pandas()
    assert len(read_df) == 1
    assert read_df.iloc[0]["name"] == "updated"
    assert read_df.iloc[0]["value"] == expected_value


def test_upsert_selective_update_columns(write_gateway: StorageGateway) -> None:
    """Verify UPSERT can selectively update columns."""
    # Insert initial data
    write_gateway.ibis.write(
        "analytics.upsert_test",
        [(1, "original", 1.0)],
        columns=["id", "name", "value"],
    )

    # Upsert updating only 'value', not 'name'
    expected_value = 99.0
    write_gateway.ibis.write(
        "analytics.upsert_test",
        [(1, "should_not_update", expected_value)],
        columns=["id", "name", "value"],
        on_conflict=OnConflict(conflict_columns=["id"], update_columns=["value"]),
    )

    # Verify only value was updated
    read_df = write_gateway.ibis.table("analytics.upsert_test").to_pandas()
    assert len(read_df) == 1
    assert read_df.iloc[0]["name"] == "original"  # Not updated
    assert read_df.iloc[0]["value"] == expected_value  # Updated


def test_write_unqualified_table_key_raises(write_gateway: StorageGateway) -> None:
    """Verify unqualified table key raises ValueError."""
    df = pd.DataFrame({"id": [1]})

    with pytest.raises(ValueError, match="schema-qualified"):
        write_gateway.ibis.write("no_schema", df)


def test_write_tuples_without_columns_raises(write_gateway: StorageGateway) -> None:
    """Verify writing tuples without columns raises ValueError."""
    rows = [(1, "a", 1.0)]

    with pytest.raises(ValueError, match="columns must be provided"):
        write_gateway.ibis.write("analytics.write_test", rows)


def test_write_unsupported_type_raises(write_gateway: StorageGateway) -> None:
    """Verify unsupported data type raises TypeError."""
    with pytest.raises(TypeError, match="Unsupported data type"):
        write_gateway.ibis.write("analytics.write_test", "not a valid data type")  # type: ignore[arg-type]
