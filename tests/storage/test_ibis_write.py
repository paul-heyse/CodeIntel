"""Tests for IbisGateway.write() unified write API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import ibis.expr.types as it
import pandas as pd
import pytest

from codeintel.storage import ibis_adapter
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_instance,
)
from tests._helpers.gateway import GatewayFactory

if TYPE_CHECKING:
    from pathlib import Path

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
    expect_is_instance(result, ibis_adapter.WriteResult)
    expect_equal(result.table_key, "analytics.write_test")
    expect_equal(result.rows_affected, expected_rows)
    expect_equal(result.method, "insert_values")

    read_df = write_gateway.ibis.table("analytics.write_test").to_pandas()
    expect_equal(len(read_df), expected_rows)
    expect_equal(list(read_df["id"]), [1, 2, 3])


def test_write_dataframe_with_columns(write_gateway: StorageGateway) -> None:
    """Verify DataFrame can be written with explicit columns."""
    df = pd.DataFrame({"a": [4, 5], "b": ["x", "y"], "c": [4.0, 5.0]})

    result = write_gateway.ibis.write(
        "analytics.write_test",
        df,
        columns=["id", "name", "value"],
    )

    expected_rows = 2
    expect_equal(result.rows_affected, expected_rows)

    read_df = write_gateway.ibis.table("analytics.write_test").to_pandas()
    expect_equal(len(read_df), expected_rows)


def test_write_tuples_basic(write_gateway: StorageGateway) -> None:
    """Verify tuples can be written to table."""
    rows = [(1, "a", 1.0), (2, "b", 2.0)]

    result = write_gateway.ibis.write(
        "analytics.write_test",
        rows,
        columns=["id", "name", "value"],
    )

    expected_rows = 2
    expect_equal(result.rows_affected, expected_rows)
    expect_equal(result.method, "insert_values")

    read_df = write_gateway.ibis.table("analytics.write_test").to_pandas()
    expect_equal(len(read_df), expected_rows)


def test_write_tuples_empty(write_gateway: StorageGateway) -> None:
    """Verify empty tuple list returns zero rows."""
    result = write_gateway.ibis.write(
        "analytics.write_test",
        [],
        columns=["id", "name", "value"],
    )

    expect_equal(result.rows_affected, 0)
    expect_equal(result.method, "noop")


def test_write_ibis_expression_basic(write_gateway: StorageGateway) -> None:
    """Verify Ibis expression can be written via INSERT...SELECT."""
    df = pd.DataFrame({"id": [10, 20], "name": ["x", "y"], "value": [10.0, 20.0]})
    write_gateway.ibis.write("analytics.write_test", df)

    source = write_gateway.ibis.table("analytics.write_test")
    transformed = source.mutate(value=cast("Any", source.value) * 2)

    result = write_gateway.ibis.write("analytics.write_test", transformed)

    expect_equal(result.method, "insert_select")

    expect_equal(result.rows_affected, -1)

    expected_rows = 4
    read_df = write_gateway.ibis.table("analytics.write_test").to_pandas()
    expect_equal(len(read_df), expected_rows)


def test_write_dataframe_fast_lane_does_not_iterate_rows(
    write_gateway: StorageGateway,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify large-frame fast lane uses relation-based INSERT...SELECT."""
    monkeypatch.setattr(ibis_adapter, "_DATAFRAME_FAST_LANE_MIN_ROWS", 5)

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise AssertionError

    monkeypatch.setattr(pd.DataFrame, "itertuples", _boom, raising=True)

    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["a", "b", "c", "d", "e"],
            "value": [1, 2, 3, 4, 5],
        }
    )
    result = write_gateway.ibis.write("analytics.write_test", df)
    expect_equal(result.method, "insert_select")
    expect_equal(result.rows_affected, len(df))


def test_upsert_from_ibis_expression_does_not_materialize_to_pandas(
    write_gateway: StorageGateway,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify expression UPSERT uses INSERT...SELECT, not pandas materialization."""
    write_gateway.ibis.write(
        "analytics.upsert_test",
        [(1, "original", 1.0)],
        columns=["id", "name", "value"],
    )
    write_gateway.ibis.write(
        "analytics.write_test",
        [(1, "updated", 9.0), (2, "new", 2.0)],
        columns=["id", "name", "value"],
    )

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise AssertionError

    monkeypatch.setattr(it.Table, "to_pandas", _boom, raising=True)

    expr = write_gateway.ibis.table("analytics.write_test").select(["id", "name", "value"])
    result = write_gateway.ibis.write(
        "analytics.upsert_test",
        expr,
        on_conflict=ibis_adapter.OnConflict(conflict_columns=["id"]),
    )
    expect_equal(result.method, "upsert_select")

    rows = write_gateway.con.execute(
        "SELECT id, name, value FROM analytics.upsert_test ORDER BY id"
    ).fetchall()
    expect_equal(rows, [(1, "updated", 9.0), (2, "new", 2.0)])


def test_upsert_insert_new_rows(write_gateway: StorageGateway) -> None:
    """Verify UPSERT inserts new rows when no conflict."""
    rows = [(1, "a", 1.0), (2, "b", 2.0)]

    result = write_gateway.ibis.write(
        "analytics.upsert_test",
        rows,
        columns=["id", "name", "value"],
        on_conflict=ibis_adapter.OnConflict(conflict_columns=["id"]),
    )

    expected_rows = 2
    expect_equal(result.method, "upsert")
    expect_equal(result.rows_affected, expected_rows)

    read_df = write_gateway.ibis.table("analytics.upsert_test").to_pandas()
    expect_equal(len(read_df), expected_rows)


def test_upsert_updates_on_conflict(write_gateway: StorageGateway) -> None:
    """Verify UPSERT updates existing rows on conflict."""
    write_gateway.ibis.write(
        "analytics.upsert_test",
        [(1, "original", 1.0)],
        columns=["id", "name", "value"],
    )

    expected_value = 99.0
    write_gateway.ibis.write(
        "analytics.upsert_test",
        [(1, "updated", expected_value)],
        columns=["id", "name", "value"],
        on_conflict=ibis_adapter.OnConflict(conflict_columns=["id"]),
    )

    read_df = write_gateway.ibis.table("analytics.upsert_test").to_pandas()
    expect_equal(len(read_df), 1)
    expect_equal(read_df.iloc[0]["name"], "updated")
    expect_equal(read_df.iloc[0]["value"], expected_value)


def test_upsert_selective_update_columns(write_gateway: StorageGateway) -> None:
    """Verify UPSERT can selectively update columns."""
    write_gateway.ibis.write(
        "analytics.upsert_test",
        [(1, "original", 1.0)],
        columns=["id", "name", "value"],
    )

    expected_value = 99.0
    write_gateway.ibis.write(
        "analytics.upsert_test",
        [(1, "should_not_update", expected_value)],
        columns=["id", "name", "value"],
        on_conflict=ibis_adapter.OnConflict(conflict_columns=["id"], update_columns=["value"]),
    )

    read_df = write_gateway.ibis.table("analytics.upsert_test").to_pandas()
    expect_equal(len(read_df), 1)
    expect_equal(read_df.iloc[0]["name"], "original")
    expect_equal(read_df.iloc[0]["value"], expected_value)


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
        write_gateway.ibis.write(
            "analytics.write_test",
            cast("Any", "not a valid data type"),
        )
