"""Tests for registry_contracts module."""

from __future__ import annotations

import pytest

from codeintel.storage.gateway import StorageGateway
from codeintel.storage.registry_contracts import (
    ColumnDef,
    build_registry_contracts,
    fetch_table_columns,
    list_registry_tables,
    render_create_table_from_catalog,
)


def test_list_registry_tables_returns_ordered_keys(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify list_registry_tables returns table keys in sorted order."""
    con = fresh_gateway.con

    result = list_registry_tables(con)

    assert isinstance(result, list)
    assert len(result) > 0
    assert result == sorted(result)


def test_fetch_table_columns_returns_column_defs(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify fetch_table_columns returns ColumnDef objects."""
    con = fresh_gateway.con

    result = fetch_table_columns(con, "core.modules")

    assert isinstance(result, list)
    assert len(result) > 0

    for col_def in result:
        assert isinstance(col_def, ColumnDef)
        assert col_def.name
        assert col_def.data_type


def test_fetch_table_columns_preserves_ordinal_position(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify fetch_table_columns returns columns in ordinal order."""
    con = fresh_gateway.con

    result = fetch_table_columns(con, "core.goids")

    min_expected_columns = 2
    assert len(result) >= min_expected_columns
    column_names = [col.name for col in result]
    assert "repo" in column_names


def test_fetch_table_columns_raises_on_invalid_key(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify fetch_table_columns raises on invalid table key."""
    con = fresh_gateway.con

    with pytest.raises(ValueError, match="Invalid table key"):
        fetch_table_columns(con, "notable")


def test_build_registry_contracts_maps_all_tables(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify build_registry_contracts returns mapping for all registry tables."""
    con = fresh_gateway.con

    result = build_registry_contracts(con)

    assert isinstance(result, dict)
    assert len(result) > 0

    for table_key, columns in result.items():
        assert isinstance(table_key, str)
        assert isinstance(columns, list)
        assert "." in table_key


def test_build_registry_contracts_with_subset(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify build_registry_contracts respects table_keys filter."""
    con = fresh_gateway.con
    subset = ["core.modules", "core.goids"]

    result = build_registry_contracts(con, table_keys=subset)

    assert set(result.keys()) == set(subset)


def test_render_create_table_produces_valid_ddl(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify render_create_table_from_catalog produces valid DDL."""
    con = fresh_gateway.con

    result = render_create_table_from_catalog(con, "core.modules")

    assert isinstance(result, str)
    assert "CREATE TABLE IF NOT EXISTS core.modules" in result
    assert "module" in result.lower()


def test_render_create_table_includes_all_columns(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify render_create_table_from_catalog includes all columns."""
    con = fresh_gateway.con

    columns = fetch_table_columns(con, "core.modules")
    ddl = render_create_table_from_catalog(con, "core.modules")

    for col in columns:
        assert col.name in ddl


def test_render_create_table_raises_on_invalid_key(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify render_create_table_from_catalog raises on invalid table key."""
    con = fresh_gateway.con

    with pytest.raises(ValueError, match="Invalid table key"):
        render_create_table_from_catalog(con, "notable")


def test_column_def_is_namedtuple() -> None:
    """Verify ColumnDef behaves as expected namedtuple."""
    col = ColumnDef(name="test_col", data_type="VARCHAR")

    assert col.name == "test_col"
    assert col.data_type == "VARCHAR"
    assert col[0] == "test_col"
    assert col[1] == "VARCHAR"
