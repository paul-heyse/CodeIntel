"""Ibis-native IO helpers backed by the storage gateway."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from codeintel.storage.gateway import ibis_facade
from codeintel.storage.helpers.table_key import split_table_key
from codeintel.storage.warehouse import MaterializeOptions, UpsertConfig, Warehouse

if TYPE_CHECKING:
    from collections.abc import Sequence

    import ibis.expr.types as ir
    import pandas as pd

    from codeintel.storage.gateway import StorageGateway

__all__ = [
    "IbisIOConfig",
    "load_dataset_df",
    "load_dataset_ibis",
    "load_ibis_table",
    "load_table_as_dataframe",
    "save_dataframe",
    "save_ibis_expression",
    "save_rows",
    "upsert_dataframe",
]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class IbisIOConfig:
    """Configuration for Ibis IO operations."""

    gateway: StorageGateway
    validate_schema: bool = True


def load_ibis_table(
    table_key: str,
    io_config: IbisIOConfig,
) -> tuple[ir.Table, dict[str, Any]]:
    """Load a table as an Ibis expression.

    Returns
    -------
    tuple[ir.Table, dict[str, Any]]
        Ibis table expression and metadata.
    """
    table = ibis_facade.table(io_config.gateway, table_key)
    schema_name, table_name = split_table_key(table_key)
    metadata: dict[str, Any] = {
        "source": "duckdb",
        "table_key": table_key,
        "schema": schema_name,
        "table": table_name,
    }
    return table, metadata


def load_table_as_dataframe(
    table_key: str,
    io_config: IbisIOConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a table as a pandas DataFrame.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        DataFrame and metadata for the dataset.
    """
    table, metadata = load_ibis_table(table_key, io_config)
    df = cast("pd.DataFrame", table.execute())
    metadata["format"] = "pandas"
    return df, metadata


def save_ibis_expression(
    output: ir.Table,
    table_key: str,
    io_config: IbisIOConfig,
) -> dict[str, Any]:
    """Save an Ibis expression to DuckDB.

    Returns
    -------
    dict[str, Any]
        Write metadata.
    """
    result = Warehouse(io_config.gateway).materialize_ibis(
        table_key,
        output,
        options=MaterializeOptions(mode="append"),
    )
    return {
        "saved_to": "duckdb",
        "table_key": table_key,
        "row_count": result.rows_written,
        "method": "warehouse_materialize_table",
    }


def save_dataframe(
    df: pd.DataFrame,
    table_key: str,
    io_config: IbisIOConfig,
) -> dict[str, Any]:
    """Save a pandas DataFrame to DuckDB.

    Returns
    -------
    dict[str, Any]
        Write metadata.
    """
    result = Warehouse(io_config.gateway).materialize_dataframe(
        table_key,
        df,
        options=MaterializeOptions(mode="append"),
    )
    return {
        "operation": "insert_values",
        "table_key": table_key,
        "row_count": result.rows_written,
        "method": "warehouse_materialize_dataframe",
    }


def save_rows(
    rows: Sequence[tuple[object, ...]],
    columns: Sequence[str],
    table_key: str,
    io_config: IbisIOConfig,
) -> dict[str, Any]:
    """Save row tuples to DuckDB.

    Returns
    -------
    dict[str, Any]
        Write metadata.
    """
    result = Warehouse(io_config.gateway).materialize_rows(
        table_key,
        rows,
        columns=columns,
        options=MaterializeOptions(mode="append"),
    )
    return {
        "operation": "insert_values",
        "table_key": table_key,
        "row_count": result.rows_written,
        "method": "warehouse_materialize_rows",
    }


def upsert_dataframe(
    df: pd.DataFrame,
    table_key: str,
    conflict_columns: Sequence[str],
    update_columns: Sequence[str],
    io_config: IbisIOConfig,
) -> dict[str, Any]:
    """Upsert a DataFrame using INSERT...ON CONFLICT.

    Returns
    -------
    dict[str, Any]
        Write metadata.
    """
    result = Warehouse(io_config.gateway).materialize_dataframe(
        table_key,
        df,
        options=MaterializeOptions(
            mode="upsert",
            upsert=UpsertConfig(
                conflict_columns=tuple(conflict_columns),
                update_columns=tuple(update_columns),
            ),
        ),
    )
    return {
        "operation": "upsert",
        "table_key": table_key,
        "row_count": result.rows_written,
        "method": "warehouse_materialize_dataframe",
    }


def load_dataset_ibis(
    *,
    gateway: StorageGateway,
    table_key: str,
    repo: str | None,
    commit: str | None,
) -> ir.Table:
    """Load a dataset as an Ibis expression with repo/commit filtering.

    Returns
    -------
    ir.Table
        Ibis table expression for the dataset.
    """
    table = ibis_facade.table(gateway, table_key)
    cols = set(table.columns)
    if repo and commit and "repo" in cols and "commit" in cols:
        predicate = cast("ir.BooleanValue", (table.repo == repo) & (table.commit == commit))
        table = table.filter(predicate)
    return table


def load_dataset_df(
    *,
    gateway: StorageGateway,
    table_key: str,
    repo: str | None,
    commit: str | None,
) -> pd.DataFrame:
    """Load a dataset as a pandas DataFrame with repo/commit filtering.

    Returns
    -------
    pd.DataFrame
        DataFrame for the dataset.
    """
    table = load_dataset_ibis(gateway=gateway, table_key=table_key, repo=repo, commit=commit)
    return cast("pd.DataFrame", table.execute())
