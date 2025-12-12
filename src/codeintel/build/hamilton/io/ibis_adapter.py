"""Ibis-native IO adapters for Hamilton materialization.

These adapters integrate Hamilton's @dataloader/@datasaver pattern with
the existing IbisGateway infrastructure for DuckDB access.

Design Principles
-----------------
1. All DuckDB operations go through IbisGateway (not DuckDBPolicyBackend directly).
2. IbisGateway internally delegates writes to DuckDBPolicyBackend for SQLGlot-based SQL.
3. Reads: IbisGateway.table() / read() / view()
4. Writes: IbisGateway.write() / insert() / upsert()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

    import ibis.expr.types as ir
    import pandas as pd

    from codeintel.build.hamilton.io.dataset_ref import DatasetRef
    from codeintel.storage.gateway import StorageGateway
    from codeintel.storage.ibis_adapter import WriteResult

__all__ = [
    "IbisIOConfig",
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
    """Configuration for Ibis IO operations.

    This config is passed to Hamilton dataloader/datasaver functions
    to provide access to the storage gateway.

    Attributes
    ----------
    gateway
        Storage gateway for database access (use gateway.ibis for operations).
    validate_schema
        Whether to validate against Pandera schema on load/save.

    Examples
    --------
    >>> from codeintel.storage.gateway import StorageGateway
    >>> gateway = StorageGateway(...)
    >>> config = IbisIOConfig(gateway=gateway, validate_schema=True)
    """

    gateway: StorageGateway
    validate_schema: bool = True


def load_ibis_table(
    dataset_ref: DatasetRef,
    io_config: IbisIOConfig,
) -> tuple[ir.Table, dict[str, Any]]:
    """Load a table as an Ibis expression.

    Uses IbisGateway.table() which handles qualified name splitting
    correctly for Ibis 11.

    Parameters
    ----------
    dataset_ref
        Reference to the table to load.
    io_config
        IO configuration with gateway access.

    Returns
    -------
    tuple[ir.Table, dict[str, Any]]
        Ibis table expression and metadata dict.

    Examples
    --------
    >>> ref = DatasetRef(table_key="analytics.function_metrics")
    >>> table, metadata = load_ibis_table(ref, io_config)
    >>> metadata["table_key"]
    'analytics.function_metrics'
    """
    # Use IbisGateway.table() for reads - handles qualified names correctly
    table = io_config.gateway.ibis.table(dataset_ref.table_key)

    metadata: dict[str, Any] = {
        "source": "duckdb",
        "table_key": dataset_ref.table_key,
        "schema": dataset_ref.schema_name,
        "table": dataset_ref.table_name,
    }

    return table, metadata


def load_table_as_dataframe(
    dataset_ref: DatasetRef,
    io_config: IbisIOConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a table as a pandas DataFrame.

    Convenience wrapper that executes the Ibis expression.

    Parameters
    ----------
    dataset_ref
        Reference to the table to load.
    io_config
        IO configuration with gateway access.

    Returns
    -------
    tuple[DataFrame, dict[str, Any]]
        Pandas DataFrame and metadata.

    Examples
    --------
    >>> ref = DatasetRef(table_key="analytics.function_metrics")
    >>> df, metadata = load_table_as_dataframe(ref, io_config)
    >>> metadata["format"]
    'pandas'
    """
    table, metadata = load_ibis_table(dataset_ref, io_config)
    # Ibis execute() returns DataFrame for table expressions
    df = cast("pd.DataFrame", table.execute())
    metadata["format"] = "pandas"
    return df, metadata


def save_ibis_expression(
    output: ir.Table,
    dataset_ref: DatasetRef,
    io_config: IbisIOConfig,
) -> dict[str, Any]:
    """Save an Ibis expression to DuckDB.

    Uses IbisGateway.write() which generates INSERT...SELECT via SQLGlot.

    Parameters
    ----------
    output
        Ibis table expression to save.
    dataset_ref
        Reference specifying where to save.
    io_config
        IO configuration with gateway access.

    Returns
    -------
    dict[str, Any]
        Metadata about the save operation.

    Examples
    --------
    >>> result = save_ibis_expression(ibis_table, ref, io_config)
    >>> result["saved_to"]
    'duckdb'
    """
    # Use IbisGateway.write() for Ibis expression writes
    # This generates INSERT...SELECT via SQLGlot internally
    result: WriteResult = io_config.gateway.ibis.write(
        dataset_ref.table_key,
        output,
    )

    return {
        "saved_to": "duckdb",
        "table_key": dataset_ref.table_key,
        "row_count": result.rows_affected,
        "method": result.method,
    }


def save_dataframe(
    df: pd.DataFrame,
    dataset_ref: DatasetRef,
    io_config: IbisIOConfig,
) -> dict[str, Any]:
    """Save a pandas DataFrame to DuckDB.

    Uses IbisGateway.write() which internally uses DuckDBPolicyBackend
    for efficient INSERT...VALUES via SQLGlot.

    Parameters
    ----------
    df
        DataFrame to save.
    dataset_ref
        Target table reference.
    io_config
        IO configuration.

    Returns
    -------
    dict[str, Any]
        Write operation metadata.

    Examples
    --------
    >>> result = save_dataframe(df, ref, io_config)
    >>> result["operation"]
    'insert_values'
    """
    # IbisGateway.write() accepts DataFrames directly
    # Internally uses DuckDBPolicyBackend.bulk_insert()
    result: WriteResult = io_config.gateway.ibis.write(
        dataset_ref.table_key,
        df,
    )

    return {
        "operation": "insert_values",
        "table_key": dataset_ref.table_key,
        "row_count": result.rows_affected,
        "method": result.method,
    }


def save_rows(
    rows: Sequence[tuple[object, ...]],
    columns: Sequence[str],
    dataset_ref: DatasetRef,
    io_config: IbisIOConfig,
) -> dict[str, Any]:
    """Save row tuples to DuckDB.

    Uses IbisGateway.write() which internally uses DuckDBPolicyBackend
    for efficient INSERT...VALUES via SQLGlot.

    Parameters
    ----------
    rows
        Sequence of row tuples.
    columns
        Column names matching row tuple positions.
    dataset_ref
        Target table reference.
    io_config
        IO configuration.

    Returns
    -------
    dict[str, Any]
        Write operation metadata.

    Examples
    --------
    >>> rows = [("goid1", 100), ("goid2", 200)]
    >>> columns = ["goid", "loc"]
    >>> result = save_rows(rows, columns, ref, io_config)
    >>> result["operation"]
    'insert_values'
    """
    # IbisGateway.write() accepts tuples directly
    # Internally uses DuckDBPolicyBackend.bulk_insert()
    result: WriteResult = io_config.gateway.ibis.write(
        dataset_ref.table_key,
        rows,
        columns=columns,
    )

    return {
        "operation": "insert_values",
        "table_key": dataset_ref.table_key,
        "row_count": result.rows_affected,
        "method": result.method,
    }


def upsert_dataframe(
    df: pd.DataFrame,
    dataset_ref: DatasetRef,
    conflict_columns: Sequence[str],
    update_columns: Sequence[str],
    io_config: IbisIOConfig,
) -> dict[str, Any]:
    """Upsert a DataFrame using INSERT...ON CONFLICT.

    Uses IbisGateway.upsert() which internally uses DuckDBPolicyBackend
    for SQLGlot-based UPSERT generation.

    Parameters
    ----------
    df
        DataFrame to upsert.
    dataset_ref
        Target table reference.
    conflict_columns
        Columns defining uniqueness constraint.
    update_columns
        Columns to update on conflict.
    io_config
        IO configuration.

    Returns
    -------
    dict[str, Any]
        Upsert operation metadata.

    Examples
    --------
    >>> result = upsert_dataframe(
    ...     df,
    ...     ref,
    ...     conflict_columns=["goid"],
    ...     update_columns=["loc"],
    ...     io_config=io_config,
    ... )
    >>> result["operation"]
    'upsert'
    """
    # IbisGateway.upsert() handles ON CONFLICT semantics
    # Internally uses DuckDBPolicyBackend.upsert()
    result: WriteResult = io_config.gateway.ibis.upsert(
        dataset_ref.table_key,
        df,
        columns=list(df.columns),
        conflict_columns=conflict_columns,
        update_columns=update_columns,
    )

    return {
        "operation": "upsert",
        "table_key": dataset_ref.table_key,
        "row_count": result.rows_affected,
        "method": result.method,
    }
