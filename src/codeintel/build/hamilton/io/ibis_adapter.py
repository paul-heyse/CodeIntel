"""Ibis-native IO adapters for Hamilton materialization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from codeintel.build.hamilton.io.dataset_ref import DatasetRef
from codeintel.storage.io.ibis_io import (
    IbisIOConfig,
)
from codeintel.storage.io.ibis_io import (
    load_dataset_df as _load_dataset_df,
)
from codeintel.storage.io.ibis_io import (
    load_dataset_ibis as _load_dataset_ibis,
)
from codeintel.storage.io.ibis_io import (
    load_ibis_table as _load_ibis_table,
)
from codeintel.storage.io.ibis_io import (
    load_table_as_dataframe as _load_table_as_dataframe,
)
from codeintel.storage.io.ibis_io import (
    save_dataframe as _save_dataframe,
)
from codeintel.storage.io.ibis_io import (
    save_ibis_expression as _save_ibis_expression,
)
from codeintel.storage.io.ibis_io import (
    save_rows as _save_rows,
)
from codeintel.storage.io.ibis_io import (
    upsert_dataframe as _upsert_dataframe,
)

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


def load_ibis_table(
    dataset_ref: DatasetRef,
    io_config: IbisIOConfig,
) -> tuple[ir.Table, dict[str, Any]]:
    """Load a table as an Ibis expression.

    Returns
    -------
    tuple[ir.Table, dict[str, Any]]
        Ibis table expression and metadata.
    """
    return _load_ibis_table(dataset_ref.table_key, io_config)


def load_table_as_dataframe(
    dataset_ref: DatasetRef,
    io_config: IbisIOConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a table as a pandas DataFrame.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        DataFrame and metadata for the dataset.
    """
    return _load_table_as_dataframe(dataset_ref.table_key, io_config)


def save_ibis_expression(
    output: ir.Table,
    dataset_ref: DatasetRef,
    io_config: IbisIOConfig,
) -> dict[str, Any]:
    """Save an Ibis expression to DuckDB.

    Returns
    -------
    dict[str, Any]
        Write metadata.
    """
    return _save_ibis_expression(output, dataset_ref.table_key, io_config)


def save_dataframe(
    df: pd.DataFrame,
    dataset_ref: DatasetRef,
    io_config: IbisIOConfig,
) -> dict[str, Any]:
    """Save a pandas DataFrame to DuckDB.

    Returns
    -------
    dict[str, Any]
        Write metadata.
    """
    return _save_dataframe(df, dataset_ref.table_key, io_config)


def save_rows(
    rows: Sequence[tuple[object, ...]],
    columns: Sequence[str],
    dataset_ref: DatasetRef,
    io_config: IbisIOConfig,
) -> dict[str, Any]:
    """Save row tuples to DuckDB.

    Returns
    -------
    dict[str, Any]
        Write metadata.
    """
    return _save_rows(rows, columns, dataset_ref.table_key, io_config)


def upsert_dataframe(
    df: pd.DataFrame,
    dataset_ref: DatasetRef,
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
    return _upsert_dataframe(
        df,
        dataset_ref.table_key,
        conflict_columns,
        update_columns,
        io_config,
    )


def load_dataset_ibis(
    *,
    gateway: StorageGateway,
    ref: DatasetRef,
) -> ir.Table:
    """Load a dataset as an Ibis expression with repo/commit filtering.

    Returns
    -------
    ir.Table
        Ibis table expression for the dataset.
    """
    return _load_dataset_ibis(
        gateway=gateway,
        table_key=ref.table_key,
        repo=ref.repo,
        commit=ref.commit,
    )


def load_dataset_df(
    *,
    gateway: StorageGateway,
    ref: DatasetRef,
) -> pd.DataFrame:
    """Load a dataset as a pandas DataFrame with repo/commit filtering.

    Returns
    -------
    pd.DataFrame
        DataFrame for the dataset.
    """
    return _load_dataset_df(
        gateway=gateway,
        table_key=ref.table_key,
        repo=ref.repo,
        commit=ref.commit,
    )
