"""Parquet exporters for the CodeIntel metadata warehouse."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.build.exports.engine import export_all_datasets
from codeintel.build.exports.engine import (
    export_parquet_for_table as _engine_export_parquet_for_table,
)

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.build.exports.common import ExportCallOptions
    from codeintel.storage.gateway import StorageGateway


def export_parquet_for_table(
    gateway: StorageGateway,
    table_name: str,
    output_path: Path,
) -> None:
    """Export a single DuckDB table to Parquet.

    Parameters
    ----------
    gateway
        StorageGateway providing the DuckDB connection.
    table_name
        Fully qualified table name (schema.table) to export.
    output_path
        Destination path for the Parquet file.

    Raises
    ------
    ValueError
        If the requested table is not registered in the dataset mapping.
    """
    dataset_mapping = gateway.datasets.mapping
    if table_name not in dataset_mapping.values():
        message = f"Refusing to export unknown dataset table: {table_name}"
        raise ValueError(message)
    _engine_export_parquet_for_table(gateway, table_name, output_path)


def export_dataset_to_parquet(
    gateway: StorageGateway,
    dataset_name: str,
    output_dir: Path,
) -> Path:
    """Export a dataset resolved through the dataset registry to Parquet.

    Parameters
    ----------
    gateway
        StorageGateway providing the DuckDB connection.
    dataset_name
        Logical dataset name to export (e.g., ``function_profile``).
    output_dir
        Destination directory for the Parquet file.

    Returns
    -------
    Path
        Path to the written Parquet file.

    Raises
    ------
    ValueError
        If the dataset name is unknown.
    """
    dataset_mapping = gateway.datasets.mapping
    parquet_mapping = gateway.datasets.parquet_mapping or {}
    if dataset_name not in dataset_mapping:
        message = f"Unknown dataset: {dataset_name}"
        raise ValueError(message)
    table_name = dataset_mapping[dataset_name]
    filename = parquet_mapping.get(table_name, f"{dataset_name}.parquet")
    output_path = output_dir / filename
    export_parquet_for_table(gateway, table_name, output_path)
    return output_path


def export_all_parquet(
    gateway: StorageGateway,
    document_output_dir: Path,
    *,
    options: ExportCallOptions | None = None,
) -> None:
    """Export configured datasets to Parquet files under `Document Output/`.

    Parameters
    ----------
    gateway
        StorageGateway providing the DuckDB connection.
    document_output_dir
        Target directory where Parquet artifacts are written.
    options
        Export options controlling dataset selection and validation.
    """
    _ = export_all_datasets(
        gateway,
        document_output_dir,
        fmt="parquet",
        options=options,
    )


__all__ = [
    "export_all_parquet",
    "export_dataset_to_parquet",
    "export_parquet_for_table",
]
