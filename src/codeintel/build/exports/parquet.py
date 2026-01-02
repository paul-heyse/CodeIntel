"""Parquet exporters for the CodeIntel metadata warehouse."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.build.exports.engine import export_all_datasets
from codeintel.build.exports.engine import (
    export_parquet_for_table as _engine_export_parquet_for_table,
)
from codeintel.core.config.settings import ExportAuditSettings

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.build.exports.common import ExportCallOptions
    from codeintel.core.gateway import BuildGateway


def export_parquet_for_table(
    gateway: BuildGateway,
    table_name: str,
    output_path: Path,
    settings: ExportAuditSettings,
) -> None:
    """Export a single DuckDB table to Parquet.

    Parameters
    ----------
    gateway
        BuildGateway providing the DuckDB connection.
    table_name
        Fully qualified table name (schema.table) to export.
    output_path
        Destination path for the Parquet file.
    settings
        Export audit settings.

    Raises
    ------
    ValueError
        If the requested table is not registered in the dataset mapping.
    """
    registry = gateway.datasets
    if table_name not in registry.by_table_key:
        message = f"Refusing to export unknown dataset table: {table_name}"
        raise ValueError(message)
    _engine_export_parquet_for_table(gateway, table_name, output_path, settings)


def export_dataset_to_parquet(
    gateway: BuildGateway,
    dataset_name: str,
    output_dir: Path,
    *,
    settings: ExportAuditSettings,
) -> Path:
    """Export a dataset resolved through the dataset registry to Parquet.

    Parameters
    ----------
    gateway
        BuildGateway providing the DuckDB connection.
    dataset_name
        Logical dataset name to export (e.g., ``function_types``).
    output_dir
        Destination directory for the Parquet file.
    settings
        Export audit settings.

    Returns
    -------
    Path
        Path to the written Parquet file.

    Raises
    ------
    ValueError
        If the dataset name is unknown.
    """
    registry = gateway.datasets
    parquet_mapping = registry.parquet_datasets
    try:
        table_name = registry.resolve_table_key(dataset_name)
    except KeyError as exc:
        message = f"Unknown dataset: {dataset_name}"
        raise ValueError(message) from exc
    filename = parquet_mapping.get(table_name, f"{dataset_name}.parquet")
    output_path = output_dir / filename
    export_parquet_for_table(gateway, table_name, output_path, settings)
    return output_path


def export_all_parquet(
    gateway: BuildGateway,
    document_output_dir: Path,
    *,
    settings: ExportAuditSettings,
    options: ExportCallOptions | None = None,
) -> None:
    """Export configured datasets to Parquet files under `Document Output/`.

    Parameters
    ----------
    gateway
        BuildGateway providing the DuckDB connection.
    document_output_dir
        Target directory where Parquet artifacts are written.
    settings
        Export audit settings.
    options
        Export options controlling dataset selection and validation.
    """
    _ = export_all_datasets(
        gateway,
        document_output_dir,
        fmt="parquet",
        settings=settings,
        options=options,
    )


__all__ = [
    "export_all_parquet",
    "export_dataset_to_parquet",
    "export_parquet_for_table",
]
