"""Migration helpers for bootstrapping Iceberg tables from Parquet files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow.dataset as ds
from pyiceberg.io.pyarrow import pyarrow_to_schema
from pyiceberg.partitioning import PartitionSpec
from pyiceberg.table import TableProperties
from pyiceberg.table.sorting import SortOrder

from codeintel.core.iceberg.catalog import IcebergCatalogProvider
from codeintel.core.iceberg.schema import name_mapping_from_arrow_schema
from codeintel.core.iceberg.snapshot_properties import (
    SnapshotPropertyInputs,
    snapshot_properties_for_write,
)
from codeintel.storage.iceberg.cache import refresh_iceberg_metadata_cache
from codeintel.storage.iceberg.statistics_file import persist_iceberg_statistics
from codeintel.storage.iceberg.stats import iceberg_stats_for_table

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pyarrow as pa
    from pyiceberg.table import Table
    from pyiceberg.table.name_mapping import NameMapping

    from codeintel.core.config.settings import IcebergSettings
    from codeintel.storage.gateway.protocol import StorageGateway


@dataclass(frozen=True, slots=True)
class IcebergAddFilesResult:
    """Result payload for adding files to an Iceberg table."""

    table_key: str
    created: bool
    file_count: int
    snapshot_id: int | None


@dataclass(frozen=True, slots=True)
class IcebergAddFilesRequest:
    """Input payload for adding Parquet files to an Iceberg table."""

    table_key: str
    data_dir: Path | None = None
    file_paths: Sequence[Path] | None = None
    snapshot_properties: Mapping[str, str] | None = None
    gateway: StorageGateway | None = None


def add_files_to_iceberg(
    request: IcebergAddFilesRequest,
    *,
    settings: IcebergSettings,
) -> IcebergAddFilesResult:
    """Add Parquet files to an Iceberg table using add_files.

    Returns
    -------
    IcebergAddFilesResult
        Summary of the add_files outcome.
    """
    files = _resolve_parquet_files(
        data_dir=request.data_dir,
        file_paths=request.file_paths,
        table_key=request.table_key,
    )

    dataset = ds.dataset([str(path) for path in files], format="parquet")
    arrow_schema = dataset.schema
    name_mapping = name_mapping_from_arrow_schema(arrow_schema, table_key=request.table_key)
    provider = IcebergCatalogProvider(settings)
    catalog = provider.load()
    identifier = provider.resolve_identifier(request.table_key)
    created = not catalog.table_exists(identifier)
    table = _ensure_table(
        provider=provider,
        identifier=identifier,
        arrow_schema=arrow_schema,
        name_mapping=name_mapping,
    )
    snapshot_properties = snapshot_properties_for_write(
        SnapshotPropertyInputs(table_key=request.table_key)
    )
    snapshot_properties.update(request.snapshot_properties or {})
    table.add_files(
        [str(path) for path in files],
        snapshot_properties=snapshot_properties,
    )
    table.refresh()
    current_snapshot = table.current_snapshot()
    snapshot_id = current_snapshot.snapshot_id if current_snapshot is not None else None
    try:
        iceberg_stats = iceberg_stats_for_table(table, snapshot_id=snapshot_id)
    except (RuntimeError, ValueError, TypeError, OSError):
        iceberg_stats = None
    if iceberg_stats is not None:
        persist_iceberg_statistics(
            table=table,
            table_key=request.table_key,
            stats=iceberg_stats,
            snapshot_properties=snapshot_properties,
        )
    if request.gateway is not None:
        refresh_iceberg_metadata_cache(
            gateway=request.gateway,
            table_key=request.table_key,
            table=table,
        )
    return IcebergAddFilesResult(
        table_key=request.table_key,
        created=created,
        file_count=len(files),
        snapshot_id=snapshot_id,
    )


def _ensure_table(
    *,
    provider: IcebergCatalogProvider,
    identifier: tuple[str, ...],
    arrow_schema: pa.Schema,
    name_mapping: NameMapping | None,
) -> Table:
    catalog = provider.load()
    if catalog.table_exists(identifier):
        return catalog.load_table(identifier)
    iceberg_schema = pyarrow_to_schema(arrow_schema, name_mapping=name_mapping)
    properties = {"write.format.default": "parquet"}
    if name_mapping is not None:
        properties[TableProperties.DEFAULT_NAME_MAPPING] = name_mapping.model_dump_json()
    return catalog.create_table(
        identifier,
        schema=iceberg_schema,
        partition_spec=PartitionSpec(),
        sort_order=SortOrder(order_id=0),
        properties=properties,
    )


def _resolve_parquet_files(
    *,
    data_dir: Path | None,
    file_paths: Sequence[Path] | None,
    table_key: str,
) -> list[Path]:
    if file_paths:
        missing = [path for path in file_paths if not path.is_file()]
        if missing:
            msg = "Missing Parquet files: " + ", ".join(str(path) for path in missing)
            raise FileNotFoundError(msg)
        return sorted({path.resolve() for path in file_paths})
    if data_dir is None:
        msg = f"Provide data_dir or file_paths for {table_key}"
        raise ValueError(msg)
    if data_dir.is_file():
        return [data_dir.resolve()]
    files = sorted(path.resolve() for path in data_dir.rglob("*.parquet"))
    if not files:
        msg = f"No Parquet files found for {table_key}"
        raise ValueError(msg)
    return files


__all__ = ["IcebergAddFilesRequest", "IcebergAddFilesResult", "add_files_to_iceberg"]
