"""Helpers for reading CodeIntel metadata from Parquet datasets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from codeintel.core.columnar.schema_metadata import decode_metadata
from codeintel.core.schemas.primitives import Column, ColumnType, TableSchema, normalize_column_type
from codeintel.core.table_key import split_table_key

_METADATA_FILENAME = "_metadata"
_COMMON_METADATA_FILENAME = "_common_metadata"


def _read_parquet_file(path: Path) -> pq.ParquetFile | None:
    if not path.is_file():
        return None
    try:
        return pq.ParquetFile(path)
    except (OSError, ValueError, pa.ArrowInvalid):
        return None


def read_parquet_metadata(path: Path) -> pq.FileMetaData | None:
    """Return Parquet metadata for a file when available.

    Returns
    -------
    pyarrow.parquet.FileMetaData | None
        Parquet metadata when present, otherwise None.
    """
    parquet_file = _read_parquet_file(path)
    if parquet_file is None:
        return None
    return parquet_file.metadata


def read_parquet_schema(path: Path) -> pa.Schema | None:
    """Return Arrow schema for a Parquet file when available.

    Returns
    -------
    pyarrow.Schema | None
        Arrow schema when present, otherwise None.
    """
    parquet_file = _read_parquet_file(path)
    if parquet_file is None:
        return None
    return parquet_file.schema_arrow


def metadata_from_schema(schema: pa.Schema) -> dict[str, object]:
    """Return decoded metadata from a Parquet-backed Arrow schema.

    Returns
    -------
    dict[str, object]
        Decoded metadata mapping (empty when metadata is missing).
    """
    return decode_metadata(schema.metadata)


def table_schema_from_parquet_metadata(schema: pa.Schema) -> TableSchema | None:
    """Build a TableSchema from Parquet metadata when available.

    Returns
    -------
    TableSchema | None
        Parsed TableSchema, or None when metadata is missing.
    """
    metadata = metadata_from_schema(schema)
    table_key = metadata.get("codeintel.table_key")
    if not isinstance(table_key, str) or not table_key:
        return None
    columns_obj = metadata.get("codeintel.columns_json")
    if not isinstance(columns_obj, Mapping):
        return None
    nullability_obj = metadata.get("codeintel.nullability_json")
    nullability_map: dict[str, bool] = {}
    if isinstance(nullability_obj, Mapping):
        nullability_map = {str(name): bool(value) for name, value in nullability_obj.items()}
    primary_obj = metadata.get("codeintel.primary_keys_json")
    primary_key: tuple[str, ...] = ()
    if isinstance(primary_obj, list):
        primary_key = tuple(str(item) for item in primary_obj)
    schema_name, table_name = split_table_key(table_key)
    columns = _columns_from_mapping(columns_obj, nullability_map)
    return TableSchema(
        schema=schema_name,
        name=table_name,
        columns=columns,
        primary_key=primary_key,
    )


def table_schema_from_dataset(dataset: ds.Dataset) -> TableSchema | None:
    """Return TableSchema derived from a Parquet dataset schema.

    Returns
    -------
    TableSchema | None
        Parsed TableSchema, or None when metadata is missing.
    """
    return table_schema_from_parquet_metadata(dataset.schema)


def column_types_from_metadata(schema: pa.Schema) -> dict[str, ColumnType] | None:
    """Return column type mapping from Parquet metadata when available.

    Returns
    -------
    dict[str, ColumnType] | None
        Column type mapping, or None when metadata is missing.
    """
    table_schema = table_schema_from_parquet_metadata(schema)
    if table_schema is None:
        return None
    return {column.name: column.type for column in table_schema.columns}


@dataclass(frozen=True, slots=True)
class DatasetMetadataContext:
    """Context for reading dataset-level Parquet metadata."""

    dataset_root: Path
    table_key: str

    def metadata_path(self) -> Path | None:
        """Return the dataset-level metadata path when present.

        Returns
        -------
        pathlib.Path | None
            Metadata path when the dataset metadata file exists.
        """
        metadata_path = self.dataset_root / _METADATA_FILENAME
        return metadata_path if metadata_path.is_file() else None

    def common_metadata_path(self) -> Path | None:
        """Return the dataset-level common metadata path when present.

        Returns
        -------
        pathlib.Path | None
            Common metadata path when the dataset common metadata file exists.
        """
        metadata_path = self.dataset_root / _COMMON_METADATA_FILENAME
        return metadata_path if metadata_path.is_file() else None

    def read_metadata(self) -> pq.FileMetaData | None:
        """Return Parquet metadata for the dataset when available.

        Returns
        -------
        pyarrow.parquet.FileMetaData | None
            Parquet metadata when present, otherwise None.
        """
        metadata_path = self.metadata_path()
        if metadata_path is None:
            return None
        return read_parquet_metadata(metadata_path)

    def read_common_metadata(self) -> pq.FileMetaData | None:
        """Return Parquet common metadata for the dataset when available.

        Returns
        -------
        pyarrow.parquet.FileMetaData | None
            Parquet metadata when present, otherwise None.
        """
        metadata_path = self.common_metadata_path()
        if metadata_path is None:
            return None
        return read_parquet_metadata(metadata_path)

    def read_schema(self) -> pa.Schema | None:
        """Return Arrow schema for the dataset when available.

        Returns
        -------
        pyarrow.Schema | None
            Arrow schema when present, otherwise None.
        """
        metadata_path = self.common_metadata_path()
        if metadata_path is None:
            return None
        return read_parquet_schema(metadata_path)


def _columns_from_mapping(
    columns_obj: Mapping[str, object],
    nullability_map: Mapping[str, bool],
) -> list[Column]:
    columns: list[Column] = []
    for name in sorted(columns_obj):
        raw_type = columns_obj.get(name)
        if not isinstance(raw_type, str):
            continue
        try:
            col_type = normalize_column_type(raw_type)
        except ValueError:
            continue
        columns.append(
            Column(
                name=str(name),
                type=col_type,
                nullable=nullability_map.get(str(name), True),
            )
        )
    return columns


__all__ = [
    "DatasetMetadataContext",
    "column_types_from_metadata",
    "metadata_from_schema",
    "read_parquet_metadata",
    "read_parquet_schema",
    "table_schema_from_dataset",
    "table_schema_from_parquet_metadata",
]
