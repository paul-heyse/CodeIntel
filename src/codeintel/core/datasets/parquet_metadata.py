"""Helpers for reading CodeIntel metadata from Parquet datasets."""

from __future__ import annotations

from collections.abc import Mapping

import pyarrow as pa
import pyarrow.dataset as ds

from codeintel.core.columnar.schema_metadata import decode_metadata
from codeintel.core.schemas.primitives import Column, ColumnType, TableSchema, normalize_column_type
from codeintel.core.table_key import split_table_key


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
    "column_types_from_metadata",
    "metadata_from_schema",
    "table_schema_from_dataset",
    "table_schema_from_parquet_metadata",
]
