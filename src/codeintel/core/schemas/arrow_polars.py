"""Arrow/Polars schema conversion helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

import polars as pl
import pyarrow as pa

from codeintel.core.columnar.schema_metadata import decode_metadata
from codeintel.core.schemas.arrow_gen import ARROW_SCHEMA_CONTRACT_VERSION, EXTRAS_POLICIES
from codeintel.core.schemas.primitives import Column, ColumnType, TableSchema, normalize_column_type
from codeintel.storage.helpers.table_key import split_table_key, validate_table_key


def table_schema_from_arrow_schema(
    *,
    arrow_schema: pa.Schema,
    table_key: str | None = None,
) -> TableSchema:
    """Convert a PyArrow schema into a TableSchema.

    Parameters
    ----------
    arrow_schema
        PyArrow schema to convert.
    table_key
        Optional table key override. When omitted, use `codeintel.table_key`
        metadata from the Arrow schema.

    Returns
    -------
    TableSchema
        TableSchema derived from the Arrow schema.
    """
    metadata = decode_metadata(arrow_schema.metadata)
    _validate_contract_metadata(metadata)
    resolved_key = _resolve_table_key(table_key=table_key, metadata=metadata)
    schema_name, table_name = split_table_key(resolved_key)
    table_description = _metadata_str(metadata, "codeintel.description")
    primary_key = _primary_key_from_metadata(metadata)
    columns = [_column_from_field(field) for field in arrow_schema]
    if not primary_key:
        primary_key = _primary_key_from_fields(arrow_schema)
    return TableSchema(
        schema=schema_name,
        name=table_name,
        columns=columns,
        primary_key=primary_key,
        description=table_description,
    )


def table_schema_from_polars_schema(*, polars_schema: pl.Schema, table_key: str) -> TableSchema:
    """Convert a Polars schema into a TableSchema.

    Parameters
    ----------
    polars_schema
        Polars schema to convert.
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    TableSchema
        TableSchema derived from the Polars schema.
    """
    return table_schema_from_arrow_schema(
        arrow_schema=polars_schema.to_arrow(),
        table_key=table_key,
    )


def table_schema_from_polars_dataframe(*, frame: pl.DataFrame, table_key: str) -> TableSchema:
    """Convert a Polars DataFrame into a TableSchema.

    Parameters
    ----------
    frame
        Polars DataFrame to derive the schema from.
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    TableSchema
        TableSchema derived from the DataFrame schema.
    """
    return table_schema_from_polars_schema(polars_schema=frame.schema, table_key=table_key)


def table_schema_from_polars_lazyframe(*, frame: pl.LazyFrame, table_key: str) -> TableSchema:
    """Convert a Polars LazyFrame into a TableSchema.

    Parameters
    ----------
    frame
        Polars LazyFrame to derive the schema from.
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    TableSchema
        TableSchema derived from the LazyFrame schema.
    """
    return table_schema_from_polars_schema(
        polars_schema=frame.collect_schema(),
        table_key=table_key,
    )


def _resolve_table_key(*, table_key: str | None, metadata: Mapping[str, object]) -> str:
    meta_value = metadata.get("codeintel.table_key")
    resolved = table_key
    if meta_value is not None:
        if not isinstance(meta_value, str):
            msg = (
                "Arrow schema metadata codeintel.table_key must be a string, "
                f"got {type(meta_value)}"
            )
            raise TypeError(msg)
        if resolved is None:
            resolved = meta_value
        elif resolved != meta_value:
            msg = f"Arrow schema table_key mismatch: {resolved!r} != {meta_value!r}"
            raise ValueError(msg)
    if resolved is None:
        msg = "table_key is required when Arrow schema metadata lacks codeintel.table_key"
        raise ValueError(msg)
    validate_table_key(resolved)
    return resolved


def _column_from_field(field: pa.Field) -> Column:
    metadata = decode_metadata(field.metadata)
    column_type = _column_type_from_metadata(metadata)
    if column_type is None:
        column_type = _column_type_from_arrow_type(field.type)
    description = _metadata_str(metadata, "codeintel.description")
    return Column(
        name=field.name,
        type=column_type,
        nullable=field.nullable,
        description=description,
    )


def _primary_key_from_metadata(metadata: Mapping[str, object]) -> tuple[str, ...]:
    raw = metadata.get("codeintel.primary_key")
    if raw is None:
        return ()
    if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
        return tuple(raw)
    if isinstance(raw, str):
        return (raw,)
    msg = f"Arrow schema metadata codeintel.primary_key must be a list of strings, got {type(raw)}"
    raise TypeError(msg)


def _primary_key_from_fields(schema: pa.Schema) -> tuple[str, ...]:
    primary: list[str] = []
    for field in schema:
        metadata = decode_metadata(field.metadata)
        role = metadata.get("codeintel.key_role")
        if role == "primary_key":
            primary.append(field.name)
    return tuple(primary)


def _metadata_str(metadata: Mapping[str, object], key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    msg = f"Arrow metadata {key} must be a string, got {type(value)}"
    raise TypeError(msg)


def _column_type_from_metadata(metadata: Mapping[str, object]) -> ColumnType | None:
    raw = metadata.get("codeintel.column_type")
    if raw is None:
        return None
    if not isinstance(raw, str):
        msg = f"Arrow metadata codeintel.column_type must be a string, got {type(raw)}"
        raise TypeError(msg)
    try:
        return normalize_column_type(raw)
    except ValueError as exc:
        msg = f"Arrow metadata codeintel.column_type is not supported: {raw!r}"
        raise ValueError(msg) from exc


def _column_type_from_arrow_type(dtype: pa.DataType) -> ColumnType:
    """Map Arrow data types to ColumnType.

    Mapping highlights:
    - bool -> BOOLEAN
    - int8/16/32/uint8/16/32 -> INTEGER
    - int64 -> BIGINT
    - uint64 -> DECIMAL(38,0)
    - float -> DOUBLE
    - decimal -> DECIMAL (DECIMAL(38,0) when precision=38 and scale=0)
    - timestamp -> TIMESTAMP/TIMESTAMPTZ
    - date/time/duration -> TIMESTAMP
    - string/string_view -> VARCHAR
    - binary/binary_view -> VARCHAR
    - list/struct/map/union -> LIST/STRUCT/MAP/UNION types

    Returns
    -------
    ColumnType
        ColumnType corresponding to the Arrow type.

    Raises
    ------
    ValueError
        If the Arrow type cannot be mapped.
    """
    for resolver in _ARROW_TYPE_RESOLVERS:
        resolved = resolver(dtype)
        if resolved is not None:
            return resolved
    msg = f"Unsupported Arrow type for TableSchema: {dtype}"
    raise ValueError(msg)


def _boolean_column_type(dtype: pa.DataType) -> ColumnType | None:
    return "BOOLEAN" if pa.types.is_boolean(dtype) else None


def _integer_column_type(dtype: pa.DataType) -> ColumnType | None:
    if not pa.types.is_integer(dtype):
        return None
    if pa.types.is_int64(dtype):
        return "BIGINT"
    if pa.types.is_uint64(dtype):
        return "DECIMAL(38,0)"
    return "INTEGER"


def _floating_column_type(dtype: pa.DataType) -> ColumnType | None:
    return "DOUBLE" if pa.types.is_floating(dtype) else None


def _decimal_column_type(dtype: pa.DataType) -> ColumnType | None:
    if not pa.types.is_decimal(dtype):
        return None
    precision = getattr(dtype, "precision", None)
    scale = getattr(dtype, "scale", None)
    if precision is None or scale is None:
        return "DECIMAL"
    return normalize_column_type(f"DECIMAL({precision},{scale})")


def _timestamp_column_type(dtype: pa.DataType) -> ColumnType | None:
    if not pa.types.is_timestamp(dtype):
        return None
    return "TIMESTAMPTZ" if dtype.tz else "TIMESTAMP"


def _temporal_column_type(dtype: pa.DataType) -> ColumnType | None:
    if not _is_temporal_type(dtype):
        return None
    return "TIMESTAMP"


def _string_column_type(dtype: pa.DataType) -> ColumnType | None:
    if not _is_string_type(dtype):
        return None
    return "VARCHAR"


def _binary_column_type(dtype: pa.DataType) -> ColumnType | None:
    if not _is_binary_type(dtype):
        return None
    return "VARCHAR"


def _dictionary_column_type(dtype: pa.DataType) -> ColumnType | None:
    if not pa.types.is_dictionary(dtype):
        return None
    return _column_type_from_arrow_type(dtype.value_type)


def _struct_column_type(dtype: pa.DataType) -> ColumnType | None:
    if not pa.types.is_struct(dtype):
        return None
    struct_type = cast("pa.StructType", dtype)
    parts = [f"{field.name} {_column_type_from_arrow_type(field.type)}" for field in struct_type]
    return normalize_column_type(f"STRUCT({', '.join(parts)})")


def _list_column_type(dtype: pa.DataType) -> ColumnType | None:
    if not _is_list_type(dtype):
        return None
    value_type = getattr(dtype, "value_type", None)
    if not isinstance(value_type, pa.DataType):
        msg = f"List type missing value_type: {dtype}"
        raise TypeError(msg)
    inner = _column_type_from_arrow_type(value_type)
    return normalize_column_type(f"LIST({inner})")


def _map_column_type(dtype: pa.DataType) -> ColumnType | None:
    if not pa.types.is_map(dtype):
        return None
    map_type = cast("pa.MapType", dtype)
    key_type = _column_type_from_arrow_type(map_type.key_type)
    value_type = _column_type_from_arrow_type(map_type.item_type)
    return normalize_column_type(f"MAP({key_type}, {value_type})")


def _union_column_type(dtype: pa.DataType) -> ColumnType | None:
    if not pa.types.is_union(dtype):
        return None
    union_type = cast("pa.UnionType", dtype)
    parts = [f"{field.name} {_column_type_from_arrow_type(field.type)}" for field in union_type]
    return normalize_column_type(f"UNION({', '.join(parts)})")


def _null_column_type(dtype: pa.DataType) -> ColumnType | None:
    return "VARCHAR" if pa.types.is_null(dtype) else None


def _is_temporal_type(dtype: pa.DataType) -> bool:
    checks = (pa.types.is_date, pa.types.is_time, pa.types.is_duration)
    return any(check(dtype) for check in checks)


def _is_string_type(dtype: pa.DataType) -> bool:
    checks = (pa.types.is_string, pa.types.is_large_string, pa.types.is_string_view)
    return any(check(dtype) for check in checks)


def _is_binary_type(dtype: pa.DataType) -> bool:
    checks = (
        pa.types.is_binary,
        pa.types.is_large_binary,
        pa.types.is_fixed_size_binary,
        pa.types.is_binary_view,
    )
    return any(check(dtype) for check in checks)


def _is_list_type(dtype: pa.DataType) -> bool:
    checks = [
        pa.types.is_list,
        pa.types.is_large_list,
        pa.types.is_fixed_size_list,
    ]
    list_view = getattr(pa.types, "is_list_view", None)
    if callable(list_view):
        checks.append(list_view)
    large_list_view = getattr(pa.types, "is_large_list_view", None)
    if callable(large_list_view):
        checks.append(large_list_view)
    return any(check(dtype) for check in checks)


_ARROW_TYPE_RESOLVERS: tuple[
    Callable[[pa.DataType], ColumnType | None],
    ...,
] = (
    _boolean_column_type,
    _integer_column_type,
    _floating_column_type,
    _decimal_column_type,
    _timestamp_column_type,
    _temporal_column_type,
    _string_column_type,
    _binary_column_type,
    _dictionary_column_type,
    _struct_column_type,
    _list_column_type,
    _map_column_type,
    _union_column_type,
    _null_column_type,
)


def _validate_contract_metadata(metadata: Mapping[str, object]) -> None:
    version = metadata.get("codeintel.schema_contract_version")
    if version is not None:
        if not isinstance(version, str):
            msg = (
                "Arrow schema metadata codeintel.schema_contract_version must be a string, "
                f"got {type(version)}"
            )
            raise TypeError(msg)
        if version != ARROW_SCHEMA_CONTRACT_VERSION:
            msg = (
                "Arrow schema contract version mismatch: "
                f"{version!r} != {ARROW_SCHEMA_CONTRACT_VERSION!r}"
            )
            raise ValueError(msg)
    extras_policy = metadata.get("codeintel.extras_policy")
    if extras_policy is not None:
        if not isinstance(extras_policy, str):
            msg = (
                "Arrow schema metadata codeintel.extras_policy must be a string, "
                f"got {type(extras_policy)}"
            )
            raise TypeError(msg)
        if extras_policy not in EXTRAS_POLICIES:
            msg = f"Arrow schema extras_policy is not supported: {extras_policy!r}"
            raise ValueError(msg)
    extras_column = metadata.get("codeintel.extras_column")
    if extras_column is not None and not isinstance(extras_column, str):
        msg = (
            "Arrow schema metadata codeintel.extras_column must be a string, "
            f"got {type(extras_column)}"
        )
        raise TypeError(msg)
    extras_schema = metadata.get("codeintel.extras_schema")
    if extras_schema is not None and not isinstance(extras_schema, Mapping):
        msg = (
            "Arrow schema metadata codeintel.extras_schema must be a mapping, "
            f"got {type(extras_schema)}"
        )
        raise TypeError(msg)


__all__ = [
    "table_schema_from_arrow_schema",
    "table_schema_from_polars_dataframe",
    "table_schema_from_polars_lazyframe",
    "table_schema_from_polars_schema",
]
