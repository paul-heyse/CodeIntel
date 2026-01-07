"""Type normalization helpers for Arrow arrays, tables, and schemas."""

from __future__ import annotations

from typing import cast

import pyarrow as pa
import pyarrow.compute as pc


def _is_list_type(data_type: pa.DataType) -> bool:
    return (
        pa.types.is_list(data_type)
        or pa.types.is_large_list(data_type)
        or pa.types.is_fixed_size_list(data_type)
    )


def is_binary_view_type(data_type: pa.DataType) -> bool:
    """Return whether the Arrow type is a binary view variant.

    Returns
    -------
    bool
        True when the data type is a binary view variant.
    """
    is_binary_view = getattr(pa.types, "is_binary_view", None)
    if is_binary_view is None:
        return False
    return is_binary_view(data_type)


def _string_view_cast_dictionary(data_type: pa.DictionaryType) -> pa.DataType:
    value_type = string_view_cast_type(data_type.value_type)
    if value_type == data_type.value_type:
        return data_type
    return pa.dictionary(data_type.index_type, value_type, ordered=data_type.ordered)


def _string_view_cast_list(data_type: pa.DataType) -> pa.DataType:
    value_type = string_view_cast_type(data_type.value_type)
    if value_type == data_type.value_type:
        return data_type
    if pa.types.is_large_list(data_type):
        return pa.large_list(value_type)
    if pa.types.is_list(data_type):
        return pa.list_(value_type)
    return pa.list_(value_type, list_size=data_type.list_size)


def _string_view_cast_struct(data_type: pa.StructType) -> pa.DataType:
    fields: list[pa.Field] = []
    changed = False
    for field in data_type:
        next_type = string_view_cast_type(field.type)
        if next_type != field.type:
            changed = True
        fields.append(
            pa.field(
                field.name,
                next_type,
                nullable=field.nullable,
                metadata=field.metadata,
            )
        )
    if not changed:
        return data_type
    return pa.struct(fields)


def _string_view_cast_map(data_type: pa.MapType) -> pa.DataType:
    key_type = string_view_cast_type(data_type.key_type)
    item_type = string_view_cast_type(data_type.item_type)
    if key_type == data_type.key_type and item_type == data_type.item_type:
        return data_type
    return pa.map_(key_type, item_type, keys_sorted=data_type.keys_sorted)


def string_view_cast_type(data_type: pa.DataType) -> pa.DataType:
    """Return a type with string view variants cast to canonical string types.

    Returns
    -------
    pyarrow.DataType
        Normalized Arrow type.
    """
    if pa.types.is_string_view(data_type):
        return pa.string()
    if pa.types.is_dictionary(data_type):
        return _string_view_cast_dictionary(cast("pa.DictionaryType", data_type))
    if _is_list_type(data_type):
        return _string_view_cast_list(data_type)
    if pa.types.is_struct(data_type):
        return _string_view_cast_struct(cast("pa.StructType", data_type))
    if pa.types.is_map(data_type):
        return _string_view_cast_map(cast("pa.MapType", data_type))
    return data_type


def _binary_view_cast_dictionary(data_type: pa.DictionaryType) -> pa.DataType:
    value_type = binary_view_cast_type(data_type.value_type)
    if value_type == data_type.value_type:
        return data_type
    return pa.dictionary(data_type.index_type, value_type, ordered=data_type.ordered)


def _binary_view_cast_list(data_type: pa.DataType) -> pa.DataType:
    value_type = binary_view_cast_type(data_type.value_type)
    if value_type == data_type.value_type:
        return data_type
    if pa.types.is_large_list(data_type):
        return pa.large_list(value_type)
    if pa.types.is_list(data_type):
        return pa.list_(value_type)
    return pa.list_(value_type, list_size=data_type.list_size)


def _binary_view_cast_struct(data_type: pa.StructType) -> pa.DataType:
    fields: list[pa.Field] = []
    changed = False
    for field in data_type:
        next_type = binary_view_cast_type(field.type)
        if next_type != field.type:
            changed = True
        fields.append(
            pa.field(
                field.name,
                next_type,
                nullable=field.nullable,
                metadata=field.metadata,
            )
        )
    if not changed:
        return data_type
    return pa.struct(fields)


def _binary_view_cast_map(data_type: pa.MapType) -> pa.DataType:
    key_type = binary_view_cast_type(data_type.key_type)
    item_type = binary_view_cast_type(data_type.item_type)
    if key_type == data_type.key_type and item_type == data_type.item_type:
        return data_type
    return pa.map_(key_type, item_type, keys_sorted=data_type.keys_sorted)


def binary_view_cast_type(data_type: pa.DataType) -> pa.DataType:
    """Return a type with binary view variants cast to canonical binary types.

    Returns
    -------
    pyarrow.DataType
        Normalized Arrow type.
    """
    if is_binary_view_type(data_type):
        return pa.binary()
    if pa.types.is_dictionary(data_type):
        return _binary_view_cast_dictionary(cast("pa.DictionaryType", data_type))
    if _is_list_type(data_type):
        return _binary_view_cast_list(data_type)
    if pa.types.is_struct(data_type):
        return _binary_view_cast_struct(cast("pa.StructType", data_type))
    if pa.types.is_map(data_type):
        return _binary_view_cast_map(cast("pa.MapType", data_type))
    return data_type


def normalize_string_view_array(
    values: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    """Cast string view arrays to canonical string types when needed.

    Returns
    -------
    pyarrow.Array | pyarrow.ChunkedArray
        Normalized array.
    """
    target_type = string_view_cast_type(values.type)
    if target_type == values.type:
        return values
    try:
        return pc.cast(values, target_type, safe=False)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, ValueError):
        return values


def normalize_binary_view_array(
    values: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    """Cast binary view arrays to canonical binary types when needed.

    Returns
    -------
    pyarrow.Array | pyarrow.ChunkedArray
        Normalized array.
    """
    target_type = binary_view_cast_type(values.type)
    if target_type == values.type:
        return values
    try:
        return pc.cast(values, target_type, safe=False)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, ValueError):
        return values


def normalize_string_view_schema(schema: pa.Schema) -> pa.Schema:
    """Return a schema with string view types normalized to canonical strings.

    Returns
    -------
    pyarrow.Schema
        Normalized schema.
    """
    fields: list[pa.Field] = []
    changed = False
    for field in schema:
        target_type = string_view_cast_type(field.type)
        if target_type != field.type:
            changed = True
        fields.append(
            pa.field(
                field.name,
                target_type,
                nullable=field.nullable,
                metadata=field.metadata,
            )
        )
    if not changed:
        return schema
    return pa.schema(fields, metadata=schema.metadata)


def normalize_binary_view_schema(schema: pa.Schema) -> pa.Schema:
    """Return a schema with binary view types normalized to canonical binaries.

    Returns
    -------
    pyarrow.Schema
        Normalized schema.
    """
    fields: list[pa.Field] = []
    changed = False
    for field in schema:
        target_type = binary_view_cast_type(field.type)
        if target_type != field.type:
            changed = True
        fields.append(
            pa.field(
                field.name,
                target_type,
                nullable=field.nullable,
                metadata=field.metadata,
            )
        )
    if not changed:
        return schema
    return pa.schema(fields, metadata=schema.metadata)


def normalize_string_view_table(table: pa.Table) -> pa.Table:
    """Return a table with string view types normalized to canonical strings.

    Returns
    -------
    pyarrow.Table
        Normalized table.
    """
    schema = normalize_string_view_schema(table.schema)
    if schema == table.schema:
        return table
    try:
        return table.cast(schema, safe=False)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, ValueError):
        return table


def normalize_binary_view_table(table: pa.Table) -> pa.Table:
    """Return a table with binary view types normalized to canonical binaries.

    Returns
    -------
    pyarrow.Table
        Normalized table.
    """
    schema = normalize_binary_view_schema(table.schema)
    if schema == table.schema:
        return table
    try:
        return table.cast(schema, safe=False)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, ValueError):
        return table


__all__ = [
    "binary_view_cast_type",
    "is_binary_view_type",
    "normalize_binary_view_array",
    "normalize_binary_view_schema",
    "normalize_binary_view_table",
    "normalize_string_view_array",
    "normalize_string_view_schema",
    "normalize_string_view_table",
    "string_view_cast_type",
]
