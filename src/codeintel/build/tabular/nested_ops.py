"""Nested Arrow helpers for struct/list/map evolution."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal, cast

import pyarrow as pa
import pyarrow.compute as pc

PromoteOptions = Literal["default", "permissive"]


def unify_schemas_with_contract_first(
    contract_schema: pa.Schema,
    schemas: Iterable[pa.Schema],
    *,
    promote: PromoteOptions = "default",
) -> pa.Schema:
    """Unify schemas with the contract schema first.

    Parameters
    ----------
    contract_schema
        Canonical contract schema.
    schemas
        Additional schemas to unify with the contract.
    promote
        Promotion policy for schema unification.

    Returns
    -------
    pyarrow.Schema
        Unified schema.
    """
    return pa.unify_schemas([contract_schema, *schemas], promote_options=promote)


def deep_cast_table_to_contract(table: pa.Table, contract_schema: pa.Schema) -> pa.Table:
    """Deep-cast a table to a contract schema.

    Parameters
    ----------
    table
        Input table to cast.
    contract_schema
        Target contract schema.

    Returns
    -------
    pyarrow.Table
        Table cast to the contract schema.
    """
    arrays: list[pa.Array | pa.ChunkedArray] = []
    fields: list[pa.Field] = []
    for field in contract_schema:
        if field.name in table.column_names:
            value = table[field.name]
        else:
            value = pa.nulls(table.num_rows, type=field.type)
        arrays.append(deep_cast_array(value, field.type))
        fields.append(field)
    return pa.Table.from_arrays(
        arrays,
        schema=pa.schema(fields, metadata=contract_schema.metadata),
    )


def deep_cast_array(
    values: pa.Array | pa.ChunkedArray,
    target_type: pa.DataType,
) -> pa.Array | pa.ChunkedArray:
    """Recursively cast nested Arrow arrays.

    Parameters
    ----------
    values
        Array or chunked array to cast.
    target_type
        Target Arrow data type.

    Returns
    -------
    pyarrow.Array | pyarrow.ChunkedArray
        Casted array.
    """
    if values.type.equals(target_type):
        return values
    if isinstance(values, pa.ChunkedArray):
        return pa.chunked_array(
            [deep_cast_array(chunk, target_type) for chunk in values.chunks],
            type=target_type,
        )
    source_type = values.type
    if _is_list_view_type(source_type) or _is_list_view_type(target_type):
        return pc.cast(values, target_type, safe=True)
    if pa.types.is_struct(target_type):
        casted = _cast_struct(values, target_type)
    elif _is_list_type(target_type):
        casted = _cast_list(values, target_type)
    elif pa.types.is_map(target_type):
        casted = _cast_map(values, target_type)
    else:
        casted = pc.cast(values, target_type, safe=True)
    return casted


def make_extras_struct(
    table: pa.Table,
    *,
    fields: Mapping[str, pa.DataType],
) -> pa.StructArray:
    """Build a struct array for extras metadata.

    Parameters
    ----------
    table
        Source table with columns used for extras.
    fields
        Mapping of struct field name to Arrow data type.

    Returns
    -------
    pyarrow.StructArray
        Struct array containing extras metadata.
    """
    arrays: list[pa.Array | pa.ChunkedArray] = []
    struct_fields: list[pa.Field] = []
    for name, dtype in fields.items():
        if name in table.column_names:
            values = pc.cast(table[name], dtype, safe=True)
        else:
            values = pa.nulls(table.num_rows, type=dtype)
        arrays.append(values)
        struct_fields.append(pa.field(name, dtype))
    return pa.StructArray.from_arrays(arrays, fields=struct_fields)


def make_extras_kv_map(
    table: pa.Table,
    *,
    keys: str,
    values: str,
) -> pa.Array:
    """Build a map array from list key/value columns.

    Parameters
    ----------
    table
        Source table with list key/value columns.
    keys
        Column containing list keys.
    values
        Column containing list values.

    Returns
    -------
    pyarrow.Array
        Map array derived from the key/value lists.

    Raises
    ------
    TypeError
        If keys/values are not list arrays.
    ValueError
        If list offsets do not match.
    """
    map_type = pa.map_(pa.string(), pa.string())
    if keys not in table.column_names or values not in table.column_names:
        return pa.nulls(table.num_rows, type=map_type)
    keys_arr = table[keys]
    values_arr = table[values]
    if isinstance(keys_arr, pa.ChunkedArray):
        keys_arr = keys_arr.combine_chunks()
    if isinstance(values_arr, pa.ChunkedArray):
        values_arr = values_arr.combine_chunks()
    is_list_keys = pa.types.is_list(keys_arr.type) or pa.types.is_large_list(keys_arr.type)
    is_list_values = pa.types.is_list(values_arr.type) or pa.types.is_large_list(values_arr.type)
    if not is_list_keys or not is_list_values:
        msg = "extras_kv requires list columns for keys and values"
        raise TypeError(msg)
    keys_list = cast("pa.ListArray | pa.LargeListArray", keys_arr)
    values_list = cast("pa.ListArray | pa.LargeListArray", values_arr)
    if not keys_list.offsets.equals(values_list.offsets):
        msg = "extras_kv list offsets must match"
        raise ValueError(msg)
    return pa.MapArray.from_arrays(
        keys_list.offsets,
        keys_list.values,
        values_list.values,
        type=map_type,
    )


def _cast_struct(values: pa.Array, target_type: pa.StructType) -> pa.Array:
    arrays: list[pa.Array | pa.ChunkedArray] = []
    fields: list[pa.Field] = []
    for field in target_type:
        if field.name in values.type:
            child = values.field(field.name)
            casted = deep_cast_array(child, field.type)
        else:
            casted = pa.nulls(len(values), type=field.type)
        arrays.append(casted)
        fields.append(field)
    return pa.StructArray.from_arrays(arrays, fields=fields)


def _cast_list(values: pa.Array, target_type: pa.DataType) -> pa.Array:
    if not _is_list_type(values.type):
        return pc.cast(values, target_type, safe=True)
    casted_values = deep_cast_array(values.values, target_type.value_type)
    if pa.types.is_fixed_size_list(target_type):
        return pa.FixedSizeListArray.from_arrays(casted_values, target_type.list_size)
    if pa.types.is_large_list(target_type):
        return pa.LargeListArray.from_arrays(values.offsets, casted_values, type=target_type)
    return pa.ListArray.from_arrays(values.offsets, casted_values, type=target_type)


def _cast_map(values: pa.Array, target_type: pa.MapType) -> pa.Array:
    if not pa.types.is_map(values.type):
        return pc.cast(values, target_type, safe=True)
    keys_cast = deep_cast_array(values.keys, target_type.key_type)
    items_cast = deep_cast_array(values.items, target_type.item_type)
    return pa.MapArray.from_arrays(values.offsets, keys_cast, items_cast, type=target_type)


def _is_list_type(data_type: pa.DataType) -> bool:
    return (
        pa.types.is_list(data_type)
        or pa.types.is_large_list(data_type)
        or pa.types.is_fixed_size_list(data_type)
    )


def _is_list_view_type(data_type: pa.DataType) -> bool:
    is_list_view = getattr(pa.types, "is_list_view", None)
    is_large_list_view = getattr(pa.types, "is_large_list_view", None)
    return bool(
        (callable(is_list_view) and is_list_view(data_type))
        or (callable(is_large_list_view) and is_large_list_view(data_type))
    )


__all__ = [
    "deep_cast_array",
    "deep_cast_table_to_contract",
    "make_extras_kv_map",
    "make_extras_struct",
    "unify_schemas_with_contract_first",
]
