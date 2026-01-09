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
    unified = pa.unify_schemas([contract_schema, *schemas], promote_options=promote)
    _validate_contract_promotions(contract_schema, unified)
    return unified


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
    *,
    cast_options: pc.CastOptions | None = None,
) -> pa.Array | pa.ChunkedArray:
    """Recursively cast nested Arrow arrays.

    Parameters
    ----------
    values
        Array or chunked array to cast.
    target_type
        Target Arrow data type.
    cast_options
        Optional Arrow cast options for explicit promotion control.

    Returns
    -------
    pyarrow.Array | pyarrow.ChunkedArray
        Casted array.
    """
    if values.type.equals(target_type):
        return values
    source_type = values.type
    _ensure_allowed_promotion(source_type, target_type)
    if isinstance(values, pa.ChunkedArray):
        return pa.chunked_array(
            [
                deep_cast_array(chunk, target_type, cast_options=cast_options)
                for chunk in values.chunks
            ],
            type=target_type,
        )
    if _is_list_view_type(source_type) or _is_list_view_type(target_type):
        if cast_options is None:
            return pc.cast(values, target_type, safe=True)
        return pc.cast(values, options=cast_options)
    if pa.types.is_struct(target_type):
        casted = _cast_struct(values, target_type, cast_options=cast_options)
    elif _is_list_type(target_type):
        casted = _cast_list(values, target_type, cast_options=cast_options)
    elif pa.types.is_map(target_type):
        casted = _cast_map(values, target_type, cast_options=cast_options)
    elif cast_options is None:
        casted = pc.cast(values, target_type, safe=True)
    else:
        casted = pc.cast(values, options=cast_options)
    return casted


def is_allowed_promotion(source_type: pa.DataType, target_type: pa.DataType) -> bool:
    """Return True when a promotion from source to target is allowed.

    Returns
    -------
    bool
        True when the promotion is allowed.
    """
    source_type = _unwrap_dictionary(source_type)
    target_type = _unwrap_dictionary(target_type)
    if source_type.equals(target_type):
        return True
    if pa.types.is_null(source_type):
        return True
    list_allowed = _list_promotion_allowed(source_type, target_type)
    if list_allowed is not None:
        return list_allowed
    struct_allowed = _struct_promotion_allowed_if_struct(source_type, target_type)
    if struct_allowed is not None:
        return struct_allowed
    map_allowed = _map_promotion_allowed_if_map(source_type, target_type)
    if map_allowed is not None:
        return map_allowed
    return _scalar_promotion_allowed(source_type, target_type)


def _list_promotion_allowed(
    source_type: pa.DataType,
    target_type: pa.DataType,
) -> bool | None:
    source_list = _list_kind(source_type)
    target_list = _list_kind(target_type)
    if not source_list and not target_list:
        return None
    if source_list != target_list:
        return False
    if source_list == "fixed_size_list" and source_type.list_size != target_type.list_size:
        return False
    return is_allowed_promotion(source_type.value_type, target_type.value_type)


def _struct_promotion_allowed_if_struct(
    source_type: pa.DataType,
    target_type: pa.DataType,
) -> bool | None:
    if not (pa.types.is_struct(source_type) or pa.types.is_struct(target_type)):
        return None
    if not (pa.types.is_struct(source_type) and pa.types.is_struct(target_type)):
        return False
    return _struct_promotion_allowed(
        cast("pa.StructType", source_type),
        cast("pa.StructType", target_type),
    )


def _map_promotion_allowed_if_map(
    source_type: pa.DataType,
    target_type: pa.DataType,
) -> bool | None:
    if not (pa.types.is_map(source_type) or pa.types.is_map(target_type)):
        return None
    if not (pa.types.is_map(source_type) and pa.types.is_map(target_type)):
        return False
    return _map_promotion_allowed(
        cast("pa.MapType", source_type),
        cast("pa.MapType", target_type),
    )


def _scalar_promotion_allowed(
    source_type: pa.DataType,
    target_type: pa.DataType,
) -> bool:
    checks = (
        _is_int_promotion,
        _is_uint_promotion,
        _is_float_promotion,
        _is_string_promotion,
        _is_timestamp_promotion,
    )
    return any(check(source_type, target_type) for check in checks)


def _ensure_allowed_promotion(source_type: pa.DataType, target_type: pa.DataType) -> None:
    if is_allowed_promotion(source_type, target_type):
        return
    msg = f"Disallowed promotion from {source_type} to {target_type}"
    raise ValueError(msg)


def _validate_contract_promotions(contract_schema: pa.Schema, unified_schema: pa.Schema) -> None:
    for field in contract_schema:
        if field.name not in unified_schema.names:
            continue
        unified_field = unified_schema.field(field.name)
        if is_allowed_promotion(unified_field.type, field.type):
            continue
        msg = f"Disallowed promotion for {field.name}: {unified_field.type} -> {field.type}"
        raise ValueError(msg)


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


def _cast_struct(
    values: pa.Array,
    target_type: pa.StructType,
    *,
    cast_options: pc.CastOptions | None = None,
) -> pa.Array:
    arrays: list[pa.Array | pa.ChunkedArray] = []
    fields: list[pa.Field] = []
    for field in target_type:
        if field.name in values.type:
            child = values.field(field.name)
            arrays.append(deep_cast_array(child, field.type, cast_options=cast_options))
        else:
            arrays.append(pa.nulls(len(values), type=field.type))
        fields.append(field)
    return pa.StructArray.from_arrays(arrays, fields=fields)


def _cast_list(
    values: pa.Array,
    target_type: pa.DataType,
    *,
    cast_options: pc.CastOptions | None = None,
) -> pa.Array:
    if not _is_list_type(target_type):
        msg = "target list type is required"
        raise TypeError(msg)
    offsets = values.offsets
    value_type = target_type.value_type
    casted_values = deep_cast_array(values.values, value_type, cast_options=cast_options)
    if pa.types.is_fixed_size_list(target_type):
        return pa.FixedSizeListArray.from_arrays(casted_values, target_type.list_size)
    return pa.ListArray.from_arrays(offsets, casted_values, type=target_type)


def _cast_map(
    values: pa.Array,
    target_type: pa.MapType,
    *,
    cast_options: pc.CastOptions | None = None,
) -> pa.Array:
    key_type = target_type.key_type
    item_type = target_type.item_type
    keys = deep_cast_array(values.keys, key_type, cast_options=cast_options)
    items = deep_cast_array(values.items, item_type, cast_options=cast_options)
    return pa.MapArray.from_arrays(values.offsets, keys, items, type=target_type)


def _is_list_type(data_type: pa.DataType) -> bool:
    return (
        pa.types.is_list(data_type)
        or pa.types.is_large_list(data_type)
        or pa.types.is_fixed_size_list(data_type)
    )


def _list_kind(data_type: pa.DataType) -> str | None:
    if pa.types.is_list(data_type):
        return "list"
    if pa.types.is_large_list(data_type):
        return "large_list"
    if pa.types.is_fixed_size_list(data_type):
        return "fixed_size_list"
    is_list_view = getattr(pa.types, "is_list_view", None)
    if callable(is_list_view) and is_list_view(data_type):
        return "list_view"
    is_large_list_view = getattr(pa.types, "is_large_list_view", None)
    if callable(is_large_list_view) and is_large_list_view(data_type):
        return "large_list_view"
    return None


def _is_list_view_type(data_type: pa.DataType) -> bool:
    is_list_view = getattr(pa.types, "is_list_view", None)
    is_large_list_view = getattr(pa.types, "is_large_list_view", None)
    return bool(
        (callable(is_list_view) and is_list_view(data_type))
        or (callable(is_large_list_view) and is_large_list_view(data_type))
    )


def _unwrap_dictionary(data_type: pa.DataType) -> pa.DataType:
    if pa.types.is_dictionary(data_type):
        return data_type.value_type
    return data_type


def _int_bit_width(data_type: pa.DataType) -> int | None:
    bit_width = getattr(data_type, "bit_width", None)
    if isinstance(bit_width, int):
        return bit_width
    return None


def _is_int_promotion(source_type: pa.DataType, target_type: pa.DataType) -> bool:
    if not (pa.types.is_signed_integer(source_type) and pa.types.is_signed_integer(target_type)):
        return False
    source_width = _int_bit_width(source_type)
    target_width = _int_bit_width(target_type)
    if source_width is None or target_width is None:
        return False
    return source_width <= target_width


def _is_uint_promotion(source_type: pa.DataType, target_type: pa.DataType) -> bool:
    if not (
        pa.types.is_unsigned_integer(source_type)
        and pa.types.is_unsigned_integer(target_type)
    ):
        return False
    source_width = _int_bit_width(source_type)
    target_width = _int_bit_width(target_type)
    if source_width is None or target_width is None:
        return False
    return source_width <= target_width


def _is_float_promotion(source_type: pa.DataType, target_type: pa.DataType) -> bool:
    if not (pa.types.is_floating(source_type) and pa.types.is_floating(target_type)):
        return False
    source_width = _int_bit_width(source_type)
    target_width = _int_bit_width(target_type)
    if source_width is None or target_width is None:
        return False
    return source_width <= target_width


def _is_string_promotion(source_type: pa.DataType, target_type: pa.DataType) -> bool:
    return (
        (pa.types.is_string(source_type) and pa.types.is_large_string(target_type))
        or (pa.types.is_string(source_type) and pa.types.is_string(target_type))
        or (pa.types.is_large_string(source_type) and pa.types.is_large_string(target_type))
    )


def _is_timestamp_promotion(source_type: pa.DataType, target_type: pa.DataType) -> bool:
    return (
        pa.types.is_timestamp(source_type)
        and pa.types.is_timestamp(target_type)
        and source_type.unit == target_type.unit
        and source_type.tz == target_type.tz
    )


def _struct_promotion_allowed(source_type: pa.StructType, target_type: pa.StructType) -> bool:
    source_fields = {field.name: field for field in source_type}
    for target_field in target_type:
        source_field = source_fields.get(target_field.name)
        if source_field is None:
            if target_field.nullable:
                continue
            return False
        if not is_allowed_promotion(source_field.type, target_field.type):
            return False
    return True


def _map_promotion_allowed(source_type: pa.MapType, target_type: pa.MapType) -> bool:
    source_key = _unwrap_dictionary(source_type.key_type)
    target_key = _unwrap_dictionary(target_type.key_type)
    if not source_key.equals(target_key):
        return False
    return is_allowed_promotion(source_type.item_type, target_type.item_type)


__all__ = [
    "deep_cast_array",
    "deep_cast_table_to_contract",
    "is_allowed_promotion",
    "make_extras_kv_map",
    "make_extras_struct",
    "unify_schemas_with_contract_first",
]
