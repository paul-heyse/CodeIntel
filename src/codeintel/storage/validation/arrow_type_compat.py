"""Arrow type compatibility helpers for columnar validation."""

from __future__ import annotations

import re
from collections.abc import Callable

import pyarrow as pa

from codeintel.core.schemas.primitives import Column, column_type_base

_DECIMAL_PATTERN = re.compile(r"^DECIMAL\\((\\d+),(\\d+)\\)$")


def decimal_scale_zero(column_type: str) -> bool:
    """Return True if a decimal type has scale=0.

    Returns
    -------
    bool
        True when the decimal scale is zero.
    """
    compact = column_type.upper().replace(" ", "")
    match = _DECIMAL_PATTERN.match(compact)
    if match is None:
        return False
    return int(match.group(2)) == 0


def is_list_like(dtype: pa.DataType) -> bool:
    """Return True if the Arrow type is list-like.

    Returns
    -------
    bool
        True when the type is list-like.
    """
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


def is_compatible_arrow_type(column: Column, actual_type: pa.DataType) -> bool:
    """Return True when the Arrow type is compatible with the column definition.

    Returns
    -------
    bool
        True when the Arrow type is compatible.
    """
    normalized = _unwrap_dictionary_type(actual_type)
    if pa.types.is_null(normalized):
        return column.nullable
    base = column_type_base(column.type)
    compatibility = _compatibility_for_base(base, column.type, normalized)
    if compatibility is None:
        return True
    return compatibility


def _compatibility_for_base(
    base: str,
    column_type: str,
    normalized: pa.DataType,
) -> bool | None:
    checker = _DIRECT_COMPAT_CHECKS.get(base)
    if checker is not None:
        return checker(normalized)
    predicate = _predicate_for_base(base, column_type)
    if predicate is None:
        return None
    return predicate(normalized)


def _predicate_for_base(
    base: str,
    column_type: str,
) -> Callable[[pa.DataType], bool] | None:
    if base == "DECIMAL" and decimal_scale_zero(column_type):
        return _is_decimal_or_int
    return _BASE_TYPE_PREDICATES.get(base)


def _unwrap_dictionary_type(data_type: pa.DataType) -> pa.DataType:
    if pa.types.is_dictionary(data_type):
        return data_type.value_type
    return data_type


def _is_decimal_or_int(data_type: pa.DataType) -> bool:
    return pa.types.is_integer(data_type) or pa.types.is_decimal(data_type)


def _is_decimal_or_float(data_type: pa.DataType) -> bool:
    return (
        pa.types.is_floating(data_type)
        or pa.types.is_decimal(data_type)
        or pa.types.is_integer(data_type)
    )


def _is_string_like(data_type: pa.DataType) -> bool:
    return pa.types.is_string(data_type) or pa.types.is_large_string(data_type)


def _is_temporal(data_type: pa.DataType) -> bool:
    return pa.types.is_timestamp(data_type) or pa.types.is_date(data_type)


def _always_true(_: pa.DataType) -> bool:
    return True


def _build_base_type_predicates() -> dict[str, Callable[[pa.DataType], bool]]:
    return {
        "INTEGER": pa.types.is_integer,
        "BIGINT": pa.types.is_integer,
        "DOUBLE": _is_decimal_or_float,
        "DECIMAL": _is_decimal_or_float,
        "BOOLEAN": pa.types.is_boolean,
        "VARCHAR": _is_string_like,
        "TIMESTAMP": _is_temporal,
        "TIMESTAMPTZ": _is_temporal,
    }


_BASE_TYPE_PREDICATES = _build_base_type_predicates()
_DIRECT_COMPAT_CHECKS: dict[str, Callable[[pa.DataType], bool]] = {
    "JSON": _always_true,
    "STRUCT": pa.types.is_struct,
    "LIST": is_list_like,
    "MAP": pa.types.is_map,
    "UNION": pa.types.is_union,
}


__all__ = [
    "decimal_scale_zero",
    "is_compatible_arrow_type",
    "is_list_like",
]
