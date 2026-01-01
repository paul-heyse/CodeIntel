"""Shared filter operator types and validation helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from codeintel.core.schemas.primitives import COMPLEX_TYPE_BASES, column_type_base

if TYPE_CHECKING:
    from codeintel.core.schemas.primitives import ColumnType

Op = Literal["eq", "ne", "lt", "lte", "gt", "gte", "in", "contains", "startswith"]
type FilterScalar = bool | int | float | str
type FilterValue = FilterScalar | list[FilterScalar]


class FilterSpecLike(Protocol):
    """Structural filter specification interface."""

    @property
    def column(self) -> str:
        """Return the filter column name."""
        ...

    @property
    def op(self) -> Op:
        """Return the filter operator."""
        ...

    @property
    def value(self) -> FilterValue:
        """Return the filter value."""
        ...


@dataclass(frozen=True, slots=True)
class FilterSpecInput:
    """Lightweight filter spec for internal usage."""

    column: str
    op: Op
    value: FilterValue


_ALL_OPS: tuple[Op, ...] = ("eq", "ne", "lt", "lte", "gt", "gte", "in", "contains", "startswith")
_ORDERING_OPS = frozenset({"lt", "lte", "gt", "gte"})
_STRING_OPS = frozenset({"contains", "startswith"})


class FilterOpError(ValueError):
    """Raised when filter operations are invalid for the column type."""


def allowed_ops_for_column_type(column_type: ColumnType | str | None) -> tuple[Op, ...]:
    """Return allowed filter operators for a given column type.

    Parameters
    ----------
    column_type
        Column type string (typically a DuckDB type like ``VARCHAR``).

    Returns
    -------
    tuple[Op, ...]
        Allowed operators for the given type.
    """
    allowed_ops = _ALL_OPS
    if column_type is not None:
        base = column_type_base(column_type)
        if base == "BOOLEAN":
            allowed_ops = ("eq", "ne")
        elif base == "VARCHAR":
            allowed_ops = ("eq", "ne", "in", "contains", "startswith")
        elif base in COMPLEX_TYPE_BASES:
            allowed_ops = ("eq", "ne")
        else:
            allowed_ops = ("eq", "ne", "lt", "lte", "gt", "gte", "in")

    return allowed_ops


def allowed_ops_for_column_types(
    column_types: Mapping[str, ColumnType] | None,
) -> dict[str, tuple[Op, ...]]:
    """Return per-column allowed operators for a column type mapping.

    Parameters
    ----------
    column_types
        Column type mapping (column name to type string).

    Returns
    -------
    dict[str, tuple[Op, ...]]
        Allowed operators per column.
    """
    if column_types is None:
        return {}
    return {
        column: allowed_ops_for_column_type(column_type)
        for column, column_type in column_types.items()
    }


def validate_filter_value(
    *,
    op: Op,
    value: FilterValue,
    column_type: ColumnType | str | None,
    allowed_ops: tuple[Op, ...] | None = None,
) -> FilterValue:
    """Validate a filter operator/value pair for a column type.

    Parameters
    ----------
    op
        Filter operator.
    value
        Filter value to validate.
    column_type
        Column type used to enforce operator constraints.
    allowed_ops
        Optional precomputed operator allowlist.

    Returns
    -------
    FilterValue
        The validated filter value.

    Raises
    ------
    FilterOpError
        If the operator/value pair is invalid for the column type.
    """
    base = column_type_base(column_type) if column_type is not None else None
    resolved_ops = allowed_ops or allowed_ops_for_column_type(column_type)
    if op not in resolved_ops:
        msg = f"Operator {op} is not supported for column type {column_type}"
        raise FilterOpError(msg)

    if op in _STRING_OPS:
        if not isinstance(value, str):
            msg = f"{op} operator requires a string value"
            raise FilterOpError(msg)
        if base is not None and base != "VARCHAR":
            msg = f"{op} operator is only supported for VARCHAR columns"
            raise FilterOpError(msg)
        return value

    if op == "in":
        if not isinstance(value, list):
            msg = "IN operator requires a list value"
            raise FilterOpError(msg)
        if base in COMPLEX_TYPE_BASES:
            msg = "IN operator is not supported for complex columns"
            raise FilterOpError(msg)
        return value

    if isinstance(value, list):
        msg = f"{op} operator does not support list value"
        raise FilterOpError(msg)
    if op in _ORDERING_OPS and base == "VARCHAR":
        msg = f"Operator {op} is not supported for string columns"
        raise FilterOpError(msg)
    return value


def parse_filter_value(column_type: ColumnType | str | None, *, op: Op, raw: str) -> object:
    """Parse a raw string into a best-effort typed filter value.

    Parameters
    ----------
    column_type
        Column type used to guide parsing.
    op
        Filter operator.
    raw
        Raw value string.

    Returns
    -------
    object
        Parsed value for use in filter specs.
    """
    if op == "in":
        items = [item.strip() for item in raw.split(",") if item.strip()]
        if column_type is None:
            return items
        return [_parse_scalar_value(column_type, item) for item in items]
    if column_type is None:
        return raw
    return _parse_scalar_value(column_type, raw)


def _parse_scalar_value(column_type: ColumnType | str, raw: str) -> object:
    base = column_type_base(column_type)
    parsed: object = raw
    if base in {"INTEGER", "BIGINT"}:
        try:
            parsed = int(raw)
        except ValueError:
            parsed = raw
    elif base in {"DOUBLE", "DECIMAL"}:
        try:
            parsed = float(raw)
        except ValueError:
            parsed = raw
    elif base == "BOOLEAN":
        lowered = raw.lower()
        if lowered in {"true", "1", "yes"}:
            parsed = True
        elif lowered in {"false", "0", "no"}:
            parsed = False
        else:
            parsed = raw

    return parsed


__all__ = [
    "FilterOpError",
    "FilterScalar",
    "FilterSpecInput",
    "FilterSpecLike",
    "FilterValue",
    "Op",
    "allowed_ops_for_column_type",
    "allowed_ops_for_column_types",
    "parse_filter_value",
    "validate_filter_value",
]
