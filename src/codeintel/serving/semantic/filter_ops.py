"""Canonical filter operator semantics for semantic queries.

This module is the single source of truth for which filter operators are valid
for a given column type across both query building and prompt guidance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.serving.semantic.models import Op

if TYPE_CHECKING:
    from codeintel.core.schemas.primitives import ColumnType


_ALL_OPS: tuple[Op, ...] = ("eq", "ne", "lt", "lte", "gt", "gte", "in", "contains", "startswith")

_NUMERIC_TYPES: frozenset[str] = frozenset(
    {
        "INTEGER",
        "BIGINT",
        "DOUBLE",
        "DECIMAL",
        "DECIMAL(38,0)",
    }
)

_TIME_TYPES: frozenset[str] = frozenset({"TIMESTAMP", "TIMESTAMPTZ"})


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
        normalized = str(column_type).upper()
        if normalized == "BOOLEAN":
            allowed_ops = ("eq", "ne")
        elif normalized == "VARCHAR":
            allowed_ops = ("eq", "ne", "in", "contains", "startswith")
        elif normalized == "JSON":
            allowed_ops = ("eq", "ne")
        else:
            allowed_ops = ("eq", "ne", "lt", "lte", "gt", "gte", "in")

    return allowed_ops


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
    normalized = str(column_type).upper()
    parsed: object = raw
    if normalized in {"INTEGER", "BIGINT"}:
        try:
            parsed = int(raw)
        except ValueError:
            parsed = raw
    elif normalized in {"DOUBLE", "DECIMAL", "DECIMAL(38,0)"}:
        try:
            parsed = float(raw)
        except ValueError:
            parsed = raw
    elif normalized == "BOOLEAN":
        lowered = raw.lower()
        if lowered in {"true", "1", "yes"}:
            parsed = True
        elif lowered in {"false", "0", "no"}:
            parsed = False
        else:
            parsed = raw

    return parsed


__all__ = ["allowed_ops_for_column_type", "parse_filter_value"]
