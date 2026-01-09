"""Join-safe projection helpers for columnar pipelines."""

from __future__ import annotations

from collections.abc import Sequence

import pyarrow as pa

from codeintel.core.validation.schema_constraints import is_list_like


def list_payload_columns(table: pa.Table) -> tuple[str, ...]:
    """Return column names containing list-like payloads.

    Parameters
    ----------
    table
        Table to inspect for list payloads.

    Returns
    -------
    tuple[str, ...]
        Column names containing list-like Arrow types.
    """
    return tuple(field.name for field in table.schema if is_list_like(field.type))


def require_join_safe_schema(
    table: pa.Table,
    *,
    allowed_columns: Sequence[str] = (),
) -> None:
    """Raise when list payloads are present in join inputs.

    Parameters
    ----------
    table
        Table to validate for join safety.
    allowed_columns
        Column names allowed to contain list payloads.

    Raises
    ------
    ValueError
        Raised when list payloads remain in disallowed columns.
    """
    allowed = set(allowed_columns)
    list_columns = [
        field.name
        for field in table.schema
        if is_list_like(field.type) and field.name not in allowed
    ]
    if not list_columns:
        return
    msg = f"Join inputs contain list payload columns: {list_columns}"
    raise ValueError(msg)


def join_safe_projection(
    table: pa.Table,
    *,
    allowed_columns: Sequence[str] = (),
) -> pa.Table:
    """Return a table projected to join-safe columns.

    Parameters
    ----------
    table
        Input table to project.
    allowed_columns
        Columns to retain when explicitly provided.

    Returns
    -------
    pyarrow.Table
        Join-safe projection of the input table.

    Raises
    ------
    ValueError
        Raised when projection removes all columns.
    """
    if allowed_columns:
        allowed_set = set(allowed_columns)
        keep = [name for name in table.column_names if name in allowed_set]
    else:
        keep = [field.name for field in table.schema if not is_list_like(field.type)]
    if not keep:
        msg = "Join-safe projection removed all columns; explode or whitelist columns."
        raise ValueError(msg)
    if keep == list(table.column_names):
        return table
    return table.select(keep)


__all__ = [
    "join_safe_projection",
    "list_payload_columns",
    "require_join_safe_schema",
]
