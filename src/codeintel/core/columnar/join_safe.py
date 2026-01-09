"""Join-safe projection helpers for columnar pipelines."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.core.schemas.primitives import resolve_join_safe_columns
from codeintel.core.schemas.service import get_schema_service
from codeintel.core.validation.schema_constraints import is_list_like

if TYPE_CHECKING:
    from codeintel.core.schemas.service import SchemaService


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
    table_key: str | None = None,
    schema_service: SchemaService | None = None,
) -> None:
    """Raise when list payloads are present in join inputs.

    Parameters
    ----------
    table
        Table to validate for join safety.
    allowed_columns
        Column names allowed to contain list payloads.
    table_key
        Optional table key used to resolve join-safe allowlists.
    schema_service
        Optional SchemaService override for policy resolution.

    Raises
    ------
    ValueError
        Raised when list payloads remain in disallowed columns.
    """
    resolved_allowed = _resolve_join_safe_allowlist(
        allowed_columns=allowed_columns,
        table_key=table_key,
        schema_service=schema_service,
    )
    allowed = set(resolved_allowed)
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
    table_key: str | None = None,
    schema_service: SchemaService | None = None,
) -> pa.Table:
    """Return a table projected to join-safe columns.

    Parameters
    ----------
    table
        Input table to project.
    allowed_columns
        Columns to retain when explicitly provided.
    table_key
        Optional table key used to resolve join-safe allowlists.
    schema_service
        Optional SchemaService override for policy resolution.

    Returns
    -------
    pyarrow.Table
        Join-safe projection of the input table.

    Raises
    ------
    ValueError
        Raised when projection removes all columns.
    """
    resolved_allowed = _resolve_join_safe_allowlist(
        allowed_columns=allowed_columns,
        table_key=table_key,
        schema_service=schema_service,
    )
    if allowed_columns:
        allowed_set = set(resolved_allowed)
        keep = [name for name in table.column_names if name in allowed_set]
    else:
        allowed_set = set(resolved_allowed)
        keep = [
            field.name
            for field in table.schema
            if not is_list_like(field.type) or field.name in allowed_set
        ]
    if not keep:
        msg = "Join-safe projection removed all columns; explode or whitelist columns."
        raise ValueError(msg)
    if keep == list(table.column_names):
        return table
    return table.select(keep)


def _resolve_join_safe_allowlist(
    *,
    allowed_columns: Sequence[str],
    table_key: str | None,
    schema_service: SchemaService | None,
) -> tuple[str, ...]:
    if allowed_columns:
        return tuple(allowed_columns)
    if table_key is None:
        return ()
    service = schema_service or get_schema_service()
    table_schema = service.get_table_schema(table_key)
    return resolve_join_safe_columns(table_schema)


__all__ = [
    "join_safe_projection",
    "list_payload_columns",
    "require_join_safe_schema",
]
