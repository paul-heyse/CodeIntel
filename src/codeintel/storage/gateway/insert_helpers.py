"""Generic insert helpers backed by DuckDBPolicyBackend.

These helpers normalize mapping rows (e.g., TypedDict) into stable column-order
tuples and delegate mutations to DuckDBPolicyBackend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from codeintel.storage.gateway.protocol import StorageGateway

__all__ = ["insert_one", "insert_rows"]


def insert_rows(
    gateway: StorageGateway,
    table_key: str,
    rows: Iterable[Mapping[str, object]],
    *,
    columns: Sequence[str] | None = None,
) -> None:
    """Insert mapping rows into the given table.

    Parameters
    ----------
    gateway
        Storage gateway.
    table_key
        Fully qualified table name (schema.table).
    rows
        Iterable of mapping-based rows (e.g., TypedDict models) whose keys
        align with the table's schema columns.
    columns
        Optional explicit column order for insertion. When omitted, columns are
        derived from the dataset contract schema.
    """
    gateway.policy.bulk_insert_mappings(table_key, rows, columns=columns)


def insert_one(
    gateway: StorageGateway,
    table_key: str,
    row: Mapping[str, object],
    *,
    columns: Sequence[str] | None = None,
) -> None:
    """Insert a single mapping row into the given table."""
    insert_rows(gateway, table_key, (row,), columns=columns)
