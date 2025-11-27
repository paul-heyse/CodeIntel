"""Shared repository helpers for DuckDB-backed storage."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from codeintel.storage.gateway import DuckDBConnection, StorageGateway

RowDict = dict[str, Any]


def fetch_one_dict(con: DuckDBConnection, sql: str, params: Sequence[object]) -> RowDict | None:
    """
    Execute a query and return the first row as a mapping.

    Returns
    -------
    RowDict | None
        Mapping of column to value when a row exists; otherwise ``None``.
    """
    result = con.execute(sql, list(params))
    row = result.fetchone()
    if row is None:
        return None
    cols = [desc[0] for desc in result.description]
    return {col: row[idx] for idx, col in enumerate(cols)}


def fetch_all_dicts(con: DuckDBConnection, sql: str, params: Sequence[object]) -> list[RowDict]:
    """
    Execute a query and return all rows as mappings.

    Returns
    -------
    list[RowDict]
        List of rows represented as dictionaries keyed by column name.
    """
    result = con.execute(sql, list(params))
    rows = result.fetchall()
    cols = [desc[0] for desc in result.description]
    return [{col: row[idx] for idx, col in enumerate(cols)} for row in rows]


@dataclass(frozen=True)
class BaseRepository:
    """Base class for repositories bound to a gateway/revision."""

    gateway: StorageGateway
    repo: str
    commit: str

    @property
    def con(self) -> DuckDBConnection:
        """Return the underlying DuckDB connection."""
        return self.gateway.con
