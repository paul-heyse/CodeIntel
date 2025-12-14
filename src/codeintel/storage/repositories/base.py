"""Shared repository helpers for DuckDB-backed storage.

This module provides base classes and helper functions for DuckDB-backed
repositories. All repositories should extend BaseRepository and use the
standardized fetch helpers for consistent data access patterns.

Method Signature Patterns
-------------------------
Repositories should follow these patterns for method signatures:

Single-row fetch:
    def get_X(self, id: int) -> RowDict | None: ...

List fetch with pagination:
    def list_X(self, *, limit: int | None = None) -> list[RowDict]: ...

Paginated fetch with truncation detection:
    def list_X_paginated(self, *, limit: int) -> PaginatedRows: ...

Existence check:
    def has_X(self, id: int) -> bool: ...
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from codeintel.config.datasets.validation import validate_df
from codeintel.core.repository import PagedResult

if TYPE_CHECKING:
    from ibis.expr import types as it

    from codeintel.storage.gateway import DuckDBConnection, StorageGateway

log = logging.getLogger(__name__)

RowDict = dict[str, Any]


class PaginatedRows(PagedResult[RowDict]):
    """Deprecated alias for PagedResult[RowDict].

    .. deprecated:: 1.0
        Use :class:`~codeintel.core.repository.PagedResult` instead.

    This class exists for backward compatibility. New code should use
    PagedResult directly from codeintel.core.repository.
    """

    def __init__(
        self,
        rows: list[RowDict],
        limit: int,
        *,
        truncated: bool,
        total_available: int | None = None,
    ) -> None:
        """Initialize with legacy field names.

        Parameters
        ----------
        rows
            The fetched rows (maps to items).
        limit
            The limit used for the query.
        truncated
            Whether more rows exist beyond the limit.
        total_available
            Total row count if known (maps to total).
        """
        warnings.warn(
            "PaginatedRows is deprecated. Use PagedResult from codeintel.core.repository instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            items=rows,
            total=total_available,
            limit=limit,
            offset=0,
            truncated=truncated,
        )

    @property
    def rows(self) -> list[RowDict]:
        """Return rows (alias for items, for backward compatibility)."""
        return self.items

    @property
    def total_available(self) -> int | None:
        """Return total available (alias for total, for backward compatibility)."""
        return self.total


@dataclass(frozen=True)
class BaseRepository:
    """
    Base class for repositories bound to a gateway/revision.

    All repositories should extend this class to ensure consistent
    connection management and revision binding.

    Parameters
    ----------
    gateway
        Storage gateway providing the DuckDB connection.
    repo
        Repository identifier (e.g., "org/repo").
    commit
        Commit hash for the snapshot.
    """

    gateway: StorageGateway
    repo: str
    commit: str

    @property
    def con(self) -> DuckDBConnection:
        """Return the underlying DuckDB connection."""
        return self.gateway.con

    def _ibis_table(self, table_key: str) -> it.Table:
        """
        Return an Ibis table expression for the given table key.

        Parameters
        ----------
        table_key
            Fully qualified table name (e.g., "core.goids").

        Returns
        -------
        it.Table
            Ibis table expression.
        """
        return self.gateway.ibis.table(table_key)

    def _ibis_to_df(
        self,
        expr: it.Table,
        table_key: str | None = None,
    ) -> pd.DataFrame:
        """
        Execute an Ibis expression and return a validated DataFrame.

        Parameters
        ----------
        expr
            Ibis table expression.
        table_key
            Optional table key for Pandera validation.

        Returns
        -------
        pd.DataFrame
            Validated DataFrame.
        """
        _ = self
        df = pd.DataFrame(expr.execute())
        if table_key:
            return validate_df(table_key, df)
        return df

    def _ibis_to_dicts(
        self,
        expr: it.Table,
        table_key: str | None = None,
    ) -> list[RowDict]:
        """
        Execute an Ibis expression and return rows as dicts.

        Parameters
        ----------
        expr
            Ibis table expression.
        table_key
            Optional table key for Pandera validation.

        Returns
        -------
        list[RowDict]
            List of row dictionaries.
        """
        df = self._ibis_to_df(expr, table_key)
        sanitized = df.astype("object").where(pd.notna(df), None)
        return sanitized.to_dict(orient="records")

    def _ibis_to_one(
        self,
        expr: it.Table,
        table_key: str | None = None,
    ) -> RowDict | None:
        """
        Execute an Ibis expression and return the first row.

        Parameters
        ----------
        expr
            Ibis table expression.
        table_key
            Optional table key for Pandera validation.

        Returns
        -------
        RowDict | None
            First row as dict, or None if no rows.
        """
        dicts = self._ibis_to_dicts(expr.limit(1), table_key)
        return dicts[0] if dicts else None

    def _ibis_exists(
        self,
        expr: it.Table,
    ) -> bool:
        """
        Check if at least one row exists in the Ibis expression result.

        Execute an Ibis expression and return True if at least one row
        is present.

        Parameters
        ----------
        expr
            Ibis table expression (typically with filters applied).

        Returns
        -------
        bool
            True if at least one row matches the expression.
        """
        _ = self

        limited = expr.limit(1)
        df = pd.DataFrame(limited.execute())
        return len(df) > 0

    def _ibis_paginated(
        self,
        expr: it.Table,
        *,
        limit: int,
        table_key: str | None = None,
    ) -> PagedResult[RowDict]:
        """
        Execute an Ibis expression with pagination and truncation detection.

        Fetch limit+1 rows to detect if more data exists beyond the page,
        returning a PagedResult with truncation metadata.

        Parameters
        ----------
        expr
            Ibis table expression (filters/ordering should already be applied).
        limit
            Maximum rows to return.
        table_key
            Optional table key for Pandera validation.

        Returns
        -------
        PagedResult[RowDict]
            Paginated result with truncation metadata.
        """
        _ = self

        fetch_limit = limit + 1
        limited_expr = expr.limit(fetch_limit)
        df = pd.DataFrame(limited_expr.execute())

        if table_key:
            df = validate_df(table_key, df)

        all_rows: list[RowDict] = df.to_dict(orient="records")
        truncated = len(all_rows) > limit
        items = all_rows[:limit]

        return PagedResult(
            items=items,
            total=None,
            limit=limit,
            offset=0,
            truncated=truncated,
        )

    @staticmethod
    def _validated_records(
        table_key: str,
        expr: it.Table,
    ) -> list[RowDict]:
        """
        Execute an Ibis expression and return validated row dictionaries.

        This method combines execution, Pandera validation, and null normalization
        into a single operation. Use this when you need validated records with
        consistent null handling.

        Parameters
        ----------
        table_key
            Dataset key used for Pandera schema lookup.
        expr
            Ibis table expression to execute.

        Returns
        -------
        list[RowDict]
            Validated records with ``None`` substituted for missing values.
        """
        df = pd.DataFrame(expr.execute())
        validated = validate_df(table_key, df)
        sanitized = validated.astype("object").where(pd.notna(validated), None)
        return sanitized.to_dict(orient="records")
