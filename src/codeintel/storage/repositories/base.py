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
    def list_X_paginated(self, *, limit: int) -> PagedResult[RowDict]: ...

Existence check:
    def has_X(self, id: int) -> bool: ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa

from codeintel.core.repository import PagedResult
from codeintel.storage.duckdb_types import (
    ColumnExpression,
    ConstantExpression,
    DuckDBCatalogException,
    Expression,
)
from codeintel.storage.query_results import records_from_arrow_table
from codeintel.storage.validation.pandera_df import validate_df

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.storage.gateway import DuckDBConnection, DuckDBRelation, StorageGateway

RowDict = dict[str, object]


def _combine_predicates(predicates: Sequence[Expression]) -> Expression | None:
    if not predicates:
        return None
    combined = predicates[0]
    for predicate in predicates[1:]:
        combined &= predicate
    return combined


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

    def _relation(self, table_key: str) -> DuckDBRelation:
        """
        Return a relation scoped to repo/commit when columns exist.

        Parameters
        ----------
        table_key
            Fully qualified table name (e.g., "core.goids").

        Returns
        -------
        DuckDBRelation
            Relation scoped to the repository snapshot when applicable.

        Raises
        ------
        DuckDBCatalogException
            If the requested table/view does not exist and cannot be created.
        """
        try:
            relation = self.gateway.relation_from_table_key(table_key)
        except DuckDBCatalogException:
            if table_key.startswith("docs."):
                self.gateway.policy.ensure_all_views(overwrite=True, strict=False)
                relation = self.gateway.relation_from_table_key(table_key)
            else:
                raise
        columns = set(relation.columns)
        if "repo" in columns and "commit" in columns:
            predicate = (ColumnExpression("repo") == ConstantExpression(self.repo)) & (
                ColumnExpression("commit") == ConstantExpression(self.commit)
            )
            relation = relation.filter(predicate)
        return relation

    @staticmethod
    def _relation_to_arrow(
        relation: DuckDBRelation,
        *,
        table_key: str | None = None,
    ) -> pa.Table:
        """Return an Arrow table from a relation, optionally validated.

        Returns
        -------
        pa.Table
            Arrow table fetched (and validated when table_key is provided).
        """
        table = relation.fetch_arrow_table()
        if table_key is None or BaseRepository._has_nested_arrow_columns(table):
            return table
        df = table.to_pandas()
        validated = validate_df(table_key, df)
        return pa.Table.from_pandas(validated, preserve_index=False)

    @staticmethod
    def _relation_to_df(
        relation: DuckDBRelation,
        *,
        table_key: str | None = None,
    ) -> pd.DataFrame:
        """Return a DataFrame from a relation, optionally validated.

        Returns
        -------
        pd.DataFrame
            DataFrame fetched (and validated when table_key is provided).
        """
        table = relation.fetch_arrow_table()
        df = table.to_pandas()
        if table_key:
            return validate_df(table_key, df)
        return df

    def _relation_to_dicts(
        self,
        relation: DuckDBRelation,
        table_key: str | None = None,
    ) -> list[RowDict]:
        table = self._relation_to_arrow(relation, table_key=table_key)
        return records_from_arrow_table(table)

    def _relation_to_one(
        self,
        relation: DuckDBRelation,
        table_key: str | None = None,
    ) -> RowDict | None:
        dicts = self._relation_to_dicts(relation.limit(1), table_key)
        return dicts[0] if dicts else None

    @staticmethod
    def _relation_exists(relation: DuckDBRelation) -> bool:
        """Check if at least one row exists in the relation result.

        Returns
        -------
        bool
            True when the relation yields at least one row.
        """
        return relation.limit(1).fetchone() is not None

    def _relation_paginated(
        self,
        relation: DuckDBRelation,
        *,
        limit: int,
        table_key: str | None = None,
    ) -> PagedResult[RowDict]:
        """
        Execute a relation with pagination and truncation detection.

        Fetch limit+1 rows to detect if more data exists beyond the page,
        returning a PagedResult with truncation metadata.

        Returns
        -------
        PagedResult[RowDict]
            Page of results with truncation metadata.
        """
        fetch_limit = limit + 1
        limited = relation.limit(fetch_limit)
        table = self._relation_to_arrow(limited, table_key=table_key)
        all_rows = records_from_arrow_table(table)
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
        relation: DuckDBRelation,
    ) -> list[RowDict]:
        """Execute a relation and return validated row dictionaries.

        Returns
        -------
        list[RowDict]
            Validated records from the relation.
        """
        table = BaseRepository._relation_to_arrow(relation, table_key=table_key)
        return records_from_arrow_table(table)

    @staticmethod
    def _has_nested_arrow_columns(table: pa.Table) -> bool:
        return any(pa.types.is_nested(field.type) for field in table.schema)

    @staticmethod
    def _predicate_eq(column: str, value: object) -> Expression:
        return ColumnExpression(column) == BaseRepository._constant_expression(value)

    @staticmethod
    def _predicate_ge(column: str, value: object) -> Expression:
        return ColumnExpression(column) >= BaseRepository._constant_expression(value)

    @staticmethod
    def _constant_expression(value: object) -> Expression:
        return ConstantExpression(value)

    @staticmethod
    def _apply_predicates(
        relation: DuckDBRelation,
        predicates: Sequence[Expression],
    ) -> DuckDBRelation:
        combined = _combine_predicates(predicates)
        if combined is None:
            return relation
        return relation.filter(combined)
