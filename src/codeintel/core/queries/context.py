"""Query context helpers for safe query operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef


@dataclass(frozen=True, slots=True)
class QueryContext:
    """Bundle snapshot identity and SQL ingress policy configuration.

    Parameters
    ----------
    snapshot
        Optional snapshot reference used for repo/commit scoping.
    allowed_tables
        Optional allowlist of table names or qualified table keys.
    allowed_schemas
        Optional allowlist of schemas for SQL ingress.
    allowed_functions
        Optional allowlist of functions for SQL ingress.
    deny_functions
        Optional denylist of functions for SQL ingress.
    allow_unqualified_tables
        Whether unqualified tables are allowed in SQL ingress.
    allow_cross_database_references
        Whether cross-database references are allowed in SQL ingress.
    enforce_safe_sql
        Whether to enforce the select-only SQL perimeter.
    """

    snapshot: SnapshotRef | None = None
    allowed_tables: frozenset[str] | None = None
    allowed_schemas: frozenset[str] | None = None
    allowed_functions: frozenset[str] | None = None
    deny_functions: frozenset[str] = frozenset()
    allow_unqualified_tables: bool = True
    allow_cross_database_references: bool = False
    enforce_safe_sql: bool = True

    @property
    def repo(self) -> str:
        """Return repository identifier from the snapshot.

        Raises
        ------
        ValueError
            If the snapshot reference is missing.
        """
        if self.snapshot is None:
            msg = "QueryContext.snapshot is required to access repo"
            raise ValueError(msg)
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Return commit identifier from the snapshot.

        Raises
        ------
        ValueError
            If the snapshot reference is missing.
        """
        if self.snapshot is None:
            msg = "QueryContext.snapshot is required to access commit"
            raise ValueError(msg)
        return self.snapshot.commit


__all__ = ["QueryContext"]
