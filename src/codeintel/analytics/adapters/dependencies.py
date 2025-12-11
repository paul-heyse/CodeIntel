"""Persistence adapter for dependency analytics.

This module provides database operations for storing and retrieving
dependency analysis results.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from codeintel.analytics.adapters.base import BatchAdapter
from codeintel.storage.sql.builder import ensure_schema

log = logging.getLogger(__name__)

# Pre-built SQL statements for INSERT (bulk inserts stay as executemany)
_INSERT_DEPENDENCY_CALL = """
INSERT INTO analytics.external_dependency_calls (
    repo, commit, dep_id, library, service_name,
    function_goid_h128, function_urn, rel_path, module, qualname,
    callsite_count, modes, evidence_json, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_DEPENDENCY_AGGREGATE = """
INSERT INTO analytics.external_dependencies (
    repo, commit, dep_id, library, service_name, category, language,
    severity, criticality, risk_score,
    function_count, callsite_count, modules_json, usage_modes,
    config_keys, risk_level, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_SQL: dict[str, str] = {
    "analytics.external_dependency_calls": _INSERT_DEPENDENCY_CALL,
    "analytics.external_dependencies": _INSERT_DEPENDENCY_AGGREGATE,
}


def _get_insert_sql(table_name: str) -> str:
    """Get insert SQL for a table.

    Parameters
    ----------
    table_name
        Table name.

    Returns
    -------
    str
        Insert SQL statement.

    Raises
    ------
    ValueError
        If table name is not in the mapping.
    """
    if table_name not in _INSERT_SQL:
        msg = f"No insert SQL for table: {table_name}"
        raise ValueError(msg)
    return _INSERT_SQL[table_name]


@dataclass(frozen=True)
class DependencyCallRow:
    """Row for external_dependency_calls table.

    Attributes
    ----------
    repo
        Repository identifier.
    commit
        Commit hash.
    dep_id
        Unique dependency identifier.
    library
        Library name.
    service_name
        Human-readable service name.
    function_goid_h128
        Function global ID (as Decimal for DuckDB hugeint).
    function_urn
        Function URN.
    rel_path
        Relative source file path.
    module
        Module name.
    qualname
        Fully qualified function name.
    callsite_count
        Number of call sites.
    modes
        List of usage modes.
    evidence_json
        Evidence data as list of dicts.
    created_at
        Row creation timestamp.
    """

    repo: str
    commit: str
    dep_id: str
    library: str
    service_name: str
    function_goid_h128: Decimal
    function_urn: str
    rel_path: str
    module: str
    qualname: str
    callsite_count: int
    modes: list[str]
    evidence_json: list[dict[str, object]]
    created_at: datetime


@dataclass(frozen=True)
class DependencyAggregateRow:
    """Row for external_dependencies table.

    Attributes
    ----------
    repo
        Repository identifier.
    commit
        Commit hash.
    dep_id
        Unique dependency identifier.
    library
        Library name.
    service_name
        Human-readable service name.
    category
        Dependency category.
    language
        Programming language.
    severity
        Severity level.
    criticality
        Criticality score.
    risk_score
        Computed risk score.
    function_count
        Number of functions using this dependency.
    callsite_count
        Total call sites.
    modules_json
        List of modules using this dependency.
    usage_modes
        List of usage modes.
    config_keys
        List of related config keys.
    risk_level
        Risk level classification.
    created_at
        Row creation timestamp.
    """

    repo: str
    commit: str
    dep_id: str
    library: str
    service_name: str
    category: str | None
    language: str
    severity: str | None
    criticality: float | None
    risk_score: float | None
    function_count: int
    callsite_count: int
    modules_json: list[str]
    usage_modes: list[str]
    config_keys: list[str]
    risk_level: str
    created_at: datetime


class DependencyCallAdapter(BatchAdapter[DependencyCallRow]):
    """Adapter for external_dependency_calls table."""

    @property
    def table_name(self) -> str:
        """Return the target table name."""
        return "analytics.external_dependency_calls"

    def load(self) -> Iterator[DependencyCallRow]:
        """Load dependency call rows (not implemented for this adapter).

        Returns
        -------
        Iterator[DependencyCallRow]
            Empty iterator - this adapter is write-only.
        """
        log.debug("DependencyCallAdapter.load skipped for table %s", self.table_name)
        return iter(())

    def persist(self, rows: Sequence[DependencyCallRow]) -> int:
        """Persist computed rows to the database.

        Parameters
        ----------
        rows
            Rows to persist.

        Returns
        -------
        int
            Number of rows persisted.
        """
        if not rows:
            return 0

        ensure_schema(self._gateway.con, self.table_name)
        values = [
            (
                row.repo,
                row.commit,
                row.dep_id,
                row.library,
                row.service_name,
                row.function_goid_h128,
                row.function_urn,
                row.rel_path,
                row.module,
                row.qualname,
                row.callsite_count,
                row.modes,
                row.evidence_json,
                row.created_at,
            )
            for row in rows
        ]
        insert_sql = _get_insert_sql(self.table_name)
        self._gateway.con.executemany(insert_sql, values)
        return len(values)


class DependencyAggregateAdapter(BatchAdapter[DependencyAggregateRow]):
    """Adapter for external_dependencies table."""

    @property
    def table_name(self) -> str:
        """Return the target table name."""
        return "analytics.external_dependencies"

    def load(self) -> Iterator[DependencyAggregateRow]:
        """Load dependency aggregate rows (not implemented for this adapter).

        Returns
        -------
        Iterator[DependencyAggregateRow]
            Empty iterator - this adapter is write-only.
        """
        log.debug("DependencyAggregateAdapter.load skipped for table %s", self.table_name)
        return iter(())

    def persist(self, rows: Sequence[DependencyAggregateRow]) -> int:
        """Persist computed rows to the database.

        Parameters
        ----------
        rows
            Rows to persist.

        Returns
        -------
        int
            Number of rows persisted.
        """
        if not rows:
            return 0

        ensure_schema(self._gateway.con, self.table_name)
        values = [
            (
                row.repo,
                row.commit,
                row.dep_id,
                row.library,
                row.service_name,
                row.category,
                row.language,
                row.severity,
                row.criticality,
                row.risk_score,
                row.function_count,
                row.callsite_count,
                row.modules_json,
                row.usage_modes,
                row.config_keys,
                row.risk_level,
                row.created_at,
            )
            for row in rows
        ]
        insert_sql = _get_insert_sql(self.table_name)
        self._gateway.con.executemany(insert_sql, values)
        return len(values)


def compute_dep_id(repo: str, commit: str, library: str) -> str:
    """Compute unique dependency identifier.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit hash.
    library
        Library name.

    Returns
    -------
    str
        SHA-1 hash prefix as dependency ID.
    """
    raw = f"{repo}:{commit}:{library}"
    return hashlib.sha1(raw.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def to_decimal(value: int) -> Decimal:
    """Convert integer to Decimal for DuckDB hugeint.

    Parameters
    ----------
    value
        Integer value.

    Returns
    -------
    Decimal
        Decimal representation.
    """
    return Decimal(value)


__all__ = [
    "DependencyAggregateAdapter",
    "DependencyAggregateRow",
    "DependencyCallAdapter",
    "DependencyCallRow",
    "compute_dep_id",
    "to_decimal",
]
