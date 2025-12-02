"""Persistence adapter for dependency analytics.

This module provides database operations for storing and retrieving
dependency analysis results.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from codeintel.analytics.adapters.base import BatchAdapter, DeleteScope
from codeintel.storage.sql_helpers import ensure_schema

if TYPE_CHECKING:
    from codeintel.analytics.compute.dependencies.classification import LibraryPattern
    from codeintel.analytics.compute.dependencies.detection import DependencyCall
    from codeintel.storage.gateway import DuckDBConnection, StorageGateway

log = logging.getLogger(__name__)


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
    file_path
        Source file path.
    module
        Module name.
    function_qualname
        Fully qualified function name.
    callsite_count
        Number of call sites.
    modes
        List of usage modes.
    evidence
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
    file_path: str
    module: str
    function_qualname: str
    callsite_count: int
    modes: list[str]
    evidence: list[dict[str, object]]
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

    table_name = "analytics.external_dependency_calls"

    def delete_scope(self, gateway: StorageGateway, scope: DeleteScope) -> int:
        """Delete rows within scope.

        Parameters
        ----------
        gateway
            Storage gateway.
        scope
            Deletion scope.

        Returns
        -------
        int
            Number of rows deleted.
        """
        ensure_schema(gateway.con, self.table_name)
        result = gateway.con.execute(
            f"DELETE FROM {self.table_name} WHERE repo = ? AND commit = ?",
            [scope.repo, scope.commit],
        )
        return result.fetchone()[0] if result else 0

    def insert_rows(
        self,
        gateway: StorageGateway,
        rows: Sequence[DependencyCallRow],
    ) -> int:
        """Insert rows into table.

        Parameters
        ----------
        gateway
            Storage gateway.
        rows
            Rows to insert.

        Returns
        -------
        int
            Number of rows inserted.
        """
        if not rows:
            return 0

        ensure_schema(gateway.con, self.table_name)
        values = [
            (
                row.repo,
                row.commit,
                row.dep_id,
                row.library,
                row.service_name,
                row.function_goid_h128,
                row.function_urn,
                row.file_path,
                row.module,
                row.function_qualname,
                row.callsite_count,
                row.modes,
                row.evidence,
                row.created_at,
            )
            for row in rows
        ]
        gateway.con.executemany(
            f"""
            INSERT INTO {self.table_name} (
                repo, commit, dep_id, library, service_name,
                function_goid_h128, function_urn, file_path, module, function_qualname,
                callsite_count, modes, evidence, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )
        return len(values)


class DependencyAggregateAdapter(BatchAdapter[DependencyAggregateRow]):
    """Adapter for external_dependencies table."""

    table_name = "analytics.external_dependencies"

    def delete_scope(self, gateway: StorageGateway, scope: DeleteScope) -> int:
        """Delete rows within scope.

        Parameters
        ----------
        gateway
            Storage gateway.
        scope
            Deletion scope.

        Returns
        -------
        int
            Number of rows deleted.
        """
        ensure_schema(gateway.con, self.table_name)
        result = gateway.con.execute(
            f"DELETE FROM {self.table_name} WHERE repo = ? AND commit = ?",
            [scope.repo, scope.commit],
        )
        return result.fetchone()[0] if result else 0

    def insert_rows(
        self,
        gateway: StorageGateway,
        rows: Sequence[DependencyAggregateRow],
    ) -> int:
        """Insert rows into table.

        Parameters
        ----------
        gateway
            Storage gateway.
        rows
            Rows to insert.

        Returns
        -------
        int
            Number of rows inserted.
        """
        if not rows:
            return 0

        ensure_schema(gateway.con, self.table_name)
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
        gateway.con.executemany(
            f"""
            INSERT INTO {self.table_name} (
                repo, commit, dep_id, library, service_name, category, language,
                severity, criticality, risk_score,
                function_count, callsite_count, modules_json, usage_modes,
                config_keys, risk_level, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )
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

