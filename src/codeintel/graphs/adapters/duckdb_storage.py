"""DuckDB storage adapter implementing StoragePort.

This module provides a concrete implementation of StoragePort that
uses DuckDB via the StorageGateway for database operations.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.graphs.ports.storage import BatchResult, QueryResult
from codeintel.ingestion.adapters import IngestStorageService
from codeintel.storage.gateway import DuckDBConnection

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


@dataclass
class DuckDBStorageAdapter:
    """StoragePort implementation using DuckDB via StorageGateway.

    Attributes
    ----------
    gateway
        StorageGateway providing DuckDB access.
    _repo_root
        Repository root path for source file access.
    """

    gateway: StorageGateway
    _repo_root: Path = field(default_factory=Path)

    @classmethod
    def from_gateway(
        cls, gateway: StorageGateway, repo_root: Path | None = None
    ) -> DuckDBStorageAdapter:
        """Construct adapter using gateway config or provided root.

        Parameters
        ----------
        gateway
            Storage gateway providing DuckDB access.
        repo_root
            Optional repository root; defaults to current directory when unknown.

        Returns
        -------
        DuckDBStorageAdapter
            Configured adapter instance.
        """
        resolved_root = repo_root or Path()
        return cls(gateway=gateway, _repo_root=resolved_root)

    def execute(self, sql: str, params: Sequence[object] | None = None) -> DuckDBConnection:
        """Execute raw SQL against the underlying gateway connection.

        Parameters
        ----------
        sql
            SQL statement to execute.
        params
            Optional parameter values to bind.

        Returns
        -------
        DuckDBConnection
            DuckDB relation/connection result from execution.

        Notes
        -----
        This method intentionally propagates DuckDB exceptions so callers
        (and tests) can observe failures.
        """
        if params is None:
            return self.gateway.con.execute(sql)
        return self.gateway.con.execute(sql, list(params))

    def execute_query(
        self,
        sql: str,
        params: Sequence[object] | None = None,
    ) -> QueryResult:
        """Execute a SQL query and return results.

        Parameters
        ----------
        sql
            SQL query string.
        params
            Parameter values to bind.

        Returns
        -------
        QueryResult
            Query results with rows and metadata.
        """
        try:
            if params is not None:
                result = self.gateway.con.execute(sql, list(params))
            else:
                result = self.gateway.con.execute(sql)
            rows = result.fetchall()
            return QueryResult.from_rows([tuple(row) for row in rows])
        except (RuntimeError, OSError, TypeError, ValueError) as exc:
            log.warning("Query execution failed: %s", exc)
            return QueryResult.empty()

    def execute_mutation(
        self,
        sql: str,
        params: Sequence[object] | None = None,
    ) -> int:
        """Execute a SQL mutation statement.

        Parameters
        ----------
        sql
            SQL mutation statement.
        params
            Parameter values to bind.

        Returns
        -------
        int
            Number of rows affected.
        """
        try:
            if params is not None:
                result = self.gateway.con.execute(sql, list(params))
            else:
                result = self.gateway.con.execute(sql)
        except (RuntimeError, OSError, TypeError, ValueError) as exc:
            log.warning("Mutation execution failed: %s", exc)
            return 0
        else:
            # DuckDB returns rowcount for mutations
            if result.description:
                row = result.fetchone()
                if row is not None:
                    return int(row[0])
            return 0

    def run_batch(
        self,
        table: str,
        rows: Sequence[tuple[object, ...]],
        *,
        delete_params: Sequence[object] | None = None,
        scope: str | None = None,
    ) -> BatchResult:
        """Insert a batch of rows into a table.

        Parameters
        ----------
        table
            Target table name.
        rows
            Row tuples to insert.
        delete_params
            Optional pre-delete parameters.
        scope
            Scope identifier for logging.

        Returns
        -------
        BatchResult
            Batch operation result.
        """
        try:
            storage_service = IngestStorageService.from_gateway(self.gateway)
            storage_service.run_batch(
                table,
                list(rows),
                delete_params=list(delete_params) if delete_params else [],
                scope=scope or table,
            )
            return BatchResult.ok(table, len(rows))
        except (RuntimeError, OSError, TypeError, ValueError) as exc:
            log.warning("Batch insert to %s failed: %s", table, exc)
            return BatchResult.fail(table, str(exc))

    def read_source(self, rel_path: str) -> str | None:
        """Read source file contents.

        Parameters
        ----------
        rel_path
            Relative path to the source file.

        Returns
        -------
        str | None
            File contents if readable, None otherwise.
        """
        file_path = self._repo_root / rel_path
        try:
            return file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            log.debug("Failed to read %s: %s", file_path, exc)
            return None

    @property
    def repo_root(self) -> Path:
        """Repository root path.

        Returns
        -------
        Path
            Absolute path to the repository root.
        """
        return self._repo_root


__all__ = ["DuckDBStorageAdapter"]
