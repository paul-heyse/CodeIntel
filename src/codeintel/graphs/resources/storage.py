"""Storage resource provider.

This module provides a resource provider for storage operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from codeintel.graphs.ports.storage import BatchResult, QueryResult
from codeintel.ingestion.adapters import IngestStorageService

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


@dataclass
class StorageResource:
    """Resource provider for storage operations.

    Implements both ResourceProvider and StoragePort protocols,
    providing unified access to database operations.

    Attributes
    ----------
    gateway
        Storage gateway providing DuckDB access.
    _repo_root
        Repository root path.
    """

    RESOURCE_NAME: ClassVar[str] = "storage"

    gateway: StorageGateway
    _repo_root: Path

    @property
    def resource_name(self) -> str:
        """Resource identifier.

        Returns
        -------
        str
            Resource name.
        """
        return self.RESOURCE_NAME

    def get(self) -> StorageResource:
        """Get storage resource.

        Returns
        -------
        StorageResource
            Self, providing access to gateway and port methods.
        """
        return self

    def invalidate(self) -> None:
        """Invalidate any cached state.

        Storage doesn't cache, so this is a no-op.
        """

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

            row = result.fetchone()
            return row[0] if row else 0
        except (RuntimeError, OSError, TypeError, ValueError) as exc:
            log.warning("Mutation execution failed: %s", exc)
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


__all__ = ["StorageResource"]
