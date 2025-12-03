"""Storage service providing high-level batch operations.

This module provides a service class that wraps IngestStoragePort for
common batch operations with automatic schema management and logging.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.ingestion.adapters.duckdb_storage import DuckDBStorageAdapter
from codeintel.ingestion.ports.storage import BatchResult, IngestStoragePort

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


@dataclass
class IngestStorageService:
    """High-level storage operations for ingestion.

    Wraps IngestStoragePort to provide convenient batch operations
    with automatic schema management and logging.

    Attributes
    ----------
    storage
        The underlying storage port for database operations.
    """

    storage: IngestStoragePort

    def run_batch(
        self,
        table_key: str,
        rows: Sequence[Sequence[object]],
        *,
        delete_params: Sequence[object] | None = None,
        scope: str | None = None,
    ) -> BatchResult:
        """Write batch with optional pre-delete.

        Ensures schema exists, optionally deletes prior rows, and inserts
        the batch with structured logging.

        Parameters
        ----------
        table_key
            Registry table key (e.g., "core.ast_nodes").
        rows
            Row payload matching the prepared insert statement.
        delete_params
            Optional parameters for the delete statement when defined.
        scope
            Optional repo@commit string for structured logging.

        Returns
        -------
        BatchResult
            Summary of rows inserted and elapsed time.
        """
        self.storage.ensure_schema(table_key)

        if delete_params is not None:
            self.storage.delete_by_params(table_key, delete_params)

        return self.storage.write_batch(table_key, rows, scope=scope)

    @classmethod
    def from_gateway(cls, gateway: StorageGateway) -> IngestStorageService:
        """Create a service instance from a StorageGateway.

        This factory method creates the appropriate DuckDB storage adapter
        for the given gateway.

        Parameters
        ----------
        gateway
            StorageGateway providing DuckDB access.

        Returns
        -------
        IngestStorageService
            Service instance wrapping the storage adapter.
        """
        return cls(storage=DuckDBStorageAdapter(gateway))


__all__ = ["IngestStorageService"]
