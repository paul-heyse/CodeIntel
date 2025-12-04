"""Adapters for entrypoints analytics persistence.

This module provides adapters for persisting entrypoint detection
results to DuckDB.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

from codeintel.analytics.adapters.base import BatchAdapter
from codeintel.config.datasets import load_columns_by_table, serialize_row
from codeintel.ingestion.adapters import IngestStorageService
from codeintel.storage.sql_helpers import ensure_schema

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


class EntrypointsAdapter(BatchAdapter[dict[str, Any]]):
    """Adapter for analytics.entrypoints table.

    Handle persisting entrypoint detection results.
    """

    def __init__(
        self,
        gateway: StorageGateway,
        snapshot: SnapshotRef,
        *,
        timestamp: datetime | None = None,
    ) -> None:
        """Initialize the adapter.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        snapshot
            Repository snapshot reference.
        timestamp
            Optional timestamp for created_at field.
        """
        super().__init__(gateway, snapshot)
        self._timestamp = timestamp

    @property
    def table_name(self) -> str:
        """Return the target table name."""
        return "analytics.entrypoints"

    def load(self) -> Iterator[dict[str, Any]]:
        """Raise NotImplementedError as entrypoints are computed not loaded.

        Raises
        ------
        NotImplementedError
            This adapter is write-only.
        """
        message = "EntrypointsAdapter does not support loading"
        raise NotImplementedError(message)

    def persist(self, rows: Sequence[dict[str, Any]]) -> int:
        """Persist entrypoint rows.

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
        self._delete_existing()

        columns = load_columns_by_table().get(self.table_name, [])
        tuple_rows = [serialize_row(row, columns) for row in rows]

        storage_service = IngestStorageService.from_gateway(self._gateway)
        storage_service.run_batch(
            self.table_name,
            tuple_rows,
            delete_params=[self.repo, self.commit],
            scope=f"{self.repo}@{self.commit}",
        )

        log.info(
            "Persisted %d entrypoint rows for %s@%s",
            len(rows),
            self.repo,
            self.commit,
        )
        return len(rows)


class EntrypointTestsAdapter(BatchAdapter[dict[str, Any]]):
    """Adapter for analytics.entrypoint_tests table.

    Handle persisting entrypoint-to-test mappings.
    """

    def __init__(
        self,
        gateway: StorageGateway,
        snapshot: SnapshotRef,
        *,
        timestamp: datetime | None = None,
    ) -> None:
        """Initialize the adapter.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        snapshot
            Repository snapshot reference.
        timestamp
            Optional timestamp for created_at field.
        """
        super().__init__(gateway, snapshot)
        self._timestamp = timestamp

    @property
    def table_name(self) -> str:
        """Return the target table name."""
        return "analytics.entrypoint_tests"

    def load(self) -> Iterator[dict[str, Any]]:
        """Raise NotImplementedError as mappings are computed not loaded.

        Raises
        ------
        NotImplementedError
            This adapter is write-only.
        """
        message = "EntrypointTestsAdapter does not support loading"
        raise NotImplementedError(message)

    def persist(self, rows: Sequence[dict[str, Any]]) -> int:
        """Persist entrypoint test mapping rows.

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
        self._delete_existing()

        columns = load_columns_by_table().get(self.table_name, [])
        tuple_rows = [serialize_row(row, columns) for row in rows]

        storage_service = IngestStorageService.from_gateway(self._gateway)
        storage_service.run_batch(
            self.table_name,
            tuple_rows,
            delete_params=[self.repo, self.commit],
            scope=f"{self.repo}@{self.commit}",
        )

        log.info(
            "Persisted %d entrypoint test rows for %s@%s",
            len(rows),
            self.repo,
            self.commit,
        )
        return len(rows)


__all__ = [
    "EntrypointTestsAdapter",
    "EntrypointsAdapter",
]
