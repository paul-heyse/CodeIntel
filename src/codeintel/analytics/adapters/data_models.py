"""Adapters for data model analytics persistence.

This module provides adapters for persisting data model usage
results to DuckDB.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from codeintel.analytics.adapters.base import BatchAdapter
from codeintel.config.datasets import load_columns_by_table, serialize_row
from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from datetime import datetime

    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


class DataModelUsageAdapter(BatchAdapter[dict[str, Any]]):
    """Adapter for analytics.data_model_usage table.

    Handle persisting data model usage patterns.
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
        return "analytics.data_model_usage"

    def load(self) -> Iterator[dict[str, Any]]:
        """Raise NotImplementedError as usage is computed not loaded.

        Raises
        ------
        NotImplementedError
            This adapter is write-only.
        """
        message = "DataModelUsageAdapter does not support loading"
        raise NotImplementedError(message)

    def persist(self, rows: Sequence[dict[str, Any]]) -> int:
        """Persist data model usage rows.

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

        columns = load_columns_by_table().get(self.table_name, [])
        tuple_rows = [serialize_row(row, columns) for row in rows]

        backend = DuckDBPolicyBackend(self._gateway)
        backend.delete_for_snapshot(self.table_name, repo=self.repo, commit=self.commit)
        backend.bulk_insert(
            self.table_name,
            tuple_rows,
            columns=columns,
        )

        log.info(
            "Persisted %d data model usage rows for %s@%s",
            len(rows),
            self.repo,
            self.commit,
        )
        return len(rows)


__all__ = [
    "DataModelUsageAdapter",
]
