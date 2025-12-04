"""Adapters for semantic roles analytics persistence.

This module provides adapters for persisting semantic role classification
results to DuckDB.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from datetime import datetime
from typing import TYPE_CHECKING

from codeintel.analytics.adapters.base import BatchAdapter
from codeintel.ingestion.adapters import IngestStorageService
from codeintel.storage.sql_builder import ensure_schema

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


class SemanticRolesFunctionsAdapter(BatchAdapter[tuple[object, ...]]):
    """Adapter for analytics.semantic_roles_functions table.

    Handle persisting function semantic role classifications.
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
        return "analytics.semantic_roles_functions"

    def load(self) -> Iterator[tuple[object, ...]]:
        """Raise NotImplementedError as roles are computed not loaded.

        Raises
        ------
        NotImplementedError
            This adapter is write-only.
        """
        message = "SemanticRolesFunctionsAdapter does not support loading"
        raise NotImplementedError(message)

    def persist(self, rows: Sequence[tuple[object, ...]]) -> int:
        """Persist function semantic role rows.

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

        storage_service = IngestStorageService.from_gateway(self._gateway)
        storage_service.run_batch(
            self.table_name,
            list(rows),
            delete_params=[self.repo, self.commit],
            scope=f"{self.repo}@{self.commit}",
        )

        log.info(
            "Persisted %d function semantic role rows for %s@%s",
            len(rows),
            self.repo,
            self.commit,
        )
        return len(rows)


class SemanticRolesModulesAdapter(BatchAdapter[tuple[object, ...]]):
    """Adapter for analytics.semantic_roles_modules table.

    Handle persisting module semantic role classifications.
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
        return "analytics.semantic_roles_modules"

    def load(self) -> Iterator[tuple[object, ...]]:
        """Raise NotImplementedError as roles are computed not loaded.

        Raises
        ------
        NotImplementedError
            This adapter is write-only.
        """
        message = "SemanticRolesModulesAdapter does not support loading"
        raise NotImplementedError(message)

    def persist(self, rows: Sequence[tuple[object, ...]]) -> int:
        """Persist module semantic role rows.

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

        storage_service = IngestStorageService.from_gateway(self._gateway)
        storage_service.run_batch(
            self.table_name,
            list(rows),
            delete_params=[self.repo, self.commit],
            scope=f"{self.repo}@{self.commit}",
        )

        log.info(
            "Persisted %d module semantic role rows for %s@%s",
            len(rows),
            self.repo,
            self.commit,
        )
        return len(rows)


__all__ = [
    "SemanticRolesFunctionsAdapter",
    "SemanticRolesModulesAdapter",
]
