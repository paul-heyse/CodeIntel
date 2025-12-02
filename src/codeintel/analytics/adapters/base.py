"""Base classes and protocols for analytics adapters.

This module provides the foundation for persistence adapters that handle
database I/O for analytics modules.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeleteScope:
    """Specification for scoped deletion before insert.

    Attributes
    ----------
    params
        Parameters for the delete query (typically repo, commit).
    columns
        Optional explicit column names for the WHERE clause.
    """

    params: Sequence[object]
    columns: tuple[str, ...] | None = None


class AnalyticsAdapter[RowT](ABC):
    """Abstract base class for analytics data adapters.

    Adapters encapsulate database I/O for specific analytics domains.
    They provide methods for loading source data and persisting results.

    Type Parameters
    ---------------
    RowT
        The row type this adapter works with.
    """

    def __init__(
        self,
        gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Initialize the adapter with storage gateway and snapshot.

        Parameters
        ----------
        gateway
            Storage gateway providing database access.
        snapshot
            Repository snapshot reference (repo, commit, repo_root).
        """
        self._gateway = gateway
        self._snapshot = snapshot

    @property
    def gateway(self) -> StorageGateway:
        """Return the storage gateway."""
        return self._gateway

    @property
    def snapshot(self) -> SnapshotRef:
        """Return the snapshot reference."""
        return self._snapshot

    @property
    def repo(self) -> str:
        """Return the repository identifier."""
        return self._snapshot.repo

    @property
    def commit(self) -> str:
        """Return the commit identifier."""
        return self._snapshot.commit

    @abstractmethod
    def load(self) -> Iterator[RowT]:
        """Load source data from the database.

        Returns
        -------
        Iterator[RowT]
            Iterator over loaded rows.
        """
        ...

    @abstractmethod
    def persist(self, rows: Sequence[RowT]) -> int:
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
        ...


class BatchAdapter[RowT](AnalyticsAdapter[RowT]):
    """Adapter that supports batched persistence operations.

    Extends the base adapter with methods for batched writes and
    configurable delete scoping.
    """

    @property
    @abstractmethod
    def table_name(self) -> str:
        """Return the target table name."""
        ...

    def delete_scope(self) -> DeleteScope:
        """Return the default delete scope for this adapter.

        Returns
        -------
        DeleteScope
            Scope specifying repo/commit-based deletion.
        """
        return DeleteScope(params=[self.repo, self.commit])

    def persist_batch(
        self,
        rows: Sequence[RowT],
        *,
        delete_before: bool = True,
    ) -> int:
        """Persist a batch of rows with optional pre-deletion.

        Parameters
        ----------
        rows
            Rows to persist.
        delete_before
            Whether to delete existing rows before inserting.

        Returns
        -------
        int
            Number of rows persisted.
        """
        if delete_before:
            self._delete_existing()

        if not rows:
            return 0

        return self.persist(rows)

    def _delete_existing(self) -> None:
        """Delete existing rows for this snapshot."""
        scope = self.delete_scope()
        table = self.table_name
        # Table name comes from subclass definition (trusted)
        query = f"DELETE FROM {table} WHERE repo = ? AND commit = ?"  # noqa: S608
        self._gateway.con.execute(query, list(scope.params))


__all__ = [
    "AnalyticsAdapter",
    "BatchAdapter",
    "DeleteScope",
]
