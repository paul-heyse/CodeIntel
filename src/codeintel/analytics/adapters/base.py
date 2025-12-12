"""Base classes and protocols for analytics adapters.

This module provides the foundation for persistence adapters that handle
database I/O for analytics modules.

Architecture
------------
The adapter hierarchy separates concerns for different I/O patterns:

- `InputAdapter[T]`: Load input data for computation
- `OutputAdapter[T]`: Load/persist output rows
- `ComputeAdapter[InputT, OutputT]`: Combined pattern for adapters that
  load source data of one type and persist results of another type

Example
-------
>>> class MyAdapter(ComputeAdapter[SourceRow, ResultRow]):
...     def load_inputs(self) -> Iterator[SourceRow]:
...         return self._load_source_data()
...
...     def load(self) -> Iterator[ResultRow]:
...         return iter([])
...
...     def persist(self, rows: Sequence[ResultRow]) -> int:
...         return self._write_results(rows)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend
from codeintel.storage.gateway.protocol import DuckDBError

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeleteScope:
    """Specification for scoped deletion before insert.

    Attributes
    ----------
    repo
        Repository identifier for the deletion scope.
    commit
        Commit hash for the deletion scope.
    columns
        Optional explicit column names for the WHERE clause.
    """

    repo: str
    commit: str
    columns: tuple[str, ...] | None = None


class InputAdapter[InputT](ABC):
    """Abstract base for adapters that load input data for computation.

    Use this when an adapter needs to load source data (InputT) that differs
    from the output row type.

    Type Parameters
    ---------------
    InputT
        The type of input data loaded for computation.
    """

    @abstractmethod
    def load_inputs(self) -> Iterator[InputT]:
        """Load input data for computation.

        Returns
        -------
        Iterator[InputT]
            Iterator over input data items.
        """
        ...


class OutputAdapter[OutputT](ABC):
    """Abstract base for adapters that load and persist output rows.

    Type Parameters
    ---------------
    OutputT
        The output row type this adapter works with.
    """

    @abstractmethod
    def load_outputs(self) -> Iterator[OutputT]:
        """Load existing output rows from the database.

        Returns
        -------
        Iterator[OutputT]
            Iterator over existing output rows.
        """
        ...

    @abstractmethod
    def persist(self, rows: Sequence[OutputT]) -> int:
        """Persist computed output rows to the database.

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


class ComputeAdapter[InputT, OutputT](InputAdapter[InputT], OutputAdapter[OutputT], ABC):
    """Abstract adapter that loads input data and persists output rows.

    This is the standard pattern for analytics adapters that:
    1. Load source data (InputT) for computation
    2. Persist results (OutputT) to the database

    The separation of input/output types makes the adapter's role explicit
    and avoids type override issues when InputT != OutputT.

    Type Parameters
    ---------------
    InputT
        The type of input data loaded for computation.
    OutputT
        The type of output rows to persist.

    Example
    -------
    >>> class MetricsAdapter(ComputeAdapter[FunctionGoid, MetricsRow]):
    ...     def load_inputs(self) -> Iterator[FunctionGoid]:
    ...         return self._goid_loader.iter_goids()
    ...
    ...     def load_outputs(self) -> Iterator[MetricsRow]:
    ...         return iter([])
    ...
    ...     def persist(self, rows: Sequence[MetricsRow]) -> int:
    ...         return self._insert_rows(rows)
    """


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


class SimpleBatchAdapter[RowT](ABC):
    """Adapter for simple batched write operations without load requirements.

    Use this base class for adapters that only need to write rows without
    implementing the full AnalyticsAdapter interface.

    Type Parameters
    ---------------
    RowT
        The row type this adapter works with.
    """

    @property
    @abstractmethod
    def table_name(self) -> str:
        """Return the target table name.

        Returns
        -------
        str
            Fully qualified table name (e.g., 'analytics.my_table').
        """
        ...

    @abstractmethod
    def insert_rows(self, gateway: StorageGateway, rows: Sequence[RowT]) -> int:
        """Insert rows into the table.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        rows
            Rows to insert.

        Returns
        -------
        int
            Number of rows inserted.
        """
        ...

    def execute_delete(self, gateway: StorageGateway, scope: DeleteScope) -> int:
        """Delete rows matching the scope using the DuckDBPolicyBackend.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        scope
            Deletion scope specifying repo/commit.

        Returns
        -------
        int
            Number of rows deleted.
        """
        count = self._count_rows_for_scope(gateway, scope)

        backend = DuckDBPolicyBackend(gateway)
        backend.delete_for_snapshot(self.table_name, repo=scope.repo, commit=scope.commit)

        return count

    def _count_rows_for_scope(self, gateway: StorageGateway, scope: DeleteScope) -> int:
        """Count rows matching the scope before deletion.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        scope
            Deletion scope specifying repo/commit.

        Returns
        -------
        int
            Number of matching rows.
        """
        try:
            tbl = gateway.ibis.table(self.table_name)
            repo_filter = cast("Any", tbl.repo == scope.repo)
            commit_filter = cast("Any", tbl.commit == scope.commit)
            return cast("int", tbl.filter(repo_filter & commit_filter).count().execute())
        except DuckDBError:
            return 0


class BatchAdapter[RowT](AnalyticsAdapter[RowT], ABC):
    """Abstract adapter that supports batched persistence operations.

    Extends the base adapter with methods for batched writes and
    configurable delete scoping. Subclasses must implement `load`, `persist`,
    and `table_name`.
    """

    @property
    @abstractmethod
    def table_name(self) -> str:
        """Return the target table name.

        Returns
        -------
        str
            Fully qualified table name.
        """
        ...

    def delete_scope(self) -> DeleteScope:
        """Return the default delete scope for this adapter.

        Returns
        -------
        DeleteScope
            Scope specifying repo/commit-based deletion.
        """
        return DeleteScope(repo=self.repo, commit=self.commit)

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
        """Delete existing rows for this snapshot using the DuckDBPolicyBackend."""
        scope = self.delete_scope()
        backend = DuckDBPolicyBackend(self._gateway)
        backend.delete_for_snapshot(self.table_name, repo=scope.repo, commit=scope.commit)


__all__ = [
    "AnalyticsAdapter",
    "BatchAdapter",
    "ComputeAdapter",
    "DeleteScope",
    "InputAdapter",
    "OutputAdapter",
    "SimpleBatchAdapter",
]
