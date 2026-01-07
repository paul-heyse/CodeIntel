"""Shared storage context helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.core.queries.context import QueryContext
from codeintel.core.schemas.provider import SchemaProvider

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway.protocol import StorageGateway


@dataclass(frozen=True, slots=True)
class StorageContext:
    """Bundle storage gateway state with snapshot and policy metadata.

    Parameters
    ----------
    gateway
        Storage gateway providing DuckDB access.
    snapshot
        Optional snapshot reference used for repo/commit scoping.
    dataset_root
        Optional dataset root directory override.
    schema_provider
        Optional schema provider override for contract resolution.
    query_context
        Optional query context for SQL ingress policies.
    """

    gateway: StorageGateway
    snapshot: SnapshotRef | None = None
    dataset_root: Path | None = None
    schema_provider: SchemaProvider | None = None
    query_context: QueryContext | None = None

    @property
    def repo(self) -> str:
        """Return repository identifier from snapshot.

        Raises
        ------
        ValueError
            If the snapshot reference is missing.
        """
        if self.snapshot is None:
            msg = "StorageContext.snapshot is required to access repo"
            raise ValueError(msg)
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Return commit identifier from snapshot.

        Raises
        ------
        ValueError
            If the snapshot reference is missing.
        """
        if self.snapshot is None:
            msg = "StorageContext.snapshot is required to access commit"
            raise ValueError(msg)
        return self.snapshot.commit

    def require_snapshot(self) -> SnapshotRef:
        """Return snapshot or raise if missing.

        Returns
        -------
        SnapshotRef
            The snapshot reference attached to the context.

        Raises
        ------
        ValueError
            If the snapshot reference is missing.
        """
        if self.snapshot is None:
            msg = "StorageContext.snapshot is required for snapshot-scoped operations"
            raise ValueError(msg)
        return self.snapshot

    def with_snapshot(self, snapshot: SnapshotRef) -> StorageContext:
        """Return a copy of the context with a new snapshot.

        Returns
        -------
        StorageContext
            A new context with the provided snapshot reference.
        """
        return replace(self, snapshot=snapshot)

    @property
    def dataset_root_dir(self) -> Path | None:
        """Return dataset root directory (explicit override first)."""
        if self.dataset_root is not None:
            return self.dataset_root
        config = getattr(self.gateway, "config", None)
        if config is None:
            return None
        return getattr(config, "dataset_root_dir", None)


__all__ = ["StorageContext"]
