"""Base execution context providing core context infrastructure.

This module defines `BaseContext`, the foundational context type for all
CodeIntel execution contexts. It provides run identity, gateway access,
and snapshot reference that all domain-specific contexts inherit.

The context hierarchy is:

    BaseContext
        └── PluginExecutionContext
                ├── IngestExecutionContext
                ├── GraphPluginExecutionContext
                └── AnalyticsExecutionContext

BaseContext provides the minimal contract that all execution paths share:
- Run identity tracking (via RunContext)
- Database access (via StorageGateway)
- Repository snapshot reference (via SnapshotRef)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.execution.context import RunContext
    from codeintel.storage.gateway import StorageGateway


@dataclass
class BaseContext:
    """Root context providing run identity, gateway, and snapshot access.

    This is the foundational context type from which all domain-specific
    execution contexts inherit. It ensures consistent access to:

    - Run metadata (run_id, kind, trigger) via RunContext
    - Database connection via StorageGateway
    - Repository snapshot reference (repo, commit, root path)

    Parameters
    ----------
    run_context
        Run metadata including unique ID, kind, and trigger source.
    gateway
        Database gateway for storage operations.
    snapshot
        Repository snapshot reference containing repo, commit, and root path.

    Examples
    --------
    >>> from codeintel.core.execution.orchestrator import new_run_context
    >>> from codeintel.config.primitives import SnapshotRef
    >>> from pathlib import Path
    >>>
    >>> run = new_run_context("full", "cli", snapshot)
    >>> ctx = BaseContext(run_context=run, gateway=gateway, snapshot=snapshot)
    >>> ctx.run_id
    'ci-...'
    >>> ctx.repo
    'org/repo'
    """

    run_context: RunContext
    gateway: StorageGateway
    snapshot: SnapshotRef

    @property
    def run_id(self) -> str:
        """Return unique identifier for this execution run.

        Returns
        -------
        str
            Run identifier from the underlying RunContext.
        """
        return self.run_context.run_id

    @property
    def repo(self) -> str:
        """Return repository slug from the snapshot reference.

        Returns
        -------
        str
            Repository slug (e.g., 'org/repo').
        """
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Return commit identifier from the snapshot reference.

        Returns
        -------
        str
            Commit SHA or identifier.
        """
        return self.snapshot.commit


__all__ = [
    "BaseContext",
]
