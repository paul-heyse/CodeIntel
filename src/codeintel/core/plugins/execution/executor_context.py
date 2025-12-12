"""Base executor context for plugin executors.

This module defines the base executor context dataclass that provides
common fields needed by all domain-specific executors. This is distinct
from PluginExecutionContext which is passed to individual plugins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.core.execution.telemetry import get_runtime_telemetry

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.execution import RunContext
    from codeintel.core.execution.telemetry import RuntimeTelemetry
    from codeintel.storage.gateway import StorageGateway


def _default_telemetry_factory() -> RuntimeTelemetry:
    """Return the default runtime telemetry singleton.

    Returns
    -------
    RuntimeTelemetry
        Default telemetry instance.
    """
    return get_runtime_telemetry()


@dataclass
class BaseExecutorContext:
    """Common executor context for all domains.

    Provide the base fields needed by all executor implementations.
    Domain-specific executor contexts extend this with additional
    fields as needed.

    This context is for the executor itself, not the individual plugins.
    Plugins receive PluginExecutionContext which is built by the executor.

    Attributes
    ----------
    gateway
        Storage gateway for database access.
    snapshot
        Repository snapshot reference.
    run_context
        Optional unified run context for cross-engine correlation.
    telemetry
        Runtime telemetry for spans and metrics.

    Examples
    --------
    >>> from codeintel.storage.gateway import StorageGateway
    >>> from codeintel.config.primitives import SnapshotRef
    >>> from pathlib import Path
    >>>
    >>>
    >>>
    """

    gateway: StorageGateway
    snapshot: SnapshotRef
    run_context: RunContext | None = None
    telemetry: RuntimeTelemetry = field(default_factory=_default_telemetry_factory)

    @property
    def repo(self) -> str:
        """Repository slug from the snapshot reference.

        Returns
        -------
        str
            Repository identifier.
        """
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier from the snapshot reference.

        Returns
        -------
        str
            Commit hash.
        """
        return self.snapshot.commit

    @property
    def effective_run_id(self) -> str | None:
        """Get run ID from run_context if present.

        Returns
        -------
        str | None
            Run ID from run_context, or None if not set.
        """
        if self.run_context is not None:
            return self.run_context.run_id
        return None


__all__ = [
    "BaseExecutorContext",
]
