"""Unified execution runtime for CodeIntel pipelines.

This package consolidates all runtime execution infrastructure:

Run Context & Identity
----------------------
- **RunContext**: Unified run metadata across all engines
- **RunKind**: Classification of run types (ingest, graphs, analytics, full)
- **TriggerKind**: How the run was triggered (cli, http, mcp, api)
- **new_run_id**: Generate unique run identifiers with prefixes

Error Handling
--------------
- **PluginFatalError**: Unrecoverable plugin failure
- **PluginSkippedError**: Plugin skipped due to missing prerequisites
- **PluginSkipRequestError**: Internal signal for skip requests
- **PluginTimeoutError**: Plugin execution timeout
- **PLUGIN_CATCHABLE_ERRORS**: Tuple of recoverable plugin errors

Singleton Patterns
------------------
Two patterns are available:

1. **SingletonHolder[T]** (from ``codeintel.core.singleton``):
   Use for registries that need ``reset()`` for testing.

2. **cached_singleton** (from this module):
   Use ``@lru_cache(maxsize=1)`` for simple singletons that don't need reset.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from codeintel.core.errors.execution import (
    PLUGIN_CATCHABLE_ERRORS,
    PluginFatalError,
    PluginSkippedError,
    PluginSkipRequestError,
    PluginTimeoutError,
)
from codeintel.core.execution.context import ExecutionContext, RunContext, RunKind, TriggerKind
from codeintel.core.execution.ids import (
    RUN_PREFIX_ANALYTICS,
    RUN_PREFIX_GRAPHS,
    RUN_PREFIX_INGEST,
    RUN_PREFIX_PIPELINE,
    RUN_PREFIX_PLAN,
    new_run_id,
)
from codeintel.core.execution.materialization import MaterializationResult, MaterializationStatus

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from codeintel.config.primitives import SnapshotRef


def cached_singleton[T](factory: Callable[[], T]) -> Callable[[], T]:
    """Create a cached singleton accessor using lru_cache.

    Use this decorator for simple singletons that don't need reset()
    functionality for testing. For registries that need reset(), use
    SingletonHolder from ``codeintel.core.singleton`` instead.

    Parameters
    ----------
    factory
        Function that creates the singleton instance.

    Returns
    -------
    Callable[[], T]
        Cached version of the factory that returns the same instance.

    Examples
    --------
    >>> @cached_singleton
    ... def get_config() -> Config:
    ...     return Config()
    >>> config1 = get_config()
    >>> config2 = get_config()
    >>> config1 is config2
    True
    """
    return lru_cache(maxsize=1)(factory)


def new_run_context(
    *,
    snapshot: SnapshotRef,
    kind: RunKind,
    trigger: TriggerKind,
    requested_operation: str | None = None,
    requested_datasets: Iterable[str] = (),
) -> RunContext:
    """Create a new RunContext with a generated run ID.

    This is the preferred factory for creating RunContext instances,
    ensuring consistent run ID generation across all entrypoints.

    Parameters
    ----------
    snapshot
        Repository snapshot reference.
    kind
        Classification of the run type.
    trigger
        How the run was triggered.
    requested_operation
        Optional operation ID that triggered this run.
    requested_datasets
        Optional dataset names requested for this run.

    Returns
    -------
    RunContext
        Fully initialized run context with generated run ID.

    Examples
    --------
    >>> from codeintel.config.primitives import SnapshotRef
    >>> from pathlib import Path
    >>> snapshot = SnapshotRef(repo="org/repo", commit="abc123", repo_root=Path("/tmp"))
    >>> ctx = new_run_context(snapshot=snapshot, kind="full", trigger="cli")
    >>> ctx.kind
    'full'
    >>> ctx.trigger
    'cli'
    >>> ctx.run_id.startswith("full-")
    True
    """
    return RunContext(
        run_id=new_run_id(kind),
        kind=kind,
        snapshot=snapshot,
        trigger=trigger,
        requested_operation=requested_operation,
        requested_datasets=tuple(requested_datasets),
    )


__all__ = [
    "PLUGIN_CATCHABLE_ERRORS",
    "RUN_PREFIX_ANALYTICS",
    "RUN_PREFIX_GRAPHS",
    "RUN_PREFIX_INGEST",
    "RUN_PREFIX_PIPELINE",
    "RUN_PREFIX_PLAN",
    "ExecutionContext",
    "MaterializationResult",
    "MaterializationStatus",
    "PluginFatalError",
    "PluginSkipRequestError",
    "PluginSkippedError",
    "PluginTimeoutError",
    "RunContext",
    "RunKind",
    "TriggerKind",
    "cached_singleton",
    "new_run_context",
    "new_run_id",
]
