"""Orchestration utilities for unified run execution.

This module provides factory functions for creating RunContext instances
and orchestrating runs across multiple engines.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from codeintel.core.execution.context import RunContext, RunKind, TriggerKind
from codeintel.core.execution.ids import new_run_id

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef


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


__all__ = ["new_run_context"]
