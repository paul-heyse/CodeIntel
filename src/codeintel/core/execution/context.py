"""Unified run context types for CodeIntel pipelines.

This module defines the canonical RunContext type that provides consistent
run identity across ingestion, graphs, and analytics engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef


RunKind = Literal["ingest", "graphs", "analytics", "full", "op_prereqs"]
"""Classification of the run type.

- ``ingest``: Ingestion-only run (repo scan, AST extraction, etc.)
- ``graphs``: Graph computation run (call graph, import graph, etc.)
- ``analytics``: Analytics computation run (metrics, profiles, etc.)
- ``full``: Full pipeline run (ingest + graphs + analytics)
- ``op_prereqs``: Prerequisite computation for a specific operation
"""

TriggerKind = Literal["cli", "http", "mcp", "api"]
"""Classification of how the run was triggered.

- ``cli``: Command-line interface invocation
- ``http``: HTTP API request
- ``mcp``: MCP tool invocation
- ``api``: Direct programmatic API call
"""


@dataclass(frozen=True)
class RunContext:
    """Unified run metadata across ingestion, graphs, and analytics engines.

    This type provides consistent run identity and metadata that flows through
    all execution contexts, enabling correlation of logs, metrics, and traces
    across the entire pipeline.

    Parameters
    ----------
    run_id
        Unique identifier for this execution run.
    kind
        Classification of the run type.
    snapshot
        Repository snapshot reference containing repo, commit, and root path.
    trigger
        How the run was triggered.
    requested_operation
        Optional operation ID that triggered this run (e.g., "functions.summary").
    requested_datasets
        Optional dataset names requested for this run.

    Examples
    --------
    >>> from codeintel.config.primitives import SnapshotRef
    >>> from pathlib import Path
    >>> snapshot = SnapshotRef(repo="org/repo", commit="abc123", repo_root=Path("/tmp"))
    >>> ctx = RunContext(
    ...     run_id="ci-abc123",
    ...     kind="full",
    ...     snapshot=snapshot,
    ...     trigger="cli",
    ... )
    >>> ctx.repo
    'org/repo'
    >>> ctx.commit
    'abc123'
    """

    run_id: str
    kind: RunKind
    snapshot: SnapshotRef
    trigger: TriggerKind
    requested_operation: str | None = None
    requested_datasets: tuple[str, ...] = ()

    @property
    def repo(self) -> str:
        """Repository slug from the snapshot reference."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier from the snapshot reference."""
        return self.snapshot.commit


__all__ = [
    "RunContext",
    "RunKind",
    "TriggerKind",
]
