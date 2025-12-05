"""Data models for build system manifest and run tracking.

This module defines the core data structures for tracking build outputs
and execution runs. These models are persisted to the ``build`` schema
in DuckDB via the ``BuildTracking`` accessor.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

BuildStatus = Literal["running", "succeeded", "failed"]
"""Status of a build run.

- ``running``: Build is currently in progress
- ``succeeded``: Build completed successfully
- ``failed``: Build failed with errors
"""


@dataclass(frozen=True)
class OutputManifest:
    """Record of a target's computation.

    Each manifest entry represents one computed output for a specific
    (target, repo, commit) tuple. The ``input_hash`` enables cache
    invalidation when dependencies change.

    Attributes
    ----------
    target
        Target name (e.g., "risk_factors", "call_graph").
    repo
        Repository slug.
    commit
        Commit SHA.
    plugin
        Plugin name that produced this target.
    computed_at
        When the target was computed (UTC).
    duration_ms
        Computation duration in milliseconds.
    input_hash
        Content-addressable hash of all inputs (dependencies + options).
    output_hash
        Optional hash of output data for integrity verification.
    row_count
        Optional count of rows written to output tables.
    options_hash
        Optional hash of plugin configuration options.

    Examples
    --------
    >>> manifest = OutputManifest(
    ...     target="risk_factors",
    ...     repo="my-org/my-repo",
    ...     commit="abc123",
    ...     plugin="risk_factors_plugin",
    ...     computed_at=datetime.now(tz=UTC),
    ...     duration_ms=1234.5,
    ...     input_hash="a1b2c3d4e5f6",
    ... )
    """

    target: str
    repo: str
    commit: str
    plugin: str
    computed_at: datetime
    duration_ms: float
    input_hash: str
    output_hash: str | None = None
    row_count: int | None = None
    options_hash: str | None = None


@dataclass(frozen=True)
class BuildRunRecord:
    """Record of a build system run.

    Tracks the execution of a build run for debugging and observability.
    A build run may compute multiple targets based on what was requested
    and what was determined to be stale.

    Attributes
    ----------
    run_id
        Unique identifier for this run.
    repo
        Repository slug.
    commit
        Commit SHA.
    requested_targets
        Targets explicitly requested by the user.
    computed_targets
        Targets that were actually computed.
    skipped_targets
        Targets that were skipped (already fresh).
    started_at
        When the run started (UTC).
    completed_at
        When the run completed (None if still running).
    status
        Current status: running, succeeded, or failed.
    error_summary
        Summary of errors if failed.
    duration_ms
        Total run duration in milliseconds.

    Examples
    --------
    >>> record = BuildRunRecord(
    ...     run_id="run-123",
    ...     repo="my-org/my-repo",
    ...     commit="abc123",
    ...     requested_targets=("risk_factors",),
    ...     computed_targets=("goids", "call_graph", "risk_factors"),
    ...     skipped_targets=("modules",),
    ...     started_at=datetime.now(tz=UTC),
    ...     status="running",
    ... )
    """

    run_id: str
    repo: str
    commit: str
    requested_targets: tuple[str, ...]
    computed_targets: tuple[str, ...]
    skipped_targets: tuple[str, ...]
    started_at: datetime
    completed_at: datetime | None = None
    status: BuildStatus = "running"
    error_summary: str | None = None
    duration_ms: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation with ISO-formatted timestamps.
        """
        return {
            "run_id": self.run_id,
            "repo": self.repo,
            "commit": self.commit,
            "requested_targets": list(self.requested_targets),
            "computed_targets": list(self.computed_targets),
            "skipped_targets": list(self.skipped_targets),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "status": self.status,
            "error_summary": self.error_summary,
            "duration_ms": self.duration_ms,
        }


__all__ = [
    "BuildRunRecord",
    "BuildStatus",
    "OutputManifest",
]
