"""Data models for build system manifest and run tracking.

These dataclasses are persisted to DuckDB by the storage layer and are used by
both build and storage. They live in ``codeintel.core`` to avoid layering
violations (storage must not import build).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from datetime import datetime

BuildStatus = Literal["running", "succeeded", "failed"]
"""Status of a build run."""


@dataclass(frozen=True)
class OutputManifest:
    """Record of a target's computation."""

    target: str
    repo: str
    commit: str
    impl_kind: str
    computed_at: datetime
    duration_ms: float
    input_hash: str
    output_hash: str | None = None
    row_count: int | None = None
    options_hash: str | None = None
    dep_hashes: dict[str, str] | None = None
    change_delta: dict[str, object] | None = None


@dataclass(frozen=True)
class BuildRunRecord:
    """Record of a build system run."""

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
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "error_summary": self.error_summary,
            "duration_ms": self.duration_ms,
        }


__all__ = [
    "BuildRunRecord",
    "BuildStatus",
    "OutputManifest",
]
