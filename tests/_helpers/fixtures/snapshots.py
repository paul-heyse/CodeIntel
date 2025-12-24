"""Snapshot variants for consistent test setup."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from codeintel.config.primitives import SnapshotRef


@dataclass(frozen=True)
class SnapshotVariant:
    """Normalized snapshot identifier used by test helpers."""

    repo: str
    commit: str
    run_id: str | None = None
    repo_root: Path | None = None

    def to_snapshot(self, *, repo_root: Path | None = None) -> SnapshotRef:
        """Create a SnapshotRef with consistent formatting.

        Parameters
        ----------
        repo_root
            Optional override for the snapshot repo root.

        Returns
        -------
        SnapshotRef
            Snapshot reference for test contexts.
        """
        resolved_root = repo_root or self.repo_root or Path.cwd()
        return SnapshotRef(repo=self.repo, commit=self.commit, repo_root=resolved_root)


DEFAULT_VARIANT: Final[SnapshotVariant] = SnapshotVariant(
    repo="demo/repo",
    commit="deadbeef",
    run_id="test-run-001",
)
GOLDEN_VARIANT: Final[SnapshotVariant] = SnapshotVariant(
    repo="golden/repo",
    commit="golden123",
)
METRICS_VARIANT: Final[SnapshotVariant] = SnapshotVariant(
    repo="test/metrics",
    commit="metrics123",
)
SPAN_VARIANT: Final[SnapshotVariant] = SnapshotVariant(
    repo="demo/repo",
    commit="deadbeef",
)


@dataclass(frozen=True)
class SnapshotVariants:
    """Catalog of snapshot variants for test helpers."""

    default: SnapshotVariant = DEFAULT_VARIANT
    golden: SnapshotVariant = GOLDEN_VARIANT
    metrics: SnapshotVariant = METRICS_VARIANT
    span: SnapshotVariant = SPAN_VARIANT


SNAPSHOT_VARIANTS: Final[SnapshotVariants] = SnapshotVariants()

__all__ = [
    "DEFAULT_VARIANT",
    "GOLDEN_VARIANT",
    "METRICS_VARIANT",
    "SNAPSHOT_VARIANTS",
    "SPAN_VARIANT",
    "SnapshotVariant",
    "SnapshotVariants",
]
