"""Shared row context helpers for analytics row builders."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from codeintel.config.primitives import SnapshotRef


@dataclass(frozen=True, slots=True)
class RowBuildContext:
    """Shared row context for repo/commit scoped rows."""

    repo: str
    commit: str
    created_at: datetime

    @classmethod
    def from_snapshot(
        cls,
        snapshot: SnapshotRef,
        *,
        created_at: datetime | None = None,
    ) -> RowBuildContext:
        """Construct row context from a snapshot reference.

        Returns
        -------
        RowBuildContext
            Row context derived from the snapshot and resolved timestamp.
        """
        return cls.from_repo_commit(
            snapshot.repo,
            snapshot.commit,
            created_at=created_at,
        )

    @classmethod
    def from_repo_commit(
        cls,
        repo: str,
        commit: str,
        *,
        created_at: datetime | None = None,
    ) -> RowBuildContext:
        """Construct row context from repo and commit identifiers.

        Returns
        -------
        RowBuildContext
            Row context derived from the identifiers and resolved timestamp.
        """
        resolved = created_at or datetime.now(UTC)
        return cls(repo=repo, commit=commit, created_at=resolved)


__all__ = ["RowBuildContext"]
