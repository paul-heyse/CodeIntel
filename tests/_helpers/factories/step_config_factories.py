"""Factory functions for creating test snapshots.

This module provides factory functions for creating SnapshotRef instances
with standard test defaults, reducing boilerplate in tests.

Example
-------
>>> from tests._helpers.factories import make_snapshot
>>> snapshot = make_snapshot(tmp_path)
>>> snapshot.repo == "demo/repo"
True
"""

from __future__ import annotations

from pathlib import Path

from codeintel.config.primitives import SnapshotRef
from tests._helpers.fixtures.snapshots import DEFAULT_VARIANT


def make_snapshot(
    repo_root: Path | None = None,
    *,
    repo: str = DEFAULT_VARIANT.repo,
    commit: str = DEFAULT_VARIANT.commit,
) -> SnapshotRef:
    """Create a standard test snapshot.

    Parameters
    ----------
    repo_root
        Optional repo root path; defaults to Path.cwd() if not provided.
    repo
        Repository identifier; defaults to DEFAULT_VARIANT.repo.
    commit
        Commit identifier; defaults to DEFAULT_VARIANT.commit.

    Returns
    -------
    SnapshotRef
        Configured snapshot reference.
    """
    return SnapshotRef(
        repo=repo,
        commit=commit,
        repo_root=repo_root if repo_root is not None else Path.cwd(),
    )


__all__ = [
    "make_snapshot",
]
