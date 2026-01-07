"""Test configuration primitives for testing.

This module provides test implementations of configuration objects
for tests that need deterministic config behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.core.execution import RunContext

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.core.execution.context import RunKind, TriggerKind


from tests._helpers.fixtures.snapshots import DEFAULT_VARIANT

DEFAULT_TEST_REPO = DEFAULT_VARIANT.repo
DEFAULT_TEST_COMMIT = DEFAULT_VARIANT.commit
DEFAULT_TEST_RUN_ID = DEFAULT_VARIANT.run_id or "test-run-001"


def create_test_snapshot(
    tmp_path: Path,
    *,
    repo: str = DEFAULT_TEST_REPO,
    commit: str = DEFAULT_TEST_COMMIT,
) -> SnapshotRef:
    """Create a real SnapshotRef for testing.

    Parameters
    ----------
    tmp_path
        Temporary path for repo_root (required for test isolation).
    repo
        Repository identifier.
    commit
        Commit hash.

    Returns
    -------
    SnapshotRef
        Configured snapshot reference.
    """
    return SnapshotRef(repo=repo, commit=commit, repo_root=tmp_path)


def create_test_build_paths(
    repo_root: Path,
    *,
    build_dir: Path | None = None,
) -> BuildPaths:
    """Create a real BuildPaths for testing.

    Parameters
    ----------
    repo_root
        Repository root for test isolation.
    build_dir
        Optional build directory override; defaults to repo_root/build.

    Returns
    -------
    BuildPaths
        Configured build paths.
    """
    resolved_build_dir = build_dir or (repo_root / "build")
    return BuildPaths.from_repo_root(repo_root, build_dir=resolved_build_dir)


def create_test_run_context(
    snapshot: SnapshotRef,
    *,
    run_id: str = DEFAULT_TEST_RUN_ID,
    kind: RunKind = "full",
    trigger: TriggerKind = "cli",
) -> RunContext:
    """Create a real RunContext for testing.

    Parameters
    ----------
    snapshot
        Snapshot reference.
    run_id
        Run identifier.
    kind
        Run kind.
    trigger
        Trigger kind.

    Returns
    -------
    RunContext
        Configured run context.
    """
    return RunContext(
        run_id=run_id,
        kind=kind,
        snapshot=snapshot,
        trigger=trigger,
    )


__all__ = [
    "DEFAULT_TEST_COMMIT",
    "DEFAULT_TEST_REPO",
    "DEFAULT_TEST_RUN_ID",
    "create_test_build_paths",
    "create_test_run_context",
    "create_test_snapshot",
]
