"""Test configuration primitives for testing.

This module provides test implementations of configuration objects
for tests that need deterministic config behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.core.execution import RunContext

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.config.models import ToolsConfig
    from codeintel.core.execution.context import RunKind, TriggerKind
    from codeintel.ingestion.engine.service import ToolService


# Import constants from central module (use canonical names)
from tests._helpers.constants import (
    DEFAULT_COMMIT as DEFAULT_TEST_COMMIT,
)
from tests._helpers.constants import (
    DEFAULT_REPO as DEFAULT_TEST_REPO,
)
from tests._helpers.constants import (
    DEFAULT_RUN_ID as DEFAULT_TEST_RUN_ID,
)

# =============================================================================
# Factory Functions
# =============================================================================


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
    tmp_path: Path,
) -> BuildPaths:
    """Create a real BuildPaths for testing.

    Parameters
    ----------
    tmp_path
        Temporary path for build directory (required for test isolation).

    Returns
    -------
    BuildPaths
        Configured build paths.
    """
    repo_root = tmp_path.parent
    return BuildPaths.from_repo_root(repo_root, build_dir=tmp_path)


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


# =============================================================================
# Test Context Types
# =============================================================================


@dataclass
class TestPluginContext:
    """Test context for config factory tests.

    Provide a minimal context with real SnapshotRef and BuildPaths types
    for proper static analysis.

    Use `TestPluginContext.from_tmp_path(tmp_path)` to create an instance
    with properly isolated paths.

    Attributes
    ----------
    snapshot : SnapshotRef
        Snapshot reference.
    paths : BuildPaths
        Build paths.
    tools : ToolsConfig | None
        Optional tools configuration.
    tracker : object | None
        Optional change tracker (using object to avoid circular imports).
    tool_service : ToolService | None
        Optional tool service.
    code_profile : object | None
        Optional code scan profile.
    config_profile : object | None
        Optional config scan profile.
    """

    snapshot: SnapshotRef
    paths: BuildPaths
    tools: ToolsConfig | None = None
    tracker: object | None = None
    tool_service: ToolService | None = None
    code_profile: object | None = None
    config_profile: object | None = None

    @classmethod
    def from_tmp_path(
        cls,
        tmp_path: Path,
        *,
        repo: str = DEFAULT_TEST_REPO,
        commit: str = DEFAULT_TEST_COMMIT,
    ) -> TestPluginContext:
        """Create a TestPluginContext with isolated paths.

        Parameters
        ----------
        tmp_path
            Temporary directory for test isolation.
        repo
            Repository identifier.
        commit
            Commit hash.

        Returns
        -------
        TestPluginContext
            Context with paths rooted under tmp_path.
        """
        snapshot = create_test_snapshot(tmp_path, repo=repo, commit=commit)
        build_dir = tmp_path / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        paths = create_test_build_paths(build_dir)
        return cls(snapshot=snapshot, paths=paths)


__all__ = [
    "DEFAULT_TEST_COMMIT",
    "DEFAULT_TEST_REPO",
    "DEFAULT_TEST_RUN_ID",
    "TestPluginContext",
    "create_test_build_paths",
    "create_test_run_context",
    "create_test_snapshot",
]
