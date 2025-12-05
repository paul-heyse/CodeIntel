"""Test configuration primitives for testing.

This module provides test implementations of configuration objects
for tests that need deterministic config behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.core.execution import RunContext
from codeintel.ingestion.engine.service import ToolService

if TYPE_CHECKING:
    from codeintel.core.execution.context import RunKind, TriggerKind


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
    tmp_path: Path | None = None,
    *,
    repo: str = DEFAULT_TEST_REPO,
    commit: str = DEFAULT_TEST_COMMIT,
) -> SnapshotRef:
    """Create a real SnapshotRef for testing.

    Parameters
    ----------
    tmp_path
        Temporary path for repo_root. If None, uses a default mock path.
    repo
        Repository identifier.
    commit
        Commit hash.

    Returns
    -------
    SnapshotRef
        Configured snapshot reference.
    """
    repo_root = tmp_path if tmp_path is not None else Path("/mock/test-repo")
    return SnapshotRef(repo=repo, commit=commit, repo_root=repo_root)


def create_test_build_paths(
    tmp_path: Path | None = None,
) -> BuildPaths:
    """Create a real BuildPaths for testing.

    Parameters
    ----------
    tmp_path
        Temporary path for build directory. If None, uses a default mock path.

    Returns
    -------
    BuildPaths
        Configured build paths.
    """
    build_dir = tmp_path if tmp_path is not None else Path("/mock/build")
    repo_root = build_dir.parent if tmp_path is not None else Path("/mock/repo")
    return BuildPaths.from_repo_root(repo_root, build_dir=build_dir)


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

    snapshot: SnapshotRef = field(default_factory=create_test_snapshot)
    paths: BuildPaths = field(default_factory=create_test_build_paths)
    tools: ToolsConfig | None = None
    tracker: object | None = None
    tool_service: ToolService | None = None
    code_profile: object | None = None
    config_profile: object | None = None


# Backward compatibility alias
FakePluginContext = TestPluginContext


__all__ = [
    "DEFAULT_TEST_COMMIT",
    "DEFAULT_TEST_REPO",
    "DEFAULT_TEST_RUN_ID",
    "FakePluginContext",  # Backward compatibility alias
    "TestPluginContext",
    "create_test_build_paths",
    "create_test_run_context",
    "create_test_snapshot",
]
