"""Fake configuration primitives for testing.

This module provides fake implementations of configuration objects
for tests that need deterministic config behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.ingestion.engine.service import ToolService
from codeintel.runtime import RunContext

if TYPE_CHECKING:
    from codeintel.runtime.context import RunKind, TriggerKind


# =============================================================================
# Constants
# =============================================================================

DEFAULT_TEST_REPO = "test/repo"
DEFAULT_TEST_COMMIT = "abc123def"
DEFAULT_TEST_RUN_ID = "test-run-001"


# =============================================================================
# Fake Implementations
# =============================================================================


@dataclass(frozen=True)
class FakeSnapshotRef:
    """Fake SnapshotRef for config factory and plugin tests.

    Mirrors the real SnapshotRef interface with sensible test defaults.

    Attributes
    ----------
    repo : str
        Repository slug.
    commit : str
        Commit identifier.
    repo_root : Path
        Path to repository root.
    branch : str | None
        Optional branch name.
    """

    repo: str = "test/repo"
    commit: str = "testcommit"
    repo_root: Path = field(default_factory=lambda: Path("/repo"))
    branch: str | None = None


@dataclass(frozen=True)
class FakeBuildPaths:
    """Fake BuildPaths for config factory and plugin tests.

    Mirrors the real BuildPaths interface with sensible test defaults.

    Attributes
    ----------
    build_dir : Path
        Root build directory.
    db_path : Path
        Path to DuckDB database.
    document_output_dir : Path
        Directory for output documents.
    scip_dir : Path
        Directory for SCIP artifacts.
    coverage_json : Path
        Path for coverage JSON.
    pytest_report : Path
        Path for pytest JSON report.
    tool_cache : Path
        Cache directory for tools.
    log_db_path : Path
        Path to logging database.
    """

    build_dir: Path = field(default_factory=lambda: Path("/build"))
    db_path: Path = field(default_factory=lambda: Path("/build/codeintel.duckdb"))
    document_output_dir: Path = field(default_factory=lambda: Path("/build/docs"))
    scip_dir: Path = field(default_factory=lambda: Path("/build/scip"))
    coverage_json: Path = field(default_factory=lambda: Path("/build/coverage.json"))
    pytest_report: Path = field(default_factory=lambda: Path("/build/pytest.json"))
    tool_cache: Path = field(default_factory=lambda: Path("/cache"))
    log_db_path: Path = field(default_factory=lambda: Path("/build/log.duckdb"))


@dataclass
class FakePluginContext:
    """Fake IngestExecutionContext for config factory tests.

    Mirrors the real IngestExecutionContext interface with typed fields
    for proper static analysis.

    Attributes
    ----------
    snapshot : FakeSnapshotRef
        Snapshot reference.
    paths : FakeBuildPaths
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

    snapshot: FakeSnapshotRef = field(default_factory=FakeSnapshotRef)
    paths: FakeBuildPaths = field(default_factory=FakeBuildPaths)
    tools: ToolsConfig | None = None
    tracker: object | None = None
    tool_service: ToolService | None = None
    code_profile: object | None = None
    config_profile: object | None = None


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


__all__ = [
    "DEFAULT_TEST_COMMIT",
    "DEFAULT_TEST_REPO",
    "DEFAULT_TEST_RUN_ID",
    "FakeBuildPaths",
    "FakePluginContext",
    "FakeSnapshotRef",
    "create_test_build_paths",
    "create_test_run_context",
    "create_test_snapshot",
]
