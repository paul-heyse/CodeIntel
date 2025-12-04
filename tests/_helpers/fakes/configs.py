"""Fake configuration primitives for testing.

This module provides fake implementations of configuration objects
for tests that need deterministic config behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from codeintel.config.models import ToolsConfig
from codeintel.ingestion.engine.service import ToolService


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


__all__ = [
    "FakeBuildPaths",
    "FakePluginContext",
    "FakeSnapshotRef",
]
