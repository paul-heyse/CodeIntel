"""Shared fixtures for Hamilton Phase 2 tests.

Provides common test fixtures for plan testing, manifest mocking,
and target graph construction. Also includes pytest configuration
for CLI golden snapshot testing.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from codeintel.build.manifest import OutputManifest
from codeintel.build.targets import OutputTarget, TargetGraph

if TYPE_CHECKING:
    from collections.abc import Sequence


# =============================================================================
# CLI Snapshot Testing Configuration
# =============================================================================


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add CLI snapshot testing options to pytest.

    Parameters
    ----------
    parser
        Pytest argument parser.
    """
    parser.addoption(
        "--update-cli-snapshots",
        action="store_true",
        default=False,
        help="Update CLI golden snapshots instead of asserting.",
    )
    parser.addoption(
        "--cli-snapshot-manifest",
        action="store",
        default=None,
        help="Path to snapshot manifest (defaults to snapshots/manifest.yaml).",
    )
    parser.addoption(
        "--cli-snapshot-tags",
        action="store",
        default=None,
        help="Comma-separated tags to filter snapshot cases (e.g., pr14,graph).",
    )
    parser.addoption(
        "--cli-snapshot-pattern",
        action="store",
        default=None,
        help="Glob patterns to filter cases by name (e.g., pr14_*).",
    )
    parser.addoption(
        "--cli-snapshot-fail-fast",
        action="store_true",
        default=False,
        help="Stop after the first failing CLI snapshot (sets maxfail=1).",
    )
    parser.addoption(
        "--list-cli-snapshots",
        action="store_true",
        default=False,
        help="List CLI snapshot cases from the manifest and exit.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest for CLI snapshot testing.

    Parameters
    ----------
    config
        Pytest configuration object.
    """
    config.addinivalue_line("markers", "cli_snapshot: CLI golden snapshot tests")

    # Handle --cli-snapshot-fail-fast
    if config.getoption("--cli-snapshot-fail-fast"):
        maxfail = getattr(config.option, "maxfail", 0)
        if maxfail in (0, None):
            config.option.maxfail = 1

    # Handle --list-cli-snapshots
    if config.getoption("--list-cli-snapshots"):
        _list_cli_snapshots_and_exit(config)


def _list_cli_snapshots_and_exit(config: pytest.Config) -> None:
    """List snapshot cases from manifest and exit.

    Parameters
    ----------
    config
        Pytest configuration object.
    """
    from tests.build.hamilton.snapshots._manifest import load_snapshot_manifest

    snapshots_dir = Path(__file__).parent / "snapshots"

    override = config.getoption("--cli-snapshot-manifest")
    if override:
        manifest_path = Path(override)
    else:
        manifest_yaml = snapshots_dir / "manifest.yaml"
        manifest_json = snapshots_dir / "manifest.json"
        manifest_path = manifest_yaml if manifest_yaml.exists() else manifest_json

    if not manifest_path.exists():
        pytest.exit(f"Manifest not found: {manifest_path}", returncode=1)

    manifest = load_snapshot_manifest(manifest_path)

    # Apply optional filters
    tags_opt = config.getoption("--cli-snapshot-tags")
    pat_opt = config.getoption("--cli-snapshot-pattern")

    tags = {t.strip() for t in (tags_opt or "").split(",") if t.strip()}
    patterns = [p.strip() for p in (pat_opt or "").split(",") if p.strip()]

    def selected(case_tags: tuple[str, ...], case_name: str) -> bool:
        if tags and set(case_tags).isdisjoint(tags):
            return False
        if patterns and not any(fnmatch.fnmatch(case_name, p) for p in patterns):
            return False
        return True

    lines: list[str] = []
    lines.append(f"Manifest: {manifest_path}")
    lines.append(f"App: {manifest.app_import}")
    lines.append("")
    for c in manifest.cases:
        if not selected(c.tags, c.name):
            continue
        tag_str = ", ".join(c.tags) if c.tags else "-"
        lines.append(f"- {c.name}")
        lines.append(f"  tags: [{tag_str}]")
        lines.append(f"  kind: {c.kind}  output: {c.output}  exit_code: {c.exit_code}")
        lines.append(f"  snapshot: {c.snapshot}")
        lines.append(f"  args: {list(c.args)}")
        lines.append("")

    pytest.exit("\n".join(lines), returncode=0)


# =============================================================================
# Fake Gateway/Accessor for Testing
# =============================================================================


@dataclass
class FakeBuildAccessor:
    """Fake build accessor that returns pre-configured manifests.

    Use this to test planner and hashing without real DB access.
    """

    manifests: dict[str, OutputManifest] = field(default_factory=dict)
    load_manifest_calls: list[str] = field(default_factory=list)
    raise_on_load: bool = False

    def load_manifest(
        self,
        target: str,
        repo: str,
        commit: str,
    ) -> OutputManifest | None:
        """Load manifest from pre-configured dict.

        Returns
        -------
        OutputManifest | None
            The manifest if found, None otherwise.

        Raises
        ------
        RuntimeError
            If raise_on_load is True (for testing manifest_index usage).
        """
        _ = (repo, commit)  # Unused in fake
        self.load_manifest_calls.append(target)
        if self.raise_on_load:
            msg = f"load_manifest called for {target} - should use manifest_index"
            raise RuntimeError(msg)
        return self.manifests.get(target)

    def list_manifests(
        self,
        repo: str,
        commit: str,
    ) -> Sequence[OutputManifest]:
        """Return all manifests as a list.

        Returns
        -------
        Sequence[OutputManifest]
            All manifests in the fake accessor.
        """
        _ = (repo, commit)  # Unused in fake
        return list(self.manifests.values())


@dataclass
class FakeGateway:
    """Fake gateway with build accessor for testing."""

    build: FakeBuildAccessor = field(default_factory=FakeBuildAccessor)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fake_gateway() -> FakeGateway:
    """Create a fake gateway with empty manifest index.

    Returns
    -------
    FakeGateway
        A fake gateway instance.
    """
    return FakeGateway()


@pytest.fixture
def sample_manifest() -> OutputManifest:
    """Create a sample manifest for testing.

    Returns
    -------
    OutputManifest
        A sample manifest for the modules target.
    """
    return OutputManifest(
        target="modules",
        repo="test/repo",
        commit="abc123",
        plugin="ingestion.modules",
        computed_at=datetime.now(tz=UTC),
        duration_ms=100.0,
        input_hash="hash123456789012",
        output_hash="out123456789012",
        row_count=100,
    )


@pytest.fixture
def manifest_index_with_modules(sample_manifest: OutputManifest) -> dict[str, OutputManifest]:
    """Create a manifest index with modules target.

    Returns
    -------
    dict[str, OutputManifest]
        A manifest index containing the sample manifest.
    """
    return {sample_manifest.target: sample_manifest}


@pytest.fixture
def minimal_target_graph() -> TargetGraph:
    """Create a minimal 3-node target graph for deterministic testing.

    Graph structure:
        a (no deps) -> b (deps on a) -> c (deps on b)

    Returns
    -------
    TargetGraph
        A minimal target graph with three nodes.
    """
    graph = TargetGraph()
    graph.register(
        OutputTarget(
            name="a",
            module="ingestion",
            plugin="ingestion.a",
            description="Target A - no dependencies",
        )
    )
    graph.register(
        OutputTarget(
            name="b",
            module="graphs",
            plugin="graphs.b",
            dependencies=("a",),
            description="Target B - depends on A",
        )
    )
    graph.register(
        OutputTarget(
            name="c",
            module="analytics",
            plugin="analytics.c",
            dependencies=("b",),
            description="Target C - depends on B",
        )
    )
    return graph


@pytest.fixture
def diamond_target_graph() -> TargetGraph:
    """Create a diamond-shaped target graph for testing.

    Graph structure:
        a -> b -> d
        a -> c -> d

    Returns
    -------
    TargetGraph
        A diamond-shaped target graph with four nodes.
    """
    graph = TargetGraph()
    graph.register(
        OutputTarget(
            name="a",
            module="ingestion",
            plugin="ingestion.a",
            description="Root target",
        )
    )
    graph.register(
        OutputTarget(
            name="b",
            module="graphs",
            plugin="graphs.b",
            dependencies=("a",),
            description="Left branch",
        )
    )
    graph.register(
        OutputTarget(
            name="c",
            module="graphs",
            plugin="graphs.c",
            dependencies=("a",),
            description="Right branch",
        )
    )
    graph.register(
        OutputTarget(
            name="d",
            module="analytics",
            plugin="analytics.d",
            dependencies=("b", "c"),
            description="Diamond tip",
        )
    )
    return graph


__all__ = [
    "FakeBuildAccessor",
    "FakeGateway",
    "diamond_target_graph",
    "fake_gateway",
    "manifest_index_with_modules",
    "minimal_target_graph",
    "pytest_addoption",
    "pytest_configure",
    "sample_manifest",
]
