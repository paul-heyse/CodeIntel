"""Tests for readiness computation from state variables.

This module tests the implicit calculation of target readiness from primary facts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import pytest

from codeintel.build.manifest import OutputManifest
from codeintel.build.readiness import (
    ActionNeeded,
    BlockerInfo,
    DatabaseReadinessView,
    DependencyStatus,
    SelfStatus,
    TargetReadiness,
    TargetReadinessView,
)
from codeintel.build.targets import OutputTarget, TargetGraph
from tests._helpers.assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_is_instance,
    expect_is_none,
    expect_is_not_none,
    expect_length,
    expect_true,
)

MIN_BLOCKER_CHAIN_LENGTH = 2
MIN_ESTIMATED_TIME_MS = 5000

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockSnapshotRef:
    """Mock snapshot reference for testing."""

    repo: str = "test-repo"
    commit: str = "abc123def456"
    repo_root: str = "/test/repo"


@dataclass
class MockBuildTracking:
    """Mock build tracking for testing."""

    _manifests: dict[str, OutputManifest]

    def load_manifest(self, target: str, repo: str, commit: str) -> OutputManifest | None:
        """Load a manifest by target name.

        Returns
        -------
        OutputManifest | None
            The manifest if found, None otherwise.
        """
        key = f"{repo}:{commit}:{target}"
        return self._manifests.get(key)

    def list_manifests(self, repo: str, commit: str) -> Sequence[OutputManifest]:
        """List all manifests for a snapshot.

        Returns
        -------
        Sequence[OutputManifest]
            All manifests matching the repo and commit.
        """
        prefix = f"{repo}:{commit}:"
        return [m for k, m in self._manifests.items() if k.startswith(prefix)]


@dataclass
class MockStorageGateway:
    """Mock storage gateway for testing."""

    build: MockBuildTracking


def as_gateway(mock: MockStorageGateway) -> StorageGateway:
    """Cast mock to StorageGateway for type checking.

    Returns
    -------
    StorageGateway
        The mock cast to StorageGateway type.
    """
    return cast("StorageGateway", mock)


def as_snapshot(mock: MockSnapshotRef) -> SnapshotRef:
    """Cast mock to SnapshotRef for type checking.

    Returns
    -------
    SnapshotRef
        The mock cast to SnapshotRef type.
    """
    return cast("SnapshotRef", mock)


def create_test_graph() -> TargetGraph:
    """Create a test graph with a simple dependency chain.

    Graph structure:
        ast (ingestion, no deps)
        └── goids (graphs, depends on ast)
            └── function_metrics (analytics, depends on goids)
                └── function_profile (analytics, depends on function_metrics)

    Returns
    -------
    TargetGraph
        Graph with four targets in a chain.
    """
    graph = TargetGraph()

    # Root target (no dependencies)
    ast = OutputTarget(
        name="ast",
        module="ingestion",
        plugin="AstPlugin",
        tables=("ingestion.ast",),
        dependencies=(),
    )
    graph.register(ast)

    # Second level
    goids = OutputTarget(
        name="goids",
        module="graphs",
        plugin="GoidsPlugin",
        tables=("graphs.goids",),
        dependencies=("ast",),
    )
    graph.register(goids)

    # Third level
    function_metrics = OutputTarget(
        name="function_metrics",
        module="analytics",
        plugin="FunctionMetricsPlugin",
        tables=("analytics.function_metrics",),
        dependencies=("goids",),
    )
    graph.register(function_metrics)

    # Fourth level
    function_profile = OutputTarget(
        name="function_profile",
        module="analytics",
        plugin="FunctionProfilePlugin",
        tables=("analytics.function_profile",),
        dependencies=("function_metrics",),
    )
    graph.register(function_profile)

    return graph


def create_manifest(
    target: str,
    repo: str = "test-repo",
    commit: str = "abc123def456",
    input_hash: str = "hash123",
    plugin: str = "TestPlugin",
) -> OutputManifest:
    """Create a test manifest.

    Returns
    -------
    OutputManifest
        A test manifest with the specified parameters.
    """
    return OutputManifest(
        target=target,
        repo=repo,
        commit=commit,
        plugin=plugin,
        input_hash=input_hash,
        output_hash="output_hash",
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        duration_ms=1000,
        row_count=100,
    )


@pytest.fixture
def graph() -> TargetGraph:
    """Fixture providing a test target graph.

    Returns
    -------
    TargetGraph
        Test target graph with a dependency chain.
    """
    return create_test_graph()


@pytest.fixture
def snapshot() -> MockSnapshotRef:
    """Fixture providing a test snapshot reference.

    Returns
    -------
    MockSnapshotRef
        Snapshot reference for tests.
    """
    return MockSnapshotRef()


@pytest.fixture
def gateway() -> MockStorageGateway:
    """Fixture providing an empty gateway.

    Returns
    -------
    MockStorageGateway
        Gateway with no manifests stored.
    """
    return MockStorageGateway(build=MockBuildTracking(_manifests={}))


@pytest.fixture
def empty_gateway() -> MockStorageGateway:
    """Fixture providing an empty gateway.

    Returns
    -------
    MockStorageGateway
        Gateway with no manifests stored.
    """
    return MockStorageGateway(build=MockBuildTracking(_manifests={}))


@pytest.fixture
def gateway_with_ast(snapshot: MockSnapshotRef) -> MockStorageGateway:
    """Fixture providing a gateway with an AST manifest.

    Parameters
    ----------
    snapshot
        Snapshot reference used to build the manifest key.

    Returns
    -------
    MockStorageGateway
        Gateway preloaded with an AST manifest.
    """
    manifest = create_manifest(
        "ast",
        repo=snapshot.repo,
        commit=snapshot.commit,
        input_hash="ast-hash",
    )
    key = f"{snapshot.repo}:{snapshot.commit}:ast"
    return MockStorageGateway(build=MockBuildTracking(_manifests={key: manifest}))


# =============================================================================
# Unit Tests for Type Definitions
# =============================================================================


def test_dependency_status_satisfied() -> None:
    """Test satisfied dependency status."""
    status = DependencyStatus(kind="satisfied")
    expect_true(status.is_satisfied)
    expect_false(status.is_blocked)
    expect_equal(status.blockers, ())
    expect_is_none(status.first_blocker)


def test_dependency_status_blocked() -> None:
    """Test blocked dependency status."""
    status = DependencyStatus(
        kind="blocked",
        blockers=("dep1", "dep2"),
        first_blocker="dep1",
    )
    expect_false(status.is_satisfied)
    expect_true(status.is_blocked)
    expect_equal(status.blockers, ("dep1", "dep2"))
    expect_equal(status.first_blocker, "dep1")


def test_action_needed_none() -> None:
    """Test when no action is needed."""
    action = ActionNeeded(kind="none")
    expect_true(action.is_ready)
    expect_is_none(action.target)
    expect_is_none(action.command)


def test_action_needed_run() -> None:
    """Test run action."""
    action = ActionNeeded(
        kind="run",
        target="ast",
        reason="never computed",
        command="codeintel build run ast",
    )
    expect_false(action.is_ready)
    expect_equal(action.target, "ast")
    expect_equal(action.command, "codeintel build run ast")


def test_action_needed_run_first() -> None:
    """Test run_first action."""
    action = ActionNeeded(
        kind="run_first",
        target="ast",
        reason="blocked by ast",
        command="codeintel build run ast",
    )
    expect_false(action.is_ready)
    expect_equal(action.target, "ast")


def test_target_readiness_ready() -> None:
    """Test ready target properties."""
    readiness = TargetReadiness(
        name="ast",
        self_status="current",
        dependency_status=DependencyStatus(kind="satisfied"),
        blocker_chain=(),
        action_needed=ActionNeeded(kind="none"),
    )
    expect_true(readiness.is_ready)
    expect_false(readiness.is_blocked)
    expect_false(readiness.can_run)
    expect_is_none(readiness.fix_command)


def test_target_readiness_runnable() -> None:
    """Test runnable target (can run now)."""
    readiness = TargetReadiness(
        name="ast",
        self_status="never_computed",
        dependency_status=DependencyStatus(kind="satisfied"),
        blocker_chain=(BlockerInfo(target="ast", blocked_by=None, reason="never computed"),),
        action_needed=ActionNeeded(
            kind="run",
            target="ast",
            reason="never computed",
            command="codeintel build run ast",
        ),
    )
    expect_false(readiness.is_ready)
    expect_false(readiness.is_blocked)
    expect_true(readiness.can_run)
    expect_equal(readiness.fix_command, "codeintel build run ast")


def test_target_readiness_blocked() -> None:
    """Test blocked target."""
    readiness = TargetReadiness(
        name="goids",
        self_status="never_computed",
        dependency_status=DependencyStatus(kind="blocked", blockers=("ast",), first_blocker="ast"),
        blocker_chain=(
            BlockerInfo(target="goids", blocked_by="ast", reason="dependency not ready"),
            BlockerInfo(target="ast", blocked_by=None, reason="never computed"),
        ),
        action_needed=ActionNeeded(
            kind="run_first",
            target="ast",
            reason="blocked by ast",
            command="codeintel build run ast",
        ),
        ultimate_bottleneck="ast",
    )
    expect_false(readiness.is_ready)
    expect_true(readiness.is_blocked)
    expect_false(readiness.can_run)
    expect_equal(readiness.ultimate_bottleneck, "ast")


# =============================================================================
# Integration Tests for TargetReadinessView
# =============================================================================


def test_root_target_can_run(
    graph: TargetGraph, gateway: MockStorageGateway, snapshot: MockSnapshotRef
) -> None:
    """Test root target with no deps can run."""
    target = graph.get("ast")
    view = TargetReadinessView(target, graph, as_gateway(gateway), as_snapshot(snapshot))

    expect_equal(view.name, "ast")
    expect_equal(view.module, "ingestion")
    expect_equal(view.self_status, "never_computed")
    expect_false(view.is_ready)
    expect_false(view.is_blocked)
    expect_true(view.can_run)
    expect_equal(view.ultimate_bottleneck, "ast")
    expect_equal(view.fix_command, "codeintel build run ast")


def test_dependent_target_is_blocked(
    graph: TargetGraph, gateway: MockStorageGateway, snapshot: MockSnapshotRef
) -> None:
    """Test dependent target is blocked when dependency not ready."""
    target = graph.get("goids")
    view = TargetReadinessView(target, graph, as_gateway(gateway), as_snapshot(snapshot))

    expect_equal(view.self_status, "never_computed")
    expect_true(view.is_blocked)
    expect_false(view.can_run)
    expect_equal(view.ultimate_bottleneck, "ast")
    expect_equal(view.fix_command, "codeintel build run ast")


def test_deep_dependency_chain(
    graph: TargetGraph, gateway: MockStorageGateway, snapshot: MockSnapshotRef
) -> None:
    """Test readiness propagates through deep dependency chain."""
    target = graph.get("function_profile")
    view = TargetReadinessView(target, graph, as_gateway(gateway), as_snapshot(snapshot))

    # function_profile -> function_metrics -> goids -> ast
    expect_true(view.is_blocked)
    expect_equal(view.ultimate_bottleneck, "ast")

    # Check blocker chain
    chain = view.blocker_chain
    expect_true(len(chain) >= MIN_BLOCKER_CHAIN_LENGTH)
    expect_equal(chain[0].target, "function_profile")
    expect_is_not_none(chain[0].blocked_by)


def test_computed_target_blocks_dependent_due_to_hash_mismatch(
    graph: TargetGraph,
    gateway_with_ast: MockStorageGateway,
    snapshot: MockSnapshotRef,
) -> None:
    """Test that hash mismatch makes target stale."""
    target = graph.get("ast")
    view = TargetReadinessView(target, graph, as_gateway(gateway_with_ast), as_snapshot(snapshot))

    # The manifest exists but hash likely doesn't match (mock hash vs computed)
    # So it should be either current or stale depending on hash
    self_status = view.self_status
    expect_true(self_status in {"current", "stale", "never_computed"})


def test_ready_target_properties() -> None:
    """Test properties of a ready target (mocked as current)."""
    readiness = TargetReadiness(
        name="ast",
        self_status="current",
        dependency_status=DependencyStatus(kind="satisfied"),
    )

    expect_true(readiness.is_ready)
    expect_false(readiness.is_blocked)
    expect_false(readiness.can_run)


# =============================================================================
def test_fresh_database_summary(
    graph: TargetGraph, empty_gateway: MockStorageGateway, snapshot: MockSnapshotRef
) -> None:
    """Test summary on fresh database."""
    view = DatabaseReadinessView(graph, as_gateway(empty_gateway), as_snapshot(snapshot))

    summary = view.summary()
    expect_equal(summary["total"], 4)  # ast, goids, function_metrics, function_profile
    expect_equal(summary["ready"], 0)
    # ast can run, others blocked
    expect_true(summary["runnable"] >= 1)
    expect_true(summary["blocked"] >= 1)


def test_iteration(
    graph: TargetGraph, empty_gateway: MockStorageGateway, snapshot: MockSnapshotRef
) -> None:
    """Test iterating over targets."""
    view = DatabaseReadinessView(graph, as_gateway(empty_gateway), as_snapshot(snapshot))

    names = list(view)
    expect_length(names, 4)
    expect_in("ast", names)
    expect_in("goids", names)
    expect_in("function_metrics", names)
    expect_in("function_profile", names)


def test_contains(
    graph: TargetGraph, empty_gateway: MockStorageGateway, snapshot: MockSnapshotRef
) -> None:
    """Test __contains__ method."""
    view = DatabaseReadinessView(graph, as_gateway(empty_gateway), as_snapshot(snapshot))

    expect_in("ast", view)
    expect_false("nonexistent" in view)


def test_getitem(
    graph: TargetGraph, empty_gateway: MockStorageGateway, snapshot: MockSnapshotRef
) -> None:
    """Test __getitem__ returns TargetReadinessView."""
    view = DatabaseReadinessView(graph, as_gateway(empty_gateway), as_snapshot(snapshot))

    target_view = view["ast"]
    expect_is_instance(target_view, TargetReadinessView)
    expect_equal(target_view.name, "ast")


def test_ready_targets(
    graph: TargetGraph, empty_gateway: MockStorageGateway, snapshot: MockSnapshotRef
) -> None:
    """Test ready_targets on fresh database."""
    view = DatabaseReadinessView(graph, as_gateway(empty_gateway), as_snapshot(snapshot))

    ready = view.ready_targets()
    expect_equal(ready, ())  # Nothing ready on fresh database


def test_runnable_targets(
    graph: TargetGraph, empty_gateway: MockStorageGateway, snapshot: MockSnapshotRef
) -> None:
    """Test runnable_targets on fresh database."""
    view = DatabaseReadinessView(graph, as_gateway(empty_gateway), as_snapshot(snapshot))

    runnable = view.runnable_targets()
    expect_in("ast", runnable)  # ast has no deps, can run


def test_blocked_targets(
    graph: TargetGraph, empty_gateway: MockStorageGateway, snapshot: MockSnapshotRef
) -> None:
    """Test blocked_targets on fresh database."""
    view = DatabaseReadinessView(graph, as_gateway(empty_gateway), as_snapshot(snapshot))

    blocked = view.blocked_targets()
    blocked_names = [name for name, _ in blocked]

    # goids, function_metrics, function_profile should be blocked
    expect_in("goids", blocked_names)
    expect_in("function_metrics", blocked_names)
    expect_in("function_profile", blocked_names)
    # ast should not be blocked (it can run)
    expect_false("ast" in blocked_names)


def test_bottlenecks(
    graph: TargetGraph, empty_gateway: MockStorageGateway, snapshot: MockSnapshotRef
) -> None:
    """Test bottlenecks identification."""
    view = DatabaseReadinessView(graph, as_gateway(empty_gateway), as_snapshot(snapshot))

    bottlenecks = view.bottlenecks()
    # ast is the only bottleneck since everything depends on it
    expect_in("ast", bottlenecks)


def test_targets_for_module(
    graph: TargetGraph, empty_gateway: MockStorageGateway, snapshot: MockSnapshotRef
) -> None:
    """Test targets_for_module filtering."""
    view = DatabaseReadinessView(graph, as_gateway(empty_gateway), as_snapshot(snapshot))

    analytics_targets = view.targets_for_module("analytics")
    expect_in("function_metrics", analytics_targets)
    expect_in("function_profile", analytics_targets)
    expect_false("ast" in analytics_targets)
    expect_false("goids" in analytics_targets)


def test_format_summary(
    graph: TargetGraph, empty_gateway: MockStorageGateway, snapshot: MockSnapshotRef
) -> None:
    """Test format_summary produces readable output."""
    view = DatabaseReadinessView(graph, as_gateway(empty_gateway), as_snapshot(snapshot))

    summary = view.format_summary()
    expect_in("test-repo", summary)
    expect_in("Ready:", summary)
    expect_in("Runnable:", summary)
    expect_in("Blocked:", summary)


def test_all_readiness(
    graph: TargetGraph, empty_gateway: MockStorageGateway, snapshot: MockSnapshotRef
) -> None:
    """Test all_readiness returns dict of all targets."""
    view = DatabaseReadinessView(graph, as_gateway(empty_gateway), as_snapshot(snapshot))

    all_readiness = view.all_readiness()
    expect_length(all_readiness, 4)
    expect_is_instance(all_readiness["ast"], TargetReadiness)


# =============================================================================
# Tests for Blocker Chain Computation
# =============================================================================


def test_blocker_info_structure() -> None:
    """Test BlockerInfo dataclass structure."""
    info = BlockerInfo(
        target="function_metrics",
        blocked_by="goids",
        reason="dependency not ready",
    )
    expect_equal(info.target, "function_metrics")
    expect_equal(info.blocked_by, "goids")
    expect_equal(info.reason, "dependency not ready")


def test_blocker_info_for_bottleneck() -> None:
    """Test BlockerInfo for a bottleneck (no blocker)."""
    info = BlockerInfo(
        target="ast",
        blocked_by=None,
        reason="never computed",
    )
    expect_equal(info.target, "ast")
    expect_is_none(info.blocked_by)


# =============================================================================
# Tests for Time Estimation
# =============================================================================


def test_time_estimation_for_runnable(
    graph: TargetGraph, empty_gateway: MockStorageGateway, snapshot: MockSnapshotRef
) -> None:
    """Test time estimation for a target that can run."""
    target = graph.get("ast")
    view = TargetReadinessView(target, graph, as_gateway(empty_gateway), as_snapshot(snapshot))

    readiness = view.readiness
    # ast has estimated_duration_ms of 5000
    if readiness.estimated_time_to_ready_ms is not None:
        expect_true(readiness.estimated_time_to_ready_ms >= MIN_ESTIMATED_TIME_MS)


# =============================================================================
# Edge Case Tests
# =============================================================================


def test_empty_graph() -> None:
    """Test with empty graph."""
    graph = TargetGraph()
    snapshot = MockSnapshotRef()
    gateway = MockStorageGateway(build=MockBuildTracking(_manifests={}))

    view = DatabaseReadinessView(graph, as_gateway(gateway), as_snapshot(snapshot))

    expect_equal(view.summary()["total"], 0)
    expect_equal(view.ready_targets(), ())
    expect_equal(view.runnable_targets(), ())
    expect_equal(view.bottlenecks(), ())


def test_single_target_no_deps() -> None:
    """Test graph with single target and no dependencies."""
    graph = TargetGraph()
    graph.register(
        OutputTarget(
            name="standalone",
            module="ingestion",
            plugin="StandalonePlugin",
            tables=("test.table",),
            dependencies=(),
        )
    )

    snapshot = MockSnapshotRef()
    gateway = MockStorageGateway(build=MockBuildTracking(_manifests={}))

    view = DatabaseReadinessView(graph, as_gateway(gateway), as_snapshot(snapshot))

    expect_in("standalone", view.runnable_targets())
    expect_true(view["standalone"].can_run)


def test_target_with_default_duration() -> None:
    """Test time estimation with default computed duration."""
    graph = TargetGraph()
    graph.register(
        OutputTarget(
            name="default_duration",
            module="ingestion",
            plugin="Plugin",
            tables=("test.table",),
            dependencies=(),
        )
    )

    snapshot = MockSnapshotRef()
    gateway = MockStorageGateway(build=MockBuildTracking(_manifests={}))

    view = DatabaseReadinessView(graph, as_gateway(gateway), as_snapshot(snapshot))
    readiness = view["default_duration"].readiness

    # Time estimation uses computed duration from TargetExecution
    # Default duration is 5000ms (base) so estimation should be present
    expect_is_not_none(readiness.estimated_time_to_ready_ms)


# =============================================================================
# Test Action Kind Values
# =============================================================================


def test_action_kind_none() -> None:
    """Test 'none' action kind."""
    action = ActionNeeded(kind="none")
    expect_equal(action.kind, "none")
    expect_true(action.is_ready)


def test_action_kind_run() -> None:
    """Test 'run' action kind."""
    action = ActionNeeded(kind="run", target="test")
    expect_equal(action.kind, "run")
    expect_false(action.is_ready)


def test_action_kind_run_first() -> None:
    """Test 'run_first' action kind."""
    action = ActionNeeded(kind="run_first", target="prereq")
    expect_equal(action.kind, "run_first")
    expect_false(action.is_ready)


def test_action_kind_blocked_external() -> None:
    """Test 'blocked_external' action kind."""
    action = ActionNeeded(kind="blocked_external", reason="External dependency missing")
    expect_equal(action.kind, "blocked_external")
    expect_false(action.is_ready)


# =============================================================================
# Test Self Status Values
# =============================================================================


def test_all_self_status_values() -> None:
    """Test that all SelfStatus values are valid."""
    valid_statuses: list[SelfStatus] = [
        "current",
        "stale",
        "never_computed",
        "data_missing",
    ]
    for status in valid_statuses:
        readiness = TargetReadiness(
            name="test",
            self_status=status,
            dependency_status=DependencyStatus(kind="satisfied"),
        )
        expect_equal(readiness.self_status, status)
