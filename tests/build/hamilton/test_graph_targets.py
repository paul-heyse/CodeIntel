"""Tests for graph_targets.py graphs module.

This module validates that the consolidated graph targets in
``codeintel.build.hamilton.native.graphs.graph_targets`` work correctly
with the executor_materialize template for Pattern D targets.

Tests cover:
- goids target (GOID extraction)
- symbol_uses target (symbol use edge extraction)
- graph_metrics target (graph-derived analytics)
- graph_validation target (integrity checks)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from codeintel.build.contracts import OutputContract
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.native.graphs.graph_targets import (
    GoidExtractResult,
    GraphValidationResult,
    SymbolUsesExtractResult,
    t__goids,
    t__graph_metrics,
    t__graph_validation,
    t__symbol_uses,
)
from codeintel.build.targets import OutputTarget, TargetGraph
from codeintel.config.primitives import SnapshotRef
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true
from tests._helpers.build import make_build_config, make_build_paths
from tests._helpers.fakes.fake_providers import FakeProviders

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.build.providers import Providers
    from codeintel.storage.gateway import StorageGateway

# Test constants to avoid magic numbers
MAX_GOID_COUNT = 50
MAX_SYMBOL_USES_COUNT = 100
MAX_GRAPH_METRICS_COUNT = 25
MAX_GRAPH_VALIDATION_ERRORS = 10


def _make_env(
    *,
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    force_targets: frozenset[str] | None = None,
) -> BuildEnv:
    """Create a BuildEnv for testing.

    Parameters
    ----------
    gateway
        Storage gateway to use.
    snapshot
        Snapshot reference.
    force_targets
        Optional set of forced targets.

    Returns
    -------
    BuildEnv
        Build environment configured for testing.
    """
    paths = make_build_paths(snapshot.repo_root)
    config = make_build_config()
    providers = cast("Providers", FakeProviders.defaults())
    default_targets = frozenset(
        {
            "goids",
            "symbol_uses",
            "graph_metrics",
            "graph_validation",
        }
    )
    return BuildEnv(
        gateway=gateway,
        snapshot=snapshot,
        paths=paths,
        providers=providers,
        config=config,
        force_targets=force_targets or default_targets,
    )


def _make_graph() -> TargetGraph:
    """Create a minimal TargetGraph for graph targets.

    Returns
    -------
    TargetGraph
        Target graph with graph targets registered.
    """
    graph = TargetGraph()
    graph.register(
        OutputTarget(
            name="goids",
            module="graphs",
            contract=OutputContract.simple(table_keys=("core.goids", "core.goid_crosswalk")),
        )
    )
    graph.register(
        OutputTarget(
            name="symbol_uses",
            module="graphs",
            contract=OutputContract.simple(table_keys=("graph.symbol_use_edges",)),
        )
    )
    graph.register(
        OutputTarget(
            name="graph_metrics",
            module="graphs",
            contract=OutputContract.simple(
                table_keys=(
                    "analytics.graph_metrics_functions",
                    "analytics.graph_metrics_modules",
                )
            ),
        )
    )
    graph.register(
        OutputTarget(
            name="graph_validation",
            module="graphs",
            contract=OutputContract.simple(table_keys=("analytics.graph_validation",)),
        )
    )
    return graph


# ---------------------------------------------------------------------------
# GoidExtractResult Tests
# ---------------------------------------------------------------------------


def test_goid_extract_result_success() -> None:
    """Verify GoidExtractResult dataclass for success case."""
    result = GoidExtractResult(
        success=True,
        goid_count=MAX_GOID_COUNT,
        crosswalk_count=MAX_GOID_COUNT,
        table_counts={
            "core.goids": MAX_GOID_COUNT,
            "core.goid_crosswalk": MAX_GOID_COUNT,
        },
    )
    expect_true(result.success, message="Result should be successful")
    expect_equal(result.goid_count, MAX_GOID_COUNT)
    expect_equal(result.crosswalk_count, MAX_GOID_COUNT)
    expect_equal(result.table_counts["core.goids"], MAX_GOID_COUNT)
    expect_equal(result.error, None)


def test_goid_extract_result_failure() -> None:
    """Verify GoidExtractResult dataclass for failure case."""
    result = GoidExtractResult(
        success=False,
        table_counts={},
        error="Upstream modules failed",
    )
    expect_true(not result.success, message="Result should indicate failure")
    expect_equal(result.error, "Upstream modules failed")


# ---------------------------------------------------------------------------
# SymbolUsesExtractResult Tests
# ---------------------------------------------------------------------------


def test_symbol_uses_result_success() -> None:
    """Verify SymbolUsesExtractResult dataclass for success case."""
    result = SymbolUsesExtractResult(
        success=True,
        edge_count=MAX_SYMBOL_USES_COUNT,
        table_counts={"graph.symbol_use_edges": MAX_SYMBOL_USES_COUNT},
    )
    expect_true(result.success, message="Result should be successful")
    expect_equal(result.edge_count, MAX_SYMBOL_USES_COUNT)
    expect_equal(result.table_counts["graph.symbol_use_edges"], MAX_SYMBOL_USES_COUNT)
    expect_equal(result.error, None)


def test_symbol_uses_result_failure() -> None:
    """Verify SymbolUsesExtractResult dataclass for failure case."""
    result = SymbolUsesExtractResult(
        success=False,
        table_counts={},
        error="Upstream scip failed",
    )
    expect_true(not result.success, message="Result should indicate failure")
    expect_equal(result.error, "Upstream scip failed")


# ---------------------------------------------------------------------------
# ExecutionResult Tests
# ---------------------------------------------------------------------------


def test_execution_result_success() -> None:
    """Verify ExecutionResult for success case."""
    result = ExecutionResult.ok(
        table_counts={
            "analytics.graph_metrics_functions": MAX_GRAPH_METRICS_COUNT,
            "analytics.graph_metrics_modules": MAX_GRAPH_METRICS_COUNT,
        }
    )
    expect_true(result.success, message="Result should be successful")
    expect_equal(
        result.table_counts["analytics.graph_metrics_functions"],
        MAX_GRAPH_METRICS_COUNT,
    )
    expect_equal(result.error, None)


def test_execution_result_failure() -> None:
    """Verify ExecutionResult for failure case."""
    result = ExecutionResult.failed("Upstream call_graph failed")
    expect_true(not result.success, message="Result should indicate failure")
    expect_equal(result.error, "Upstream call_graph failed")


# ---------------------------------------------------------------------------
# GraphValidationResult Tests
# ---------------------------------------------------------------------------


def test_graph_validation_result_success() -> None:
    """Verify GraphValidationResult dataclass for success case (no errors)."""
    result = GraphValidationResult(
        success=True,
        error_count=0,
        errors=[],
        table_counts={"analytics.graph_validation": 0},
    )
    expect_true(result.success, message="Result should be successful")
    expect_equal(result.error_count, 0)
    expect_equal(len(result.errors), 0)
    expect_equal(result.error, None)


def test_graph_validation_result_with_errors() -> None:
    """Verify GraphValidationResult dataclass when validation finds issues."""
    validation_errors = [
        "Found 5 call graph edges with orphan caller GOIDs",
        "Found 3 import edges with missing source modules",
    ]
    result = GraphValidationResult(
        success=False,
        error_count=len(validation_errors),
        errors=validation_errors,
        table_counts={"analytics.graph_validation": len(validation_errors)},
    )
    expect_true(not result.success, message="Result should indicate failure")
    expect_equal(result.error_count, len(validation_errors))
    expect_equal(len(result.errors), len(validation_errors))


def test_graph_validation_result_fatal_failure() -> None:
    """Verify GraphValidationResult dataclass for fatal error case."""
    result = GraphValidationResult(
        success=False,
        table_counts={},
        error="Upstream call_graph target failed",
    )
    expect_true(not result.success, message="Result should indicate failure")
    expect_equal(result.error, "Upstream call_graph target failed")


# ---------------------------------------------------------------------------
# Materialize Function Tests - goids
# ---------------------------------------------------------------------------


def test_goids_materialize_success(
    fake_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Verify t__goids returns success record.

    Parameters
    ----------
    fake_gateway
        In-memory storage gateway fixture.
    tmp_path
        Temporary directory fixture.
    """
    snapshot = SnapshotRef(repo="test/repo", commit="abc123", repo_root=tmp_path)
    env = _make_env(gateway=fake_gateway, snapshot=snapshot)
    graph = _make_graph()

    compute_result = GoidExtractResult(
        success=True,
        goid_count=MAX_GOID_COUNT,
        crosswalk_count=MAX_GOID_COUNT,
        table_counts={
            "core.goids": MAX_GOID_COUNT,
            "core.goid_crosswalk": MAX_GOID_COUNT,
        },
    )

    record = t__goids(env, graph, compute_result)

    expect_equal(record.status, "succeeded")
    expect_true(
        record.row_counts.get("core.goids", 0) == MAX_GOID_COUNT,
        message="Row count should match compute result",
    )


def test_goids_materialize_failure(
    fake_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Verify t__goids returns failure record when compute fails.

    Parameters
    ----------
    fake_gateway
        In-memory storage gateway fixture.
    tmp_path
        Temporary directory fixture.
    """
    snapshot = SnapshotRef(repo="test/repo", commit="abc123", repo_root=tmp_path)
    env = _make_env(gateway=fake_gateway, snapshot=snapshot)
    graph = _make_graph()

    compute_result = GoidExtractResult(
        success=False,
        table_counts={},
        error="Upstream modules failed",
    )

    record = t__goids(env, graph, compute_result)

    expect_equal(record.status, "failed")
    expect_true(
        "Upstream modules failed" in (record.error or ""),
        message="Error message should be propagated",
    )


# ---------------------------------------------------------------------------
# Materialize Function Tests - symbol_uses
# ---------------------------------------------------------------------------


def test_symbol_uses_materialize_success(
    fake_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Verify t__symbol_uses returns success record.

    Parameters
    ----------
    fake_gateway
        In-memory storage gateway fixture.
    tmp_path
        Temporary directory fixture.
    """
    snapshot = SnapshotRef(repo="test/repo", commit="abc123", repo_root=tmp_path)
    env = _make_env(gateway=fake_gateway, snapshot=snapshot)
    graph = _make_graph()

    compute_result = SymbolUsesExtractResult(
        success=True,
        edge_count=MAX_SYMBOL_USES_COUNT,
        table_counts={"graph.symbol_use_edges": MAX_SYMBOL_USES_COUNT},
    )

    record = t__symbol_uses(env, graph, compute_result)

    expect_equal(record.status, "succeeded")
    expect_true(
        record.row_counts.get("graph.symbol_use_edges", 0) == MAX_SYMBOL_USES_COUNT,
        message="Row count should match compute result",
    )


def test_symbol_uses_materialize_failure(
    fake_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Verify t__symbol_uses returns failure record when compute fails.

    Parameters
    ----------
    fake_gateway
        In-memory storage gateway fixture.
    tmp_path
        Temporary directory fixture.
    """
    snapshot = SnapshotRef(repo="test/repo", commit="abc123", repo_root=tmp_path)
    env = _make_env(gateway=fake_gateway, snapshot=snapshot)
    graph = _make_graph()

    compute_result = SymbolUsesExtractResult(
        success=False,
        table_counts={},
        error="Upstream scip failed",
    )

    record = t__symbol_uses(env, graph, compute_result)

    expect_equal(record.status, "failed")
    expect_true(
        "Upstream scip failed" in (record.error or ""),
        message="Error message should be propagated",
    )


# ---------------------------------------------------------------------------
# Materialize Function Tests - graph_metrics
# ---------------------------------------------------------------------------


def test_graph_metrics_materialize_success(
    fake_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Verify t__graph_metrics returns success record.

    Parameters
    ----------
    fake_gateway
        In-memory storage gateway fixture.
    tmp_path
        Temporary directory fixture.
    """
    snapshot = SnapshotRef(repo="test/repo", commit="abc123", repo_root=tmp_path)
    env = _make_env(gateway=fake_gateway, snapshot=snapshot)
    graph = _make_graph()

    compute_result = ExecutionResult.ok(
        table_counts={
            "analytics.graph_metrics_functions": MAX_GRAPH_METRICS_COUNT,
            "analytics.graph_metrics_modules": MAX_GRAPH_METRICS_COUNT,
        }
    )

    record = t__graph_metrics(env, graph, compute_result)

    expect_equal(record.status, "succeeded")
    expect_true(
        record.row_counts.get("analytics.graph_metrics_functions", 0) == MAX_GRAPH_METRICS_COUNT,
        message="Row count should match compute result",
    )


def test_graph_metrics_materialize_failure(
    fake_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Verify t__graph_metrics returns failure record when compute fails.

    Parameters
    ----------
    fake_gateway
        In-memory storage gateway fixture.
    tmp_path
        Temporary directory fixture.
    """
    snapshot = SnapshotRef(repo="test/repo", commit="abc123", repo_root=tmp_path)
    env = _make_env(gateway=fake_gateway, snapshot=snapshot)
    graph = _make_graph()

    compute_result = ExecutionResult.failed("Upstream call_graph failed")

    record = t__graph_metrics(env, graph, compute_result)

    expect_equal(record.status, "failed")
    expect_true(
        "Upstream call_graph failed" in (record.error or ""),
        message="Error message should be propagated",
    )


# ---------------------------------------------------------------------------
# Materialize Function Tests - graph_validation
# ---------------------------------------------------------------------------


def test_graph_validation_materialize_success(
    fake_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Verify t__graph_validation returns success record.

    Parameters
    ----------
    fake_gateway
        In-memory storage gateway fixture.
    tmp_path
        Temporary directory fixture.
    """
    snapshot = SnapshotRef(repo="test/repo", commit="abc123", repo_root=tmp_path)
    env = _make_env(gateway=fake_gateway, snapshot=snapshot)
    graph = _make_graph()

    compute_result = GraphValidationResult(
        success=True,
        error_count=0,
        errors=[],
        table_counts={"analytics.graph_validation": 0},
    )

    record = t__graph_validation(env, graph, compute_result)

    expect_equal(record.status, "succeeded")


def test_graph_validation_materialize_failure(
    fake_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Verify t__graph_validation returns failure record when validation fails.

    Parameters
    ----------
    fake_gateway
        In-memory storage gateway fixture.
    tmp_path
        Temporary directory fixture.
    """
    snapshot = SnapshotRef(repo="test/repo", commit="abc123", repo_root=tmp_path)
    env = _make_env(gateway=fake_gateway, snapshot=snapshot)
    graph = _make_graph()

    compute_result = GraphValidationResult(
        success=False,
        error_count=2,
        errors=["Error 1", "Error 2"],
        table_counts={"analytics.graph_validation": 2},
    )

    record = t__graph_validation(env, graph, compute_result)

    expect_equal(record.status, "failed")
    expect_true(
        "Error 1" in (record.error or "") or "Error 2" in (record.error or ""),
        message="Error messages should be propagated",
    )
