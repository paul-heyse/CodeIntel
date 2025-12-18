"""Tests for coverage_targets.py analytics module.

This module validates that the consolidated coverage targets in
``codeintel.build.hamilton.native.analytics.coverage_targets`` work correctly
with the executor_materialize template for Pattern D targets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from codeintel.build.contracts import OutputContract
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.native.analytics.coverage_targets import (
    t__behavioral_coverage,
    t__coverage_test_edges,
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
    return BuildEnv(
        gateway=gateway,
        snapshot=snapshot,
        paths=paths,
        providers=providers,
        config=config,
        force_targets=force_targets or frozenset({"coverage_test_edges", "behavioral_coverage"}),
    )


def _make_graph() -> TargetGraph:
    """Create a minimal TargetGraph for coverage targets.

    Returns
    -------
    TargetGraph
        Target graph with coverage targets registered.
    """
    graph = TargetGraph()
    graph.register(
        OutputTarget(
            name="coverage_test_edges",
            module="analytics",
            contract=OutputContract.simple(table_keys=("analytics.test_coverage_edges",)),
        )
    )
    graph.register(
        OutputTarget(
            name="behavioral_coverage",
            module="analytics",
            contract=OutputContract.simple(table_keys=("analytics.behavioral_coverage",)),
        )
    )
    return graph


# ---------------------------------------------------------------------------
# ExecutionResult Tests
# ---------------------------------------------------------------------------


def test_execution_result_success() -> None:
    """Verify ExecutionResult for success case."""
    result = ExecutionResult.ok(table_counts={"analytics.test_coverage_edges": 100})
    expect_true(result.success, message="Result should be successful")
    expect_equal(result.table_counts["analytics.test_coverage_edges"], 100)
    expect_equal(result.error, None)


def test_execution_result_failure() -> None:
    """Verify ExecutionResult for failure case."""
    result = ExecutionResult.failed("Upstream failed")
    expect_true(not result.success, message="Result should indicate failure")
    expect_equal(result.error, "Upstream failed")


# ---------------------------------------------------------------------------
# Materialize Function Tests
# ---------------------------------------------------------------------------


def test_coverage_test_edges_materialize_success(
    fake_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Verify t__coverage_test_edges returns success record.

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

    compute_result = ExecutionResult.ok(table_counts={"analytics.test_coverage_edges": 25})

    record = t__coverage_test_edges(env, graph, compute_result)

    expected_count = 25
    expect_equal(record.status, "succeeded")
    expect_true(
        record.row_counts.get("analytics.test_coverage_edges", 0) == expected_count,
        message="Row count should match compute result",
    )


def test_coverage_test_edges_materialize_failure(
    fake_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Verify t__coverage_test_edges returns failure record when compute fails.

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

    compute_result = ExecutionResult.failed("Upstream goids failed")

    record = t__coverage_test_edges(env, graph, compute_result)

    expect_equal(record.status, "failed")
    expect_true(
        "Upstream goids failed" in (record.error or ""),
        message="Error message should be propagated",
    )


def test_behavioral_coverage_materialize_success(
    fake_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Verify t__behavioral_coverage returns success record.

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

    compute_result = ExecutionResult.ok(table_counts={"analytics.behavioral_coverage": 15})

    record = t__behavioral_coverage(env, graph, compute_result)

    expected_count = 15
    expect_equal(record.status, "succeeded")
    expect_true(
        record.row_counts.get("analytics.behavioral_coverage", 0) == expected_count,
        message="Row count should match compute result",
    )


def test_behavioral_coverage_materialize_failure(
    fake_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Verify t__behavioral_coverage returns failure record when compute fails.

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

    compute_result = ExecutionResult.failed("Test profile failed")

    record = t__behavioral_coverage(env, graph, compute_result)

    expect_equal(record.status, "failed")
    expect_true(
        "Test profile failed" in (record.error or ""),
        message="Error message should be propagated",
    )
