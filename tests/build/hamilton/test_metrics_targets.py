"""Tests for metrics_targets.py analytics module.

This module validates that the consolidated metrics targets in
``codeintel.build.hamilton.native.analytics.metrics_targets`` work correctly
with the executor_materialize template for Pattern D targets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from codeintel.build.contracts import OutputContract
from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.native.analytics.metrics_targets import (
    SubsystemAgreementComputeResult,
    SubsystemGraphMetricsComputeResult,
    SymbolGraphMetricsComputeResult,
    t__subsystem_agreement,
    t__subsystem_graph_metrics,
    t__symbol_graph_metrics,
)
from codeintel.build.targets import OutputTarget, TargetGraph
from codeintel.config.primitives import SnapshotRef
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true
from tests._helpers.build import TEST_BUILD_SETTINGS, make_build_config, make_build_paths
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
        settings=TEST_BUILD_SETTINGS,
        force_targets=force_targets
        or frozenset({"subsystem_graph_metrics", "symbol_graph_metrics", "subsystem_agreement"}),
    )


def _make_graph() -> TargetGraph:
    """Create a minimal TargetGraph for metrics targets.

    Returns
    -------
    TargetGraph
        Target graph with metrics targets registered.
    """
    graph = TargetGraph()
    graph.register(
        OutputTarget(
            name="subsystem_graph_metrics",
            module="analytics",
            contract=OutputContract.simple(table_keys=("analytics.subsystem_graph_metrics",)),
        )
    )
    graph.register(
        OutputTarget(
            name="symbol_graph_metrics",
            module="analytics",
            contract=OutputContract.simple(
                table_keys=(
                    "analytics.symbol_graph_metrics_modules",
                    "analytics.symbol_graph_metrics_functions",
                )
            ),
        )
    )
    graph.register(
        OutputTarget(
            name="subsystem_agreement",
            module="analytics",
            contract=OutputContract.simple(table_keys=("analytics.subsystem_agreement",)),
        )
    )
    return graph


def _make_materialization(
    table_key: str,
    row_count: int,
    *,
    status: str = "succeeded",
    error: str | None = None,
) -> MaterializationMetadata:
    return {
        "status": status,
        "table_key": table_key,
        "row_count": row_count,
        "duration_ms": 0.0,
        "input_hash": "test",
        "error": error,
    }


# ---------------------------------------------------------------------------
# ExecutionResult Tests
# ---------------------------------------------------------------------------


def test_execution_result_success() -> None:
    """Verify ExecutionResult for success case."""
    result = ExecutionResult.ok(table_counts={"analytics.subsystem_graph_metrics": 100})
    expect_true(result.success, message="Result should be successful")
    expect_equal(result.table_counts["analytics.subsystem_graph_metrics"], 100)
    expect_equal(result.error, None)


def test_execution_result_failure() -> None:
    """Verify ExecutionResult for failure case."""
    result = ExecutionResult.failed("Upstream subsystems failed")
    expect_true(not result.success, message="Result should indicate failure")
    expect_equal(result.error, "Upstream subsystems failed")


# ---------------------------------------------------------------------------
# Materialize Function Tests
# ---------------------------------------------------------------------------


def test_subsystem_graph_metrics_materialize_success(
    fake_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Verify t__subsystem_graph_metrics returns success record.

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

    compute_result = SubsystemGraphMetricsComputeResult(rows=[])
    materialization = _make_materialization("analytics.subsystem_graph_metrics", 25)

    record = t__subsystem_graph_metrics(env, graph, compute_result, materialization)

    expected_count = 25
    expect_equal(record.status, "succeeded")
    expect_true(
        record.row_counts.get("analytics.subsystem_graph_metrics", 0) == expected_count,
        message="Row count should match compute result",
    )


def test_subsystem_graph_metrics_materialize_failure(
    fake_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Verify t__subsystem_graph_metrics returns failure record when compute fails.

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

    compute_result = SubsystemGraphMetricsComputeResult(
        rows=None,
        error="Upstream subsystems failed",
    )
    materialization = _make_materialization(
        "analytics.subsystem_graph_metrics",
        0,
        status="failed",
        error="Upstream subsystems failed",
    )

    record = t__subsystem_graph_metrics(env, graph, compute_result, materialization)

    expect_equal(record.status, "failed")
    expect_true(
        "Upstream subsystems failed" in (record.error or ""),
        message="Error message should be propagated",
    )


def test_symbol_graph_metrics_materialize_success(
    fake_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Verify t__symbol_graph_metrics returns success record.

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

    compute_result = SymbolGraphMetricsComputeResult(module_rows=[], function_rows=[])
    modules_meta = _make_materialization("analytics.symbol_graph_metrics_modules", 10)
    functions_meta = _make_materialization("analytics.symbol_graph_metrics_functions", 50)

    record = t__symbol_graph_metrics(env, graph, compute_result, modules_meta, functions_meta)

    expect_equal(record.status, "succeeded")


def test_symbol_graph_metrics_materialize_failure(
    fake_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Verify t__symbol_graph_metrics returns failure record when compute fails.

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

    compute_result = SymbolGraphMetricsComputeResult(
        module_rows=None,
        function_rows=None,
        error="Upstream symbol_uses failed",
    )
    modules_meta = _make_materialization(
        "analytics.symbol_graph_metrics_modules",
        0,
        status="failed",
        error="Upstream symbol_uses failed",
    )
    functions_meta = _make_materialization(
        "analytics.symbol_graph_metrics_functions",
        0,
        status="failed",
        error="Upstream symbol_uses failed",
    )

    record = t__symbol_graph_metrics(env, graph, compute_result, modules_meta, functions_meta)

    expect_equal(record.status, "failed")
    expect_true(
        "Upstream symbol_uses failed" in (record.error or ""),
        message="Error message should be propagated",
    )


def test_subsystem_agreement_materialize_success(
    fake_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Verify t__subsystem_agreement returns success record.

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

    compute_result = SubsystemAgreementComputeResult(rows=[])
    materialization = _make_materialization("analytics.subsystem_agreement", 15)

    record = t__subsystem_agreement(env, graph, compute_result, materialization)

    expected_count = 15
    expect_equal(record.status, "succeeded")
    expect_true(
        record.row_counts.get("analytics.subsystem_agreement", 0) == expected_count,
        message="Row count should match compute result",
    )


def test_subsystem_agreement_materialize_failure(
    fake_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Verify t__subsystem_agreement returns failure record when compute fails.

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

    compute_result = SubsystemAgreementComputeResult(
        rows=None,
        error="Upstream subsystems failed",
    )
    materialization = _make_materialization(
        "analytics.subsystem_agreement",
        0,
        status="failed",
        error="Upstream subsystems failed",
    )

    record = t__subsystem_agreement(env, graph, compute_result, materialization)

    expect_equal(record.status, "failed")
    expect_true(
        "Upstream subsystems failed" in (record.error or ""),
        message="Error message should be propagated",
    )
