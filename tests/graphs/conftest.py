"""Shared fixtures for graph plugin tests.

This module provides graph-specific test fixtures that wrap common setup
patterns, reducing boilerplate across the graph test suite. All fixtures
follow the Testing Charter principles of production parity and real
technology usage.

Available Fixtures
------------------
graph_gateway
    In-memory gateway with full schema and macros applied.
graph_snapshot
    Standard snapshot reference for testing.
graph_executor_env
    Combined gateway + snapshot environment with automatic cleanup.
graph_plugin_context
    Full GraphPluginExecutionContext ready for plugin testing.
graph_telemetry_context
    Context configured for telemetry/span testing.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from codeintel.config.primitives import SnapshotRef
from codeintel.core.plugins.execution.context import PluginScratch
from codeintel.graphs.core.context import GraphPluginExecutionContext
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.schema import apply_all_schemas
from tests._helpers.factories import make_snapshot
from tests._helpers.fakes.configs import create_test_snapshot
from tests._helpers.fakes.graph_contexts import (
    GraphExecutorTestEnv,
    GraphTelemetryTestEnv,
    create_graph_executor_env,
    create_graph_telemetry_env,
)
from tests._helpers.gateway import gateway_with_macros, open_ingestion_gateway_with_macros
from tests._helpers.seeds.golden_graphs import (
    GOLDEN_COMMIT,
    GOLDEN_REPO,
    seed_golden_graphs,
)

# ---------------------------------------------------------------------------
# Gateway Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def graph_gateway() -> Iterator[StorageGateway]:
    """Provide an in-memory gateway with full schema and macros for graph tests.

    This fixture creates a gateway with:
    - Schema applied
    - Views ensured
    - Schema validated
    - All ingest macros registered

    Yields
    ------
    StorageGateway
        Configured gateway; automatically closed after test.
    """
    gateway = gateway_with_macros(
        apply_schema=True,
        ensure_views=True,
        validate_schema=True,
    )
    apply_all_schemas(gateway.con)
    try:
        yield gateway
    finally:
        gateway.close()


@pytest.fixture
def graph_snapshot(tmp_path: Path) -> SnapshotRef:
    """Provide a standard snapshot reference for graph tests.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.

    Returns
    -------
    SnapshotRef
        Standard test snapshot with demo/repo and deadbeef commit.
    """
    return create_test_snapshot(tmp_path)


# ---------------------------------------------------------------------------
# Combined Environment Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def graph_executor_env(tmp_path: Path) -> Iterator[GraphExecutorTestEnv]:
    """Provide a combined gateway + snapshot environment for executor tests.

    This fixture handles all setup and cleanup, eliminating try/finally blocks
    in test code. The environment includes:
    - In-memory gateway with full schema
    - Standard snapshot reference
    - Automatic cleanup on test completion

    Yields
    ------
    GraphExecutorTestEnv
        Environment with gateway and snapshot; automatically closed.
    """
    env = create_graph_executor_env(tmp_path)
    try:
        yield env
    finally:
        env.close()


@pytest.fixture
def graph_telemetry_env() -> Iterator[GraphTelemetryTestEnv]:
    """Provide a full execution context environment for telemetry tests.

    This fixture provides:
    - In-memory gateway with full schema
    - GraphPluginExecutionContext with scratch store
    - ResourceContainer for graph resources
    - Automatic cleanup on test completion

    Yields
    ------
    GraphTelemetryTestEnv
        Environment with context and gateway; automatically closed.
    """
    env = create_graph_telemetry_env()
    try:
        yield env
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Execution Context Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def graph_plugin_context(
    graph_gateway: StorageGateway,
    graph_snapshot: SnapshotRef,
) -> GraphPluginExecutionContext:
    """Provide a GraphPluginExecutionContext for plugin testing.

    Combines the gateway and snapshot fixtures into a full execution context
    ready for graph plugin tests.

    Parameters
    ----------
    graph_gateway
        Gateway fixture with full schema.
    graph_snapshot
        Snapshot reference fixture.

    Returns
    -------
    GraphPluginExecutionContext
        Configured execution context for plugin testing.
    """
    return GraphPluginExecutionContext(
        gateway=graph_gateway,
        snapshot=graph_snapshot,
        scratch=PluginScratch(),
        plugin_name="test_plugin",
        run_id="test-run-123",
    )


# ---------------------------------------------------------------------------
# Golden Dataset Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def golden_gateway() -> Iterator[StorageGateway]:
    """Provide a gateway seeded with golden graph data.

    This fixture creates a gateway pre-populated with the golden dataset,
    useful for end-to-end pipeline scenario tests.

    Yields
    ------
    StorageGateway
        Gateway with golden dataset seeded; automatically closed.
    """
    gateway = open_ingestion_gateway_with_macros(
        apply_schema=True, ensure_views=True, validate_schema=True
    )
    apply_all_schemas(gateway.con)
    seed_golden_graphs(gateway, repo=GOLDEN_REPO, commit=GOLDEN_COMMIT)
    try:
        yield gateway
    finally:
        gateway.close()


@pytest.fixture
def golden_snapshot(tmp_path: Path) -> SnapshotRef:
    """Provide a snapshot reference for the golden dataset.

    Parameters
    ----------
    tmp_path
        Pytest temporary path.

    Returns
    -------
    SnapshotRef
        Snapshot reference for the golden repo and commit.
    """
    return make_snapshot(repo=GOLDEN_REPO, commit=GOLDEN_COMMIT, repo_root=tmp_path)


__all__ = [
    "golden_gateway",
    "golden_snapshot",
    "graph_executor_env",
    "graph_gateway",
    "graph_plugin_context",
    "graph_snapshot",
    "graph_telemetry_env",
]
