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
from codeintel.core.plugins.context import PluginScratch
from codeintel.graphs.core.context import GraphPluginExecutionContext
from codeintel.graphs.resources.container import ResourceContainer
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.schemas import apply_all_schemas
from tests._helpers.fakes.graph_contexts import (
    GraphExecutorTestEnv,
    GraphTelemetryTestEnv,
    create_graph_executor_env,
    create_graph_telemetry_env,
)
from tests._helpers.gateway import gateway_with_macros


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_REPO = "demo/repo"
DEFAULT_COMMIT = "deadbeef"


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
    return SnapshotRef(repo=DEFAULT_REPO, commit=DEFAULT_COMMIT, repo_root=tmp_path)


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
        graph_resources=ResourceContainer(),
        scratch=PluginScratch(),
        plugin_name="test_plugin",
        run_id="test-run-123",
    )


__all__ = [
    "DEFAULT_COMMIT",
    "DEFAULT_REPO",
    "graph_executor_env",
    "graph_gateway",
    "graph_plugin_context",
    "graph_snapshot",
    "graph_telemetry_env",
]

