"""MCP tools should set RequestContext around backend calls."""

from __future__ import annotations

import logging

import pytest

from codeintel.serving.backend import BackendLimits
from codeintel.serving.context import RequestContext
from codeintel.serving.mcp import registry as mcp_registry
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.services.observability import (
    ServiceCallMetrics,
    ServiceObservability,
)
from codeintel.storage.gateway import StorageGateway


class CapturingObservability(ServiceObservability):
    """Observability sink that records metrics and context payloads."""

    def __init__(self) -> None:
        logger = logging.getLogger("test_request_context_mcp")
        logger.setLevel(logging.INFO)
        super().__init__(enabled=True, logger=logger)
        self.events: list[tuple[ServiceCallMetrics, RequestContext | None]] = []

    def record(self, metrics: ServiceCallMetrics, context: RequestContext | None = None) -> None:
        """Capture the metrics payload and context for later assertions."""
        self.events.append((metrics, context))


def test_mcp_tools_set_request_context(architecture_gateway: StorageGateway) -> None:
    """MCP tool invocations should enrich observability metrics with RequestContext."""
    fastmcp_mod = pytest.importorskip("mcp.server.fastmcp")
    mcp = fastmcp_mod.FastMCP("test")
    if not hasattr(mcp, "tools"):
        pytest.skip("FastMCP tools registry not available")

    observability = CapturingObservability()
    backend = DuckDBBackend(
        gateway=architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=BackendLimits(default_limit=2, max_rows_per_call=5),
        observability=observability,
    )
    mcp_registry.register_tools(mcp, backend)

    tool = mcp.tools["read_dataset_rows"]
    result = tool({"dataset_name": "function_profile", "limit": 1})
    if isinstance(result, dict) and "error" in result:
        pytest.fail(f"Unexpected MCP error from tool: {result}")

    if not observability.events:
        pytest.fail("Expected observability event from MCP tool invocation")
    metrics, context = observability.events[-1]
    checks = [
        (metrics.external_transport == "mcp", f"External transport missing: {metrics}"),
        (metrics.operation == "datasets.rows", f"Operation not projected: {metrics}"),
        (metrics.dataset == "function_profile", f"Dataset not captured in metrics: {metrics}"),
        (bool(metrics.correlation_id), "Correlation id was not recorded on metrics"),
        (context is not None, "RequestContext was not passed to observability sink"),
    ]
    if context is not None:
        checks.extend(
            [
                (context.transport == "mcp", f"Unexpected context transport: {context}"),
                (context.dataset == "function_profile", f"Unexpected context dataset: {context}"),
            ]
        )
    for condition, message in checks:
        if not condition:
            pytest.fail(message)
