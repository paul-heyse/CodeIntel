"""Ensure TargetPluginAdapter populates metadata from target graph signals."""

from __future__ import annotations

from typing import ClassVar

import pytest
from pydantic import BaseModel

from codeintel.build.context import TargetExecutionContext
from codeintel.build.plugin import TargetPlugin
from codeintel.build.result import TargetResult
from codeintel.core.plugins.types.protocol import PluginResourceHints
from codeintel.graphs.core.adapters import TargetPluginAdapter
from codeintel.graphs.engine import GraphKind
from tests._helpers.assertions import expect_equal, expect_true


class _StubCallGraphPlugin(TargetPlugin):
    plugin_name: ClassVar[str] = "callgraph"
    plugin_version: ClassVar[str] = "9.9.9"
    plugin_description: ClassVar[str] = "Call graph builder"

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        _ = (self, ctx)
        return TargetResult.succeeded(row_counts={"graph.call_graph_nodes": 1})


class _StubMetricPlugin(TargetPlugin):
    plugin_name: ClassVar[str] = "graph_metrics.core"
    plugin_version: ClassVar[str] = "9.9.9"
    plugin_description: ClassVar[str] = "Graph metrics core"

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        _ = (self, ctx)
        return TargetResult.succeeded()


def test_adapter_infers_metadata_from_target_graph() -> None:
    """Adapter uses target graph metadata for tables, isolation, and dependencies."""
    adapter = TargetPluginAdapter(_StubCallGraphPlugin())
    metadata = adapter.metadata

    expect_equal(metadata.name, "callgraph")
    expect_equal(
        metadata.produces_tables,
        ("graph.call_graph_nodes", "graph.call_graph_edges"),
    )
    expect_equal(metadata.row_count_tables, metadata.produces_tables)
    expect_equal(metadata.depends_on, ("goid_builder", "scip_ingest"))
    expect_equal(metadata.produces_graphs, ("GraphKind.CALL_GRAPH",))
    expect_equal(metadata.produces_graph_kinds, (GraphKind.CALL_GRAPH,))
    expect_true(metadata.supports_incremental)
    expect_equal(metadata.isolation_kind, "thread")
    expect_true(metadata.requires_isolation)
    resource_hints = metadata.resource_hints
    if resource_hints is None:
        raise AssertionError("Expected resource_hints to be populated.")
    expect_equal(resource_hints.max_runtime_ms, 60000)


def test_adapter_sets_metric_requirements() -> None:
    """Metric plugins inherit graph requirements and target tables."""
    adapter = TargetPluginAdapter(_StubMetricPlugin())
    metadata = adapter.metadata

    expect_equal(metadata.name, "graph_metrics.core")
    expect_equal(
        metadata.produces_tables,
        (
            "analytics.graph_metrics_functions",
            "analytics.graph_metrics_functions_ext",
            "analytics.graph_metrics_modules",
            "analytics.graph_metrics_modules_ext",
        ),
    )
    expect_equal(metadata.requires_graphs, ("GraphKind.CALL_GRAPH", "GraphKind.IMPORT_GRAPH"))
    expect_equal(metadata.requires_graph_kinds, (GraphKind.CALL_GRAPH, GraphKind.IMPORT_GRAPH))
    expect_equal(metadata.produces_graphs, ())
    expect_true(metadata.supports_incremental)
    expect_equal(metadata.isolation_kind, "thread")
    expect_true(metadata.requires_isolation)


@pytest.mark.parametrize(
    ("attr_name", "attr_value", "metadata_field"),
    [
        ("plugin_provides", ["cap_a", "cap_b"], "provides"),
        ("plugin_requires", "needs_x", "requires"),
        ("plugin_cache_populates", {"cache_x"}, "cache_populates"),
        ("plugin_cache_consumes", ("cache_y",), "cache_consumes"),
        ("plugin_contract_checkers", ["checker_a"], "contract_checkers"),
    ],
)
def test_adapter_normalizes_tuple_attributes(
    attr_name: str,
    attr_value: object,
    metadata_field: str,
) -> None:
    """Ensure tuple-like plugin attributes are normalized on metadata."""

    class _Plugin(TargetPlugin):
        plugin_name: ClassVar[str] = "callgraph"
        plugin_version: ClassVar[str] = "1.2.3"
        plugin_description: ClassVar[str] = "Callgraph with extras"

        async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
            _ = (self, ctx)
            return TargetResult.succeeded()

    setattr(_Plugin, attr_name, attr_value)

    adapter = TargetPluginAdapter(_Plugin())
    metadata = adapter.metadata

    if isinstance(attr_value, (tuple, list, set)):
        expected = tuple(str(item) for item in attr_value)
    elif isinstance(attr_value, str):
        expected = (attr_value,)
    else:
        expected = (str(attr_value),)
    expect_equal(getattr(metadata, metadata_field), expected)


def test_adapter_applies_options_model_and_default() -> None:
    """Options model and default values are passed through to metadata."""

    class _Options(BaseModel):
        threshold: int = 5

    class _Plugin(TargetPlugin):
        plugin_name: ClassVar[str] = "callgraph"
        plugin_version: ClassVar[str] = "1.0.0"
        plugin_description: ClassVar[str] = "Callgraph with options"
        plugin_options_model: ClassVar[type[BaseModel]] = _Options
        plugin_options_default: ClassVar[dict[str, int]] = {"threshold": 10}

        async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
            _ = (self, ctx)
            return TargetResult.succeeded()

    metadata = TargetPluginAdapter(_Plugin()).metadata
    expect_equal(metadata.options_model, _Options)
    expect_equal(metadata.options_default, {"threshold": 10})


def test_adapter_respects_plugin_resource_hints_and_incremental_override() -> None:
    """Plugin-specific resource hints and incremental flags override target defaults."""

    class _Plugin(TargetPlugin):
        plugin_name: ClassVar[str] = "callgraph"
        plugin_version: ClassVar[str] = "1.0.0"
        plugin_description: ClassVar[str] = "Callgraph tuned"
        plugin_resource_hints: ClassVar[PluginResourceHints] = PluginResourceHints(
            max_runtime_ms=123,
            max_memory_mb=256,
            cpu_intensive=True,
            io_intensive=True,
            requires_gpu=True,
            priority=5,
        )
        plugin_supports_incremental: ClassVar[bool] = False

        async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
            _ = (self, ctx)
            return TargetResult.succeeded()

    metadata = TargetPluginAdapter(_Plugin()).metadata
    expect_true(metadata.resource_hints is not None)
    expect_equal(metadata.resource_hints, _Plugin.plugin_resource_hints)
    expect_true(metadata.supports_incremental is False)


def test_adapter_appends_plugin_depends_on() -> None:
    """Plugin-defined dependencies are appended to target-derived dependencies."""

    class _Plugin(TargetPlugin):
        plugin_name: ClassVar[str] = "callgraph"
        plugin_version: ClassVar[str] = "1.0.0"
        plugin_description: ClassVar[str] = "Callgraph with extra dependency"
        plugin_depends_on: ClassVar[tuple[str, ...]] = ("extra_dep",)

        async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
            _ = (self, ctx)
            return TargetResult.succeeded()

    metadata = TargetPluginAdapter(_Plugin()).metadata
    expect_equal(
        metadata.depends_on,
        ("goid_builder", "scip_ingest", "extra_dep"),
    )
