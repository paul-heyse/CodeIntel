"""Extended tests for graph runtime executor module.

This module provides additional test coverage for the executor module,
focusing on specific execution paths not covered by test_runtime.py:

- Dry run and manifest-based skip paths
- Fatal error propagation
- Status counts aggregation
- Pipeline run tracking integration
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Final

import pytest

from codeintel.config.steps_graphs import GraphPluginPolicy, GraphRunScope
from codeintel.core.execution.errors import PluginFatalError
from codeintel.core.plugins.execution.policy import BaseExecutionPolicy
from codeintel.core.plugins.types.result import PluginExecutionRecord
from codeintel.core.resources import ResourceNotFoundError
from codeintel.graphs.catalog import FunctionCatalog, FunctionCatalogProvider
from codeintel.graphs.core.context import (
    GraphPluginExecutionContextBuilder,
)
from codeintel.graphs.core.protocol import GraphPluginProtocol
from codeintel.graphs.engine import NxGraphEngine
from codeintel.graphs.resources.graphs import GraphResource
from codeintel.graphs.runtime import graph_executor
from codeintel.graphs.runtime.graph_executor import (
    GraphExecutorContext,
    GraphPluginExecutor,
    GraphRunReport,
)
from codeintel.graphs.runtime.planning import (
    GraphPlanContext,
    plan_graph_plugin_run,
)
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_length,
    expect_true,
)
from tests._helpers.fakes.graph_contexts import GraphTestEnv
from tests._helpers.fakes.graph_plugins import make_graph_plugin, plugin_registrar


class _MockFunctionCatalogProvider(FunctionCatalogProvider):
    """Minimal provider satisfying FunctionCatalogProvider protocol for tests."""

    def __init__(self) -> None:
        self._catalog = FunctionCatalog(functions=(), module_by_path={})

    def catalog(self) -> FunctionCatalog:
        return self._catalog

    @staticmethod
    def urn_for_goid(goid: int) -> str | None:  # pragma: no cover - protocol stub
        del goid
        return None

    @staticmethod
    def module_for_path(rel_path: str) -> str | None:  # pragma: no cover - protocol stub
        del rel_path
        return None

    @staticmethod
    def lookup_goid(
        rel_path: str, start_line: int, end_line: int | None, qualname: str | None
    ) -> int | None:  # pragma: no cover - protocol stub
        del rel_path, start_line, end_line, qualname
        return None


# Constants
STATUS_SUCCESS_COUNT: Final = 2
STATUS_FAILURE_COUNT: Final = 1
STATUS_SKIPPED_COUNT: Final = 1
REPORT_SUCCESS_COUNT: Final = 2
REPORT_FAILURE_COUNT: Final = 1
REPORT_MIXED_SUCCESS_COUNT: Final = 1
_EXECUTOR_PRIVATES: Final = graph_executor.__dict__
STATUS_COUNTS = _EXECUTOR_PRIVATES["_status_counts"]


# Test Helpers


def _make_test_plugin(name: str, config: Mapping[str, object] | None = None) -> GraphPluginProtocol:
    """Create a configurable test plugin.

    Parameters
    ----------
    name
        Plugin name.
    config
        Optional configuration mapping:
        - succeed: bool (default True)
        - row_counts: Mapping[str, int]
        - raise_exception: type[Exception]
        - delay_ms: int
        - input_hash: str
        - options_hash: str

    Returns
    -------
    GraphPluginProtocol
        Configured test plugin instance.
    """
    runtime: dict[str, object] = {}
    runtime.update(config or {})
    runtime.setdefault("succeed", True)
    runtime.setdefault("exception_message", f"{name} exception")
    runtime.setdefault("error_message", f"Plugin {name} failed")
    if "raise_exception" in runtime:
        runtime["exception_type"] = runtime.pop("raise_exception")
    if "row_counts" in runtime and isinstance(runtime["row_counts"], Mapping):
        runtime["row_counts"] = dict(runtime["row_counts"])
    return make_graph_plugin(name, runtime=runtime)


def test_status_counts_aggregates_correctly() -> None:
    """Status counts correctly aggregate success/failure/skip counts."""
    now = datetime.now(tz=UTC)
    records = [
        PluginExecutionRecord(
            plugin_name="p1", status="succeeded", started_at=now, ended_at=now, duration_ms=10
        ),
        PluginExecutionRecord(
            plugin_name="p2", status="succeeded", started_at=now, ended_at=now, duration_ms=20
        ),
        PluginExecutionRecord(
            plugin_name="p3", status="failed", started_at=now, ended_at=now, duration_ms=30
        ),
        PluginExecutionRecord(
            plugin_name="p4", status="skipped", started_at=now, ended_at=now, duration_ms=0
        ),
    ]

    counts = STATUS_COUNTS(records)

    expect_equal(counts["success"], STATUS_SUCCESS_COUNT)
    expect_equal(counts["failure"], STATUS_FAILURE_COUNT)
    expect_equal(counts["skipped"], STATUS_SKIPPED_COUNT)


def test_status_counts_empty_records() -> None:
    """Status counts returns zeros for empty record list."""
    counts = STATUS_COUNTS([])

    expect_equal(counts["success"], 0)
    expect_equal(counts["failure"], 0)
    expect_equal(counts["skipped"], 0)


def test_graph_plugin_executor_dry_run_skips_execution(
    graph_executor_env: GraphTestEnv,
) -> None:
    """Dry run mode skips actual plugin execution."""
    plugin = _make_test_plugin("dry_run_plugin", {"succeed": True})

    with plugin_registrar([plugin]):
        context = GraphPlanContext(
            runtime_snapshot=graph_executor_env.snapshot,
            policy=GraphPluginPolicy(dry_run=True),
        )
        plan = plan_graph_plugin_run(
            plugin_names=[plugin.metadata.name],
            context=context,
        )

        executor_context = GraphExecutorContext(
            gateway=graph_executor_env.gateway,
            snapshot=graph_executor_env.snapshot,
        )

        # Convert policy and use GraphPluginExecutor directly
        base_policy = BaseExecutionPolicy(
            dry_run=True,
        )
        graph_executor = GraphPluginExecutor(
            policy=base_policy,
            scope=plan.scope,
        )

        report = graph_executor.execute(
            executor_ctx=executor_context,
            plugins=plan.plugins,
            run_id=plan.run_id,
            settings_by_plugin=plan.settings_by_plugin,
        )

        expect_equal(report.skip_count, 1)
        expect_equal(report.success_count, 0)
        expect_true(report.records)
    expect_equal(report.records[0].meta.get("skipped_reason"), "dry_run")


def test_graph_plugin_execution_context_require_graphs_by_type_and_name(
    graph_executor_env: GraphTestEnv,
) -> None:
    """require_graphs resolves resources by type and name; errors when missing."""
    engine = NxGraphEngine(gateway=graph_executor_env.gateway, snapshot=graph_executor_env.snapshot)
    engine_resource = GraphResource(engine=engine)
    base_builder = GraphPluginExecutionContextBuilder(
        gateway=graph_executor_env.gateway,
        snapshot=graph_executor_env.snapshot,
        run_id="run",
    )
    ctx_missing = base_builder.build_graph_context()
    with pytest.raises(RuntimeError, match="No GraphResource"):
        ctx_missing.require_graphs()

    ctx = base_builder.build_graph_context()

    # Register by type
    ctx.resources.register(GraphResource, engine_resource)
    expect_true(ctx.has_resource(GraphResource))
    expect_true(ctx.require_graphs() is engine_resource)

    # Register by name only and ensure fallback works
    ctx.resources.register_provider(engine_resource)
    expect_true(ctx.has_graph_resource(GraphResource.RESOURCE_NAME))
    expect_true(ctx.require_graphs() is engine_resource)


def test_graph_plugin_execution_context_require_graph_resource_by_name(
    graph_executor_env: GraphTestEnv,
) -> None:
    """require_graph_resource_by_name raises ResourceNotFoundError when missing."""
    builder = GraphPluginExecutionContextBuilder(
        gateway=graph_executor_env.gateway,
        snapshot=graph_executor_env.snapshot,
        run_id="run",
    )
    ctx = builder.build_graph_context()

    with pytest.raises(ResourceNotFoundError):
        ctx.require_graph_resource_by_name("missing")


def test_graph_plugin_execution_context_builder_wiring(graph_executor_env: GraphTestEnv) -> None:
    """Builder should propagate scope, catalog provider, and registered resources."""
    scope = GraphRunScope(paths=("a.py",))
    catalog_provider = _MockFunctionCatalogProvider()
    engine = NxGraphEngine(gateway=graph_executor_env.gateway, snapshot=graph_executor_env.snapshot)
    resource = GraphResource(engine=engine)

    builder = GraphPluginExecutionContextBuilder(
        gateway=graph_executor_env.gateway,
        snapshot=graph_executor_env.snapshot,
        run_id="builder-run",
    )
    builder = (
        builder.with_scope(scope)
        .with_catalog_provider(catalog_provider)
        .register_graph_resource(resource)
        .with_resource(GraphResource, resource)
    )
    ctx = builder.build_graph_context()

    expect_true(ctx.scope is scope)
    expect_true(ctx.catalog_provider is catalog_provider)
    expect_true(ctx.has_resource(GraphResource))
    expect_true(ctx.require_graphs() is resource)


def test_graph_plugin_executor_skip_on_unchanged(
    graph_executor_env: GraphTestEnv,
) -> None:
    """Plugin skipped when manifest shows inputs unchanged."""
    plugin = _make_test_plugin("unchanged_plugin", {"succeed": True})

    with plugin_registrar([plugin]):
        context = GraphPlanContext(
            runtime_snapshot=graph_executor_env.snapshot,
            policy=GraphPluginPolicy(skip_on_unchanged=True),
            prior_manifest={
                plugin.metadata.name: {
                    "input_hash": "will_be_computed",
                    "options_hash": None,
                }
            },
        )
        plan = plan_graph_plugin_run(
            plugin_names=[plugin.metadata.name],
            context=context,
        )

        # Get the actual computed input hash
        settings = plan.settings_by_plugin[plugin.metadata.name]
        # Update prior manifest with actual hash
        prior_manifest: Mapping[str, Mapping[str, object]] = {
            plugin.metadata.name: {
                "input_hash": settings.input_hash,
                "options_hash": settings.options_hash,
            }
        }

        # Create new context with correct prior manifest
        context_with_correct_hash = GraphPlanContext(
            runtime_snapshot=graph_executor_env.snapshot,
            policy=GraphPluginPolicy(skip_on_unchanged=True),
            prior_manifest=prior_manifest,
        )
        plan_with_correct_hash = plan_graph_plugin_run(
            plugin_names=[plugin.metadata.name],
            context=context_with_correct_hash,
        )

        executor_context = GraphExecutorContext(
            gateway=graph_executor_env.gateway,
            snapshot=graph_executor_env.snapshot,
        )

        # Use GraphPluginExecutor directly
        base_policy = BaseExecutionPolicy(
            skip_on_unchanged=True,
        )
        graph_executor = GraphPluginExecutor(
            policy=base_policy,
            prior_manifest=plan_with_correct_hash.prior_manifest,
            scope=plan_with_correct_hash.scope,
        )

        report = graph_executor.execute(
            executor_ctx=executor_context,
            plugins=plan_with_correct_hash.plugins,
            run_id=plan_with_correct_hash.run_id,
            settings_by_plugin=plan_with_correct_hash.settings_by_plugin,
        )

        expect_equal(report.skip_count, 1)
        expect_true(report.records)
        expect_equal(report.records[0].meta.get("skipped_reason"), "unchanged")


def test_graph_plugin_executor_builds_manifest(
    graph_executor_env: GraphTestEnv,
) -> None:
    """Successful plugin execution populates manifest in report."""
    plugin = _make_test_plugin("manifest_build_plugin", {"succeed": True})

    with plugin_registrar([plugin]):
        context = GraphPlanContext(
            runtime_snapshot=graph_executor_env.snapshot,
            policy=GraphPluginPolicy(),
        )
        plan = plan_graph_plugin_run(
            plugin_names=[plugin.metadata.name],
            context=context,
        )

        executor_context = GraphExecutorContext(
            gateway=graph_executor_env.gateway,
            snapshot=graph_executor_env.snapshot,
        )

        # Use GraphPluginExecutor directly
        graph_executor = GraphPluginExecutor(
            scope=plan.scope,
        )

        report = graph_executor.execute(
            executor_ctx=executor_context,
            plugins=plan.plugins,
            run_id=plan.run_id,
            settings_by_plugin=plan.settings_by_plugin,
        )

        expect_in(plugin.metadata.name, report.manifest)
        entry = report.manifest[plugin.metadata.name]
        expect_in("input_hash", entry)
        expect_in("executed_at", entry)


def test_graph_plugin_executor_fatal_stops_remaining(
    graph_executor_env: GraphTestEnv,
) -> None:
    """Fatal plugin error stops execution of remaining plugins."""
    fatal_plugin = _make_test_plugin("fatal_first", {"raise_exception": RuntimeError})
    second_plugin = _make_test_plugin("second_should_not_run", {"succeed": True})

    with plugin_registrar([fatal_plugin, second_plugin]):
        context = GraphPlanContext(
            runtime_snapshot=graph_executor_env.snapshot,
            policy=GraphPluginPolicy(default_severity="fatal", fail_fast=True),
        )
        plan = plan_graph_plugin_run(
            plugin_names=[fatal_plugin.metadata.name, second_plugin.metadata.name],
            context=context,
        )

        executor_context = GraphExecutorContext(
            gateway=graph_executor_env.gateway,
            snapshot=graph_executor_env.snapshot,
        )

        # Use GraphPluginExecutor directly
        base_policy = BaseExecutionPolicy(
            default_severity="fatal",
            fail_fast=True,
        )
        graph_executor = GraphPluginExecutor(
            policy=base_policy,
            scope=plan.scope,
        )

        report = graph_executor.execute(
            executor_ctx=executor_context,
            plugins=plan.plugins,
            run_id=plan.run_id,
            settings_by_plugin=plan.settings_by_plugin,
        )

        expect_true(report.fatal_error)
        # Only the fatal plugin should have a record
        expect_length(report.records, 1)
        expect_equal(report.records[0].plugin_name, "fatal_first")


def test_graph_run_report_captures_all_fields() -> None:
    """GraphRunReport correctly captures all execution fields."""
    now = datetime.now(tz=UTC)
    record = PluginExecutionRecord(
        plugin_name="test_plugin",
        status="succeeded",
        started_at=now,
        ended_at=now,
        duration_ms=100.0,
        meta={"row_counts": {"t": 10}},
    )

    report = GraphRunReport(
        run_id="run-123",
        repo="test/repo",
        commit="abc123",
        records=(record,),
        duration_ms=150.0,
        started_at=now,
        ended_at=now,
        fatal_error=False,
        manifest={"test_plugin": {"executed_at": now}},
    )

    expect_equal(report.run_id, "run-123")
    expect_equal(report.repo, "test/repo")
    expect_equal(report.commit, "abc123")
    expect_length(report.records, 1)
    expect_equal(report.success_count, 1)
    expect_true(not report.fatal_error)
    expect_in("test_plugin", report.manifest)


def test_plugin_fatal_error_preserves_context() -> None:
    """PluginFatalError preserves record and exception message."""
    now = datetime.now(tz=UTC)
    record = PluginExecutionRecord(
        plugin_name="fatal_plugin",
        status="failed",
        started_at=now,
        ended_at=now,
        duration_ms=50.0,
        error="Original error",
    )
    original = ValueError("Original exception message")

    exc = PluginFatalError(record, original)

    expect_equal(exc.record.plugin_name, "fatal_plugin")
    expect_equal(exc.record.status, "failed")
    expect_true("Original exception message" in str(exc))


def test_graph_plugin_executor_batch_executes_multiple(
    graph_executor_env: GraphTestEnv,
) -> None:
    """Batch execution runs multiple plugins and reports results."""
    plugins = [
        _make_test_plugin("batch_p1", {"succeed": True}),
        _make_test_plugin("batch_p2", {"succeed": True}),
    ]

    with plugin_registrar(plugins):
        executor_context = GraphExecutorContext(
            gateway=graph_executor_env.gateway,
            snapshot=graph_executor_env.snapshot,
        )

        graph_executor = GraphPluginExecutor()

        report = graph_executor.execute(
            executor_ctx=executor_context,
            plugins=tuple(plugins),
            run_id="test-batch-run",
        )

        expect_equal(report.success_count, REPORT_SUCCESS_COUNT)
        expect_equal(report.failure_count, 0)


def test_graph_plugin_executor_batch_with_mixed_results(
    graph_executor_env: GraphTestEnv,
) -> None:
    """Batch execution handles mixed success and failure results."""
    plugins = [
        _make_test_plugin("batch_success", {"succeed": True}),
        _make_test_plugin("batch_fail", {"succeed": False}),
    ]

    with plugin_registrar(plugins):
        executor_context = GraphExecutorContext(
            gateway=graph_executor_env.gateway,
            snapshot=graph_executor_env.snapshot,
        )

        graph_executor = GraphPluginExecutor()

        report = graph_executor.execute(
            executor_ctx=executor_context,
            plugins=tuple(plugins),
            run_id="test-mixed-run",
        )

        expect_equal(report.success_count, REPORT_MIXED_SUCCESS_COUNT)
        expect_equal(report.failure_count, REPORT_FAILURE_COUNT)
