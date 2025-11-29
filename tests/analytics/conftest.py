"""Shared fixtures for analytics plugin tests."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Literal

import pytest

from codeintel.analytics.graph_runtime import GraphRuntimeOptions, resolve_graph_runtime
from codeintel.analytics.graph_service_runtime import (
    GraphPluginRunOptions,
    GraphPluginRunReport,
    GraphServiceRuntime,
)
from codeintel.analytics.graphs.contracts import PluginContractResult
from codeintel.analytics.graphs.plugins import (
    GraphMetricExecutionContext,
    GraphMetricPlugin,
    GraphPluginResult,
    register_graph_metric_plugin,
    unregister_graph_metric_plugin,
)
from codeintel.config.primitives import GraphBackendConfig, SnapshotRef
from codeintel.config.steps_graphs import GraphMetricsStepConfig, GraphPluginPolicy
from codeintel.storage.gateway import StorageGateway, open_memory_gateway


class PluginTestHarness:
    """Utility to register plugins and assert idempotent manifest behavior."""

    def __init__(self, tmp_path: Path) -> None:
        self._tmp_path = tmp_path
        snapshot = SnapshotRef(repo="demo/repo", commit="deadbeef", repo_root=Path())
        self.gateway: StorageGateway = open_memory_gateway(
            apply_schema=True, ensure_views=True, validate_schema=True
        )
        runtime = resolve_graph_runtime(
            self.gateway,
            snapshot,
            GraphRuntimeOptions(snapshot=snapshot, backend=GraphBackendConfig()),
        )
        self.service = GraphServiceRuntime(gateway=self.gateway, runtime=runtime)
        self.cfg = GraphMetricsStepConfig(snapshot=snapshot)
        self._registered: set[str] = set()

    def register(self, plugin: GraphMetricPlugin) -> None:
        """Register a plugin for the duration of the harness lifecycle."""
        register_graph_metric_plugin(plugin)
        self._registered.add(plugin.name)

    def build_manifest_path(self, name: str) -> Path:
        """
        Construct a manifest path rooted in the harness tmp dir.

        Returns
        -------
        Path
            Location for the manifest file.
        """
        return self._tmp_path / f"{name}.json"

    def run_plugins_with_cfg(
        self,
        plugin_names: Sequence[str],
        cfg: GraphMetricsStepConfig | None = None,
        run_options: GraphPluginRunOptions | None = None,
        manifest_name: str | None = None,
    ) -> GraphPluginRunReport:
        """
        Execute plugins with optional config and manifest.

        Parameters
        ----------
        plugin_names:
            Plugins to execute.
        cfg:
            Graph metrics config override. Defaults to harness config.
        run_options:
            Execution options override. Defaults to harness defaults with manifest path.
        manifest_name:
            Optional manifest file stem; defaults to 'manifest'.

        Returns
        -------
        GraphPluginRunReport
            Report of the executed plugins.
        """
        manifest_path = self.build_manifest_path(manifest_name or "manifest")
        if run_options is None:
            options = GraphPluginRunOptions(manifest_path=manifest_path)
        elif run_options.manifest_path is None:
            options = GraphPluginRunOptions(
                plugin_options=run_options.plugin_options,
                manifest_path=manifest_path,
                scope=run_options.scope,
                dry_run=run_options.dry_run,
            )
        else:
            options = run_options
        return self.service.run_plugins(plugin_names, cfg=cfg or self.cfg, run_options=options)

    def cleanup(self) -> None:
        """Unregister all plugins registered by this harness."""
        for name in list(self._registered):
            unregister_graph_metric_plugin(name)
            self._registered.discard(name)

    def run_twice_assert_idempotent(self, plugin_name: str) -> None:
        """
        Run plugin twice with skip_on_unchanged policy and assert skip on second run.

        Raises
        ------
        AssertionError
            If the first run fails or the second run does not skip due to unchanged inputs.
        """
        manifest_path = self._tmp_path / f"{plugin_name}-manifest.json"
        policy_cfg = GraphMetricsStepConfig(
            snapshot=self.cfg.snapshot,
            plugin_policy=GraphPluginPolicy(skip_on_unchanged=True),
        )
        first_report = self.service.run_plugins(
            (plugin_name,),
            cfg=policy_cfg,
            run_options=GraphPluginRunOptions(manifest_path=manifest_path),
        )
        first_record = first_report.records[0]
        if first_record.status != "succeeded":
            message = "First run should succeed to seed manifest"
            raise AssertionError(message)
        second_report = self.service.run_plugins(
            (plugin_name,),
            cfg=policy_cfg,
            run_options=GraphPluginRunOptions(manifest_path=manifest_path),
        )
        second_record = second_report.records[0]
        if second_record.status != "skipped" or second_record.skipped_reason != "unchanged":
            message = "Second run should skip due to unchanged manifest/row_counts"
            raise AssertionError(message)

    def run_with_contracts(
        self,
        plugin_name: str,
        contract_assertions: Callable[[Iterable[PluginContractResult]], None],
    ) -> None:
        """
        Execute plugin and allow caller to assert on emitted contracts.

        Parameters
        ----------
        plugin_name:
            Registered plugin to execute.
        contract_assertions:
            Callback used to assert on emitted contracts.
        """
        report = self.service.run_plugins((plugin_name,), cfg=self.cfg)
        contracts = report.records[0].contracts
        contract_assertions(contracts)


def make_provider_plugin(
    name: str = "provider", provides: tuple[str, ...] = ("res",)
) -> GraphMetricPlugin:
    """
    Create a no-op provider plugin advertising provided resources.

    Returns
    -------
    GraphMetricPlugin
        Plugin that reports provided resources.
    """

    def _run(_ctx: GraphMetricExecutionContext) -> GraphPluginResult:
        return GraphPluginResult(row_counts={"analytics.provider": 1})

    return GraphMetricPlugin(
        name=name,
        description="provider",
        stage="core",
        enabled_by_default=False,
        run=_run,
        provides=provides,
    )


def make_consumer_plugin(
    name: str = "consumer",
    requires: tuple[str, ...] = ("res",),
    depends_on: tuple[str, ...] = (),
) -> GraphMetricPlugin:
    """
    Create a consumer plugin that requires resources/dependencies.

    Returns
    -------
    GraphMetricPlugin
        Plugin that consumes provided resources.
    """

    def _run(_ctx: GraphMetricExecutionContext) -> GraphPluginResult:
        return GraphPluginResult(row_counts={"analytics.consumer": 1})

    return GraphMetricPlugin(
        name=name,
        description="consumer",
        stage="core",
        enabled_by_default=False,
        run=_run,
        requires=requires,
        depends_on=depends_on,
    )


def make_scope_plugin(
    name: str = "scope_plugin",
    *,
    require_scope: bool = True,
    severity: Literal["fatal", "soft_fail", "skip_on_error"] = "fatal",
) -> GraphMetricPlugin:
    """
    Create a scope-aware plugin that can enforce presence of scope.

    Returns
    -------
    GraphMetricPlugin
        Plugin configured for scope-aware execution.
    """

    def _run(ctx: GraphMetricExecutionContext) -> GraphPluginResult:
        if require_scope and not (ctx.scope.paths or ctx.scope.modules or ctx.scope.time_window):
            message = "scope required"
            raise ValueError(message)
        return GraphPluginResult(row_counts={"analytics.scope": 1})

    return GraphMetricPlugin(
        name=name,
        description="scope aware",
        stage="core",
        enabled_by_default=False,
        run=_run,
        scope_aware=True,
        supported_scopes=("paths", "modules", "time_window"),
        severity=severity,
    )


def make_isolation_plugin(
    name: str = "isolation_plugin",
    isolation_kind: Literal["process", "thread"] | None = "process",
) -> GraphMetricPlugin:
    """
    Create an isolation-capable plugin that returns row counts.

    Returns
    -------
    GraphMetricPlugin
        Plugin configured to run in isolation.
    """

    def _run(_ctx: GraphMetricExecutionContext) -> GraphPluginResult:
        return GraphPluginResult(row_counts={"analytics.isolation": 1})

    return GraphMetricPlugin(
        name=name,
        description="isolation capable",
        stage="core",
        enabled_by_default=False,
        run=_run,
        requires_isolation=True,
        isolation_kind=isolation_kind,
    )


@pytest.fixture(name="plugin_harness")
def _plugin_harness(tmp_path: Path) -> Iterator[PluginTestHarness]:
    """Yield a plugin test harness with automatic cleanup.

    Yields
    ------
    PluginTestHarness
        Harness configured with in-memory gateway/runtime.
    """
    harness = PluginTestHarness(tmp_path)
    try:
        yield harness
    finally:
        harness.cleanup()
