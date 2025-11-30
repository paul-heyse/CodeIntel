"""Graph runtime context utilities shared across analytics graph metrics."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING
from uuid import uuid4

from codeintel.analytics.context import AnalyticsContext
from codeintel.analytics.graph_runtime import GraphRuntime
from codeintel.analytics.graphs.runtime import (
    DEFAULT_BETWEENNESS_SAMPLE,
    GraphContext,
    GraphContextCaps,
    GraphContextSpec,
    GraphPluginRunOptions,
    GraphPluginRunRecord,
    GraphPluginRunReport,
    GraphRuntimeTelemetry,
    PluginExecutionPlan,
    PluginExecutionSettings,
    PluginFatalError,
    build_graph_context,
    resolve_graph_context,
)
from codeintel.analytics.graphs.runtime.analytics_adapter import (
    analytics_to_graph_run,
    graph_run_to_analytics,
)
from codeintel.analytics.graphs.runtime.manifest import load_prior_manifest, write_manifest
from codeintel.analytics.plugin_runtime import plan_analytics_plugin_run, run_analytics_plugins
from codeintel.analytics.runtime_manifest import encode_manifest
from codeintel.config import GraphMetricsStepConfig
from codeintel.config.steps_graphs import GraphPluginPolicy, GraphRunScope

if TYPE_CHECKING:
    from codeintel.graphs.function_catalog_service import FunctionCatalogProvider
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


@dataclass
class GraphServiceRuntime:
    """Lightweight orchestrator for graph analytics using a shared runtime."""

    gateway: StorageGateway
    runtime: GraphRuntime
    analytics_context: AnalyticsContext | None = None
    catalog_provider: FunctionCatalogProvider | None = None
    telemetry: GraphRuntimeTelemetry | None = None

    def run_plugins(
        self,
        plugin_names: Sequence[str],
        *,
        cfg: GraphMetricsStepConfig | None = None,
        target: tuple[str, str] | None = None,
        run_options: GraphPluginRunOptions | None = None,
    ) -> GraphPluginRunReport:
        """
        Execute a sequence of graph metric plugins against this runtime.

        Parameters
        ----------
        plugin_names
            Names of plugins to execute, in order.
        cfg
            Optional graph metrics configuration; provided to plugins via
            GraphMetricExecutionContext.
        target
            Optional (repo, commit) override when config is not supplied.
        run_options
            Optional execution controls (per-plugin options, manifest path, scopes).

        Raises
        ------
        ValueError
            If neither a config nor runtime snapshot is available to derive repo/commit.
        PluginFatalError
            When a fatal plugin failure occurs and fail-fast is enabled.

        Returns
        -------
        GraphPluginRunReport
            Telemetry for executed plugins.
        """
        policy = cfg.plugin_policy if cfg is not None else GraphPluginPolicy()
        if run_options is not None and run_options.dry_run is not None:
            policy = replace(policy, dry_run=run_options.dry_run)
        manifest_path = run_options.manifest_path if run_options is not None else None
        prior_manifest = load_prior_manifest(manifest_path)

        if cfg is None and target is None and self.runtime.options.snapshot is None:
            message = "Graph runtime missing snapshot; cannot derive repo/commit"
            raise ValueError(message)

        repo, commit = self._resolve_target(cfg=cfg, target=target)
        scope = self._resolve_scope(cfg=cfg, run_options=run_options)
        cfg_options = cfg.plugin_options if cfg is not None else {}
        runtime_options = (run_options.plugin_options if run_options is not None else {}) or {}
        run_id = uuid4().hex

        plan = plan_analytics_plugin_run(
            plugin_names=plugin_names,
            policy=policy,
            repo=repo,
            commit=commit,
            scope=scope,
            prior_manifest=prior_manifest or {},
            cfg_options=cfg_options,
            runtime_options=runtime_options,
            run_id=run_id,
        )

        analytics_report = run_analytics_plugins(
            plan=plan,
            gateway=self.gateway,
            analytics_context=self.analytics_context,
            graph_runtime=self.runtime,
            cfgs={"graph": cfg} if cfg is not None else {},
            extra={},
            catalog_provider=self.catalog_provider,
        )

        report = analytics_to_graph_run(analytics_report)
        if manifest_path is not None:
            manifest_payload = encode_manifest(graph_run_to_analytics(report))
            write_manifest(manifest_path, manifest_payload)
        return report

    def _resolve_target(
        self, cfg: GraphMetricsStepConfig | None, target: tuple[str, str] | None
    ) -> tuple[str, str]:
        if cfg is not None:
            return cfg.repo, cfg.commit
        if target is not None:
            return target
        snapshot = self.runtime.options.snapshot
        if snapshot is None:
            message = "Graph runtime missing snapshot; cannot derive repo/commit"
            raise ValueError(message)
        return snapshot.repo, snapshot.commit

    def _resolve_scope(
        self, cfg: GraphMetricsStepConfig | None, run_options: GraphPluginRunOptions | None
    ) -> GraphRunScope:
        if run_options is not None and run_options.scope is not None:
            return run_options.scope
        if cfg is not None:
            return cfg.scope
        return GraphRunScope()


__all__ = [
    "DEFAULT_BETWEENNESS_SAMPLE",
    "GraphContext",
    "GraphContextCaps",
    "GraphContextSpec",
    "GraphPluginRunOptions",
    "GraphPluginRunRecord",
    "GraphPluginRunReport",
    "GraphRuntimeTelemetry",
    "GraphServiceRuntime",
    "PluginExecutionPlan",
    "PluginExecutionSettings",
    "PluginFatalError",
    "build_graph_context",
    "compute_input_hash",
    "compute_options_hash",
    "resolve_graph_context",
]
