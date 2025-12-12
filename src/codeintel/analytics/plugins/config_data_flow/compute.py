"""Config data flow plugin.

This plugin tracks configuration key usage and data flow at the function level.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import networkx as nx

from codeintel.analytics.graphs import compute_config_data_flow, compute_config_graph_metrics
from codeintel.analytics.parsing.ast_cache import FunctionAstLoadRequest, load_function_asts
from codeintel.analytics.plugins._metadata import to_plugin_metadata
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config.steps_graphs import ConfigDataFlowStepConfig
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain

if TYPE_CHECKING:
    from codeintel.analytics.parsing.ast_cache import FunctionAst
    from codeintel.build.context import TargetExecutionContext
    from codeintel.core.plugins.types.protocol import PluginMetadata

log = logging.getLogger(__name__)


CONFIG_DATA_FLOW_METADATA = CorePluginMetadata(
    name="analytics.config_data_flow",
    version="3.0.0",
    description="Track configuration key usage and data flow at the function level.",
    domain=PluginDomain.ANALYTICS,
    kind="metric",
    stage="config",
    provides=(
        "analytics.config_data_flow",
        "analytics.config_graph_metrics_keys",
        "analytics.config_graph_metrics_modules",
        "analytics.config_projection_key_edges",
        "analytics.config_projection_module_edges",
    ),
    requires=("graph.call_graph_edges", "core.goids"),
    produces_tables=(
        "analytics.config_data_flow",
        "analytics.config_graph_metrics_keys",
        "analytics.config_graph_metrics_modules",
        "analytics.config_projection_key_edges",
        "analytics.config_projection_module_edges",
    ),
    consumes_tables=("graph.call_graph_edges", "core.goids"),
)


class ConfigDataFlowPlugin(TargetPlugin):
    """Track configuration key usage and data flow at the function level.

    Tracks configuration usage:
    - Config key reads at function level
    - Config key propagation through calls
    - Function-level config dependencies

    Outputs
    -------
    - analytics.config_data_flow: Config data flow tracking
    - analytics.config_graph_metrics_keys: Config key graph metrics
    - analytics.config_graph_metrics_modules: Config module graph metrics
    - analytics.config_projection_key_edges: Config key projection edges
    - analytics.config_projection_module_edges: Config module projection edges
    """

    plugin_name: ClassVar[str] = "config_data_flow"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = (
        "Track configuration key usage and data flow at the function level."
    )
    _core_metadata: ClassVar[CorePluginMetadata] = CONFIG_DATA_FLOW_METADATA

    @property
    def metadata(self) -> PluginMetadata:
        """Return protocol-compatible metadata."""
        return to_plugin_metadata(self._core_metadata)

    @property
    def core_metadata(self) -> CorePluginMetadata:
        """Return canonical metadata."""
        return self._core_metadata

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute the plugin.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        TargetResult
            Execution result.
        """
        _ = self

        cfg = ConfigDataFlowStepConfig(
            snapshot=ctx.snapshot,
        )

        graph_runtime = ctx.resources.graph_runtime

        call_graph: nx.DiGraph = nx.DiGraph()
        if graph_runtime is not None:
            try:
                call_graph = graph_runtime.ensure_call_graph()
            except (RuntimeError, ValueError, OSError) as e:
                log.warning("Failed to load call graph: %s", e)

        ast_by_goid: dict[int, FunctionAst] = {}
        missing_goids: set[int] = set()
        catalog_provider = ctx.resources.catalog
        if catalog_provider is not None:
            request = FunctionAstLoadRequest(
                repo=ctx.repo,
                commit=ctx.commit,
                repo_root=ctx.snapshot.repo_root,
                catalog_provider=catalog_provider,
            )
            ast_by_goid, missing_goids = load_function_asts(ctx.gateway, request)

        try:
            compute_config_data_flow(
                ctx.gateway,
                cfg,
                call_graph=call_graph,
                ast_by_goid=ast_by_goid,
                missing_goids=missing_goids,
            )

            compute_config_graph_metrics(
                ctx.gateway,
                repo=ctx.repo,
                commit=ctx.commit,
                runtime=graph_runtime,
            )

        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Config data flow computation failed: {e}")

        return TargetResult.succeeded()


__all__ = ["CONFIG_DATA_FLOW_METADATA", "ConfigDataFlowPlugin"]
