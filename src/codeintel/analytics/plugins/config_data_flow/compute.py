"""Config data flow plugin.

This plugin tracks configuration key usage and data flow at the function level.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import networkx as nx

from codeintel.analytics.graphs import compute_config_data_flow, compute_config_graph_metrics
from codeintel.analytics.parsing.ast_cache import FunctionAstLoadRequest, load_function_asts
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config.steps_graphs import ConfigDataFlowStepConfig

if TYPE_CHECKING:
    from codeintel.analytics.parsing.ast_cache import FunctionAst
    from codeintel.build.context import TargetExecutionContext

log = logging.getLogger(__name__)


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
        _ = self  # Protocol method requires instance

        cfg = ConfigDataFlowStepConfig(
            snapshot=ctx.snapshot,
        )

        graph_runtime = ctx.resources.graph_runtime

        # Get the call graph from graph runtime
        call_graph: nx.DiGraph = nx.DiGraph()
        if graph_runtime is not None:
            try:
                call_graph = graph_runtime.ensure_call_graph()
            except (RuntimeError, ValueError, OSError) as e:
                log.warning("Failed to load call graph: %s", e)

        # Get AST data from catalog if available
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
            # Compute config data flow
            compute_config_data_flow(
                ctx.gateway,
                cfg,
                call_graph=call_graph,
                ast_by_goid=ast_by_goid,
                missing_goids=missing_goids,
            )

            # Compute config graph metrics (keys, modules, projections)
            compute_config_graph_metrics(
                ctx.gateway,
                repo=ctx.repo,
                commit=ctx.commit,
                runtime=graph_runtime,
            )

        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Config data flow computation failed: {e}")

        return TargetResult.succeeded()


__all__ = ["ConfigDataFlowPlugin"]
