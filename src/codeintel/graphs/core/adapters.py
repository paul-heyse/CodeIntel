"""Adapters bridging TargetPlugin to GraphPluginProtocol.

This module provides adapters that wrap TargetPlugin instances as
GraphPluginProtocol implementations, enabling registration with
the GraphPluginRegistry.

The adapters bridge the GraphPluginExecutionContext to TargetExecutionContext,
allowing graph plugins implemented as TargetPlugin to be executed through
the graph runtime infrastructure.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from codeintel.build.context import ContextResources, TargetExecutionContext
from codeintel.build.parameters import EMPTY_PARAMETERS
from codeintel.build.registry import get_target_graph
from codeintel.config.primitives import BuildPaths
from codeintel.core.plugins.types.result import PluginResult
from codeintel.graphs.core.protocol import (
    GraphPluginKind,
    GraphPluginMetadata,
    GraphPluginStage,
    create_graph_metadata,
)

if TYPE_CHECKING:
    from codeintel.build.context import TargetResult
    from codeintel.build.plugin import TargetPlugin
    from codeintel.build.targets import OutputTarget
    from codeintel.graphs.core.context import GraphPluginExecutionContext

log = logging.getLogger(__name__)


# Mapping of plugin names to their kind and stage for metadata creation
_PLUGIN_KIND_STAGE_MAP: dict[str, tuple[GraphPluginKind, GraphPluginStage]] = {
    # Builders
    "goid_builder": ("builder", "goid"),
    "callgraph": ("builder", "edges"),
    "import_graph": ("builder", "edges"),
    "cfg_dfg": ("builder", "cfg"),
    "symbol_uses": ("builder", "symbol"),
    # Metrics
    "graph_metrics.core": ("metric", "core"),
    "graph_metrics.secondary": ("metric", "core"),
    # Validation
    "graph_validation": ("validation", "validation"),
}

# Mapping of plugin names to their corresponding target names in the build graph
_PLUGIN_TO_TARGET_MAP: dict[str, str] = {
    "goid_builder": "goids",
    "callgraph": "call_graph",
    "import_graph": "import_graph",
    "cfg_dfg": "cfg",
    "symbol_uses": "symbol_uses",
    "graph_metrics.core": "graph_metrics",
    "graph_metrics.secondary": "graph_metrics_secondary",
    "graph_validation": "graph_validation",
}


class TargetPluginAdapter:
    """Adapter wrapping a TargetPlugin as GraphPluginProtocol.

    This adapter enables TargetPlugin instances to be registered with
    the GraphPluginRegistry by providing the required metadata property
    and execute method signature.

    The execute method bridges the GraphPluginExecutionContext to a
    TargetExecutionContext and runs the actual plugin via asyncio.run().
    This allows graph plugins implemented as TargetPlugin to be executed
    through the graph runtime infrastructure with all its features
    (timeouts, retries, manifest tracking, telemetry).

    Attributes
    ----------
    _plugin
        The wrapped TargetPlugin instance.
    _metadata
        Cached GraphPluginMetadata derived from the plugin.
    """

    _plugin: TargetPlugin
    _kind_override: GraphPluginKind | None
    _stage_override: GraphPluginStage | None
    _metadata: GraphPluginMetadata

    def __init__(
        self,
        target_plugin: TargetPlugin,
        *,
        kind: GraphPluginKind | None = None,
        stage: GraphPluginStage | None = None,
    ) -> None:
        """Initialize the adapter with a TargetPlugin.

        Parameters
        ----------
        target_plugin
            The TargetPlugin instance to wrap.
        kind
            Override for plugin kind (builder, metric, validation).
        stage
            Override for plugin stage.
        """
        self._plugin = target_plugin
        self._kind_override = kind
        self._stage_override = stage
        self._metadata = self._create_metadata()

    @property
    def metadata(self) -> GraphPluginMetadata:
        """Return plugin metadata for GraphPluginProtocol compliance.

        Returns
        -------
        GraphPluginMetadata
            Metadata derived from the wrapped TargetPlugin.
        """
        return self._metadata

    def _create_metadata(self) -> GraphPluginMetadata:
        """Build GraphPluginMetadata from TargetPlugin class variables.

        Returns
        -------
        GraphPluginMetadata
            Populated metadata instance.
        """
        plugin_name = self._plugin.plugin_name
        description = self._plugin.plugin_description or f"Plugin: {plugin_name}"

        # Determine kind and stage from map or overrides
        kind: GraphPluginKind = "builder"
        stage: GraphPluginStage = "edges"
        if self._kind_override is not None and self._stage_override is not None:
            kind = self._kind_override
            stage = self._stage_override
        elif plugin_name in _PLUGIN_KIND_STAGE_MAP:
            kind, stage = _PLUGIN_KIND_STAGE_MAP[plugin_name]

        return create_graph_metadata(
            name=plugin_name,
            description=description,
            kind=kind,
            stage=stage,
            severity="fatal",
            enabled_by_default=True,
            version_hash=self._plugin.plugin_version,
        )

    def execute(self, ctx: GraphPluginExecutionContext) -> PluginResult:
        """Execute the wrapped TargetPlugin via context bridging.

        This method bridges the GraphPluginExecutionContext to a
        TargetExecutionContext and executes the actual plugin.

        Parameters
        ----------
        ctx
            Graph plugin execution context.

        Returns
        -------
        PluginResult
            Result from the wrapped plugin execution.
        """
        try:
            target_ctx = self._build_target_context(ctx)
            result = asyncio.run(self._plugin.execute(target_ctx))
            return self._convert_result(result)
        except Exception:
            log.exception(
                "adapter.execute.error plugin=%s",
                self._plugin.plugin_name,
            )
            return PluginResult.fail("Plugin execution failed")

    def _build_target_context(
        self,
        ctx: GraphPluginExecutionContext,
    ) -> TargetExecutionContext:
        """Build a TargetExecutionContext from GraphPluginExecutionContext.

        Parameters
        ----------
        ctx
            Graph plugin execution context.

        Returns
        -------
        TargetExecutionContext
            Context suitable for TargetPlugin execution.
        """
        target = self._resolve_target()

        # Build paths from snapshot if not provided in context
        paths = ctx.paths
        if paths is None:
            paths = BuildPaths.from_layout(repo_root=ctx.snapshot.repo_root)

        # Build resources from context
        resources = ContextResources(
            gateway=ctx.gateway,
            modules=(),
        )

        return TargetExecutionContext(
            target=target,
            snapshot=ctx.snapshot,
            paths=paths,
            resources=resources,
            parameters=EMPTY_PARAMETERS,
        )

    def _resolve_target(self) -> OutputTarget:
        """Resolve the OutputTarget for this plugin.

        Returns
        -------
        OutputTarget
            The target from the build graph.
        """
        plugin_name = self._plugin.plugin_name
        target_name = _PLUGIN_TO_TARGET_MAP.get(plugin_name, plugin_name)

        graph = get_target_graph()
        return graph.get(target_name)

    @staticmethod
    def _convert_result(result: TargetResult) -> PluginResult:
        """Convert a TargetResult to a PluginResult.

        Parameters
        ----------
        result
            Result from TargetPlugin execution.

        Returns
        -------
        PluginResult
            Equivalent PluginResult.
        """
        if result.success:
            return PluginResult.ok(row_counts=dict(result.row_counts))
        return PluginResult.fail(result.error_message or "Unknown error")


def adapt_target_plugin(
    plugin: TargetPlugin,
    *,
    kind: GraphPluginKind | None = None,
    stage: GraphPluginStage | None = None,
) -> TargetPluginAdapter:
    """Create an adapter for a TargetPlugin.

    Parameters
    ----------
    plugin
        The TargetPlugin to wrap.
    kind
        Optional override for plugin kind.
    stage
        Optional override for plugin stage.

    Returns
    -------
    TargetPluginAdapter
        Adapter implementing GraphPluginProtocol.
    """
    return TargetPluginAdapter(plugin, kind=kind, stage=stage)


__all__ = [
    "TargetPluginAdapter",
    "adapt_target_plugin",
]
