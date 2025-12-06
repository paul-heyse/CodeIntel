"""Adapters bridging TargetPlugin to GraphPluginProtocol.

This module provides adapters that wrap TargetPlugin instances as
GraphPluginProtocol implementations, enabling registration with
the GraphPluginRegistry.

The adapters allow the existing GraphPluginRegistry infrastructure
to work with build system plugins without modification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.core.plugins.types.result import PluginResult
from codeintel.graphs.core.protocol import (
    GraphPluginKind,
    GraphPluginMetadata,
    GraphPluginStage,
    create_graph_metadata,
)

if TYPE_CHECKING:
    from codeintel.build.plugin import TargetPlugin
    from codeintel.graphs.core.context import GraphPluginExecutionContext


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


class TargetPluginAdapter:
    """Adapter wrapping a TargetPlugin as GraphPluginProtocol.

    This adapter enables TargetPlugin instances to be registered with
    the GraphPluginRegistry by providing the required metadata property
    and execute method signature.

    The execute method returns a placeholder result since actual execution
    goes through the build system's TargetExecutionContext, not the graph
    plugin execution context. This adapter is primarily for planning and
    introspection purposes.

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
        """Execute method satisfying GraphPluginProtocol.

        This method returns a placeholder result. Actual plugin execution
        goes through the build system using TargetExecutionContext.

        Parameters
        ----------
        ctx
            Graph plugin execution context (not used for actual execution).

        Returns
        -------
        PluginResult
            Placeholder success result.
        """
        # Execution goes through the build system, not this adapter.
        # This satisfies the protocol for planning/introspection.
        # Reference self to indicate this is an instance method (protocol requirement).
        _ = (ctx, self._plugin)
        return PluginResult.ok()


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
