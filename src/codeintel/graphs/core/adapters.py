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

from pydantic import BaseModel

from codeintel.build.context import ContextResources, TargetExecutionContext
from codeintel.build.parameters import EMPTY_PARAMETERS
from codeintel.build.registry import get_target_graph
from codeintel.config.primitives import BuildPaths
from codeintel.core.plugins.types.protocol import PluginResourceHints
from codeintel.core.plugins.types.result import PluginResult
from codeintel.graphs.core.protocol import (
    GraphPluginMetadataConfig,
    create_graph_metadata,
)
from codeintel.graphs.engine import GraphKind

if TYPE_CHECKING:
    from codeintel.build.context import TargetResult
    from codeintel.build.plugin import TargetPlugin
    from codeintel.build.targets import OutputTarget
    from codeintel.graphs.core.context import GraphPluginExecutionContext
    from codeintel.graphs.core.protocol import (
        GraphPluginKind,
        GraphPluginMetadata,
        GraphPluginStage,
    )

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
    "graph_metrics.secondary": "graph_metrics",
    "graph_validation": "graph_validation",
}

_PLUGIN_PRODUCES_GRAPH_KINDS: dict[str, tuple[GraphKind, ...]] = {
    "callgraph": (GraphKind.CALL_GRAPH,),
    "import_graph": (GraphKind.IMPORT_GRAPH,),
    "cfg_dfg": (GraphKind.CFG_GRAPH,),
    "symbol_uses": (GraphKind.SYMBOL,),
}

_PLUGIN_REQUIRES_GRAPH_KINDS: dict[str, tuple[GraphKind, ...]] = {
    "graph_metrics.core": (GraphKind.CALL_GRAPH, GraphKind.IMPORT_GRAPH),
    "graph_metrics.secondary": (GraphKind.CALL_GRAPH, GraphKind.IMPORT_GRAPH),
    "graph_validation": (GraphKind.ALL,),
}


def _tuple_attr(plugin: TargetPlugin, attribute: str) -> tuple[str, ...]:
    """Return a plugin attribute normalized to a tuple of strings.

    Returns
    -------
    tuple[str, ...]
        Normalized tuple value for the attribute.
    """
    value = getattr(plugin, attribute, ())
    if value is None:
        return ()
    if isinstance(value, tuple):
        return tuple(str(item) for item in value)
    if isinstance(value, (list, set)):
        return tuple(str(item) for item in value)
    if isinstance(value, str):
        return (value,)
    return (str(value),)


def _options_model(plugin: TargetPlugin) -> type[BaseModel] | None:
    """Extract an options model from a plugin if provided.

    Returns
    -------
    type[BaseModel] | None
        Options model class when declared, otherwise None.
    """
    candidate = getattr(plugin, "plugin_options_model", None)
    if isinstance(candidate, type) and issubclass(candidate, BaseModel):
        return candidate
    return None


def _plugin_resource_hints(plugin: TargetPlugin) -> PluginResourceHints | None:
    """Extract resource hints defined on the plugin, if any.

    Returns
    -------
    PluginResourceHints | None
        Plugin-defined resource hints or None if not set.
    """
    hints = getattr(plugin, "plugin_resource_hints", None)
    if isinstance(hints, PluginResourceHints):
        return hints
    return None


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
        kind, stage = self._determine_kind_stage(plugin_name)
        target = self._resolve_target()
        metadata_config = self._build_metadata_config(
            plugin_name=plugin_name,
            kind=kind,
            plugin=self._plugin,
            target=target,
        )

        return create_graph_metadata(
            name=plugin_name,
            description=description,
            kind=kind,
            stage=stage,
            config=metadata_config,
        )

    def _determine_kind_stage(
        self,
        plugin_name: str,
    ) -> tuple[GraphPluginKind, GraphPluginStage]:
        """Resolve the plugin kind and stage, honoring overrides when provided.

        Returns
        -------
        tuple[GraphPluginKind, GraphPluginStage]
            Kind and stage resolved from overrides or known defaults.
        """
        kind: GraphPluginKind = "builder"
        stage: GraphPluginStage = "edges"
        if self._kind_override is not None and self._stage_override is not None:
            kind = self._kind_override
            stage = self._stage_override
        elif plugin_name in _PLUGIN_KIND_STAGE_MAP:
            kind, stage = _PLUGIN_KIND_STAGE_MAP[plugin_name]

        return kind, stage

    def _build_metadata_config(
        self,
        *,
        plugin_name: str,
        kind: GraphPluginKind,
        plugin: TargetPlugin,
        target: OutputTarget,
    ) -> GraphPluginMetadataConfig:
        """Construct GraphPluginMetadataConfig from target metadata.

        Returns
        -------
        GraphPluginMetadataConfig
            Populated configuration for metadata construction.
        """
        produces_graph_kinds = _PLUGIN_PRODUCES_GRAPH_KINDS.get(plugin_name, ())
        requires_graph_kinds = _PLUGIN_REQUIRES_GRAPH_KINDS.get(plugin_name, ())
        if kind == "metric" and not requires_graph_kinds:
            requires_graph_kinds = (GraphKind.ALL,)
        resource_hints = _plugin_resource_hints(plugin) or PluginResourceHints(
            max_runtime_ms=target.execution.max_runtime_ms,
            max_memory_mb=None,
            cpu_intensive=target.execution.cpu_intensive,
            io_intensive=target.execution.io_intensive,
            requires_gpu=False,
            priority=0,
        )
        isolation_kind = target.execution.isolation
        depends_on = self._resolve_dependency_plugins(target.dependencies)
        plugin_depends_on = _tuple_attr(plugin, "plugin_depends_on")
        if plugin_depends_on:
            depends_on = (*depends_on, *plugin_depends_on)
        provides = _tuple_attr(plugin, "plugin_provides")
        requires = _tuple_attr(plugin, "plugin_requires")
        cache_populates = _tuple_attr(plugin, "plugin_cache_populates")
        cache_consumes = _tuple_attr(plugin, "plugin_cache_consumes")
        contract_checkers = _tuple_attr(plugin, "plugin_contract_checkers")
        options_model = _options_model(plugin)
        options_default = getattr(plugin, "plugin_options_default", None)
        supports_incremental = bool(
            getattr(plugin, "plugin_supports_incremental", target.execution.supports_incremental)
        )

        return GraphPluginMetadataConfig(
            version_hash=self._plugin.plugin_version,
            produces_tables=target.table_keys,
            row_count_tables=target.table_keys,
            produces_graph_kinds=produces_graph_kinds,
            requires_graph_kinds=requires_graph_kinds,
            resource_hints=resource_hints,
            supports_incremental=supports_incremental,
            isolation_kind=isolation_kind,
            requires_isolation=isolation_kind != "none",
            depends_on=depends_on,
            provides=provides,
            requires=requires,
            cache_populates=cache_populates,
            cache_consumes=cache_consumes,
            contract_checkers=contract_checkers,
            options_model=options_model,
            options_default=options_default,
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
    def _resolve_dependency_plugins(dependencies: tuple[str, ...]) -> tuple[str, ...]:
        """Translate target dependencies to their plugin names.

        Returns
        -------
        tuple[str, ...]
            Plugin names corresponding to dependency targets.
        """
        if not dependencies:
            return ()
        graph = get_target_graph()
        return tuple(graph.get(dep).plugin for dep in dependencies)

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
