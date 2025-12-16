"""Plugin stubs for backward compatibility during migration.

This module provides stub plugin classes that can be used during the
migration from plugins to native Hamilton modules. These stubs are
intended for test helpers only and will be removed after Phase 6.

Phase 3: Migration stubs for graphs domain.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, ClassVar

from codeintel.build.plugin import MetadataPlugin
from codeintel.build.result import TargetResult
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext


def _deprecation_warning(name: str) -> None:
    """Emit a deprecation warning for plugin stub usage.

    Parameters
    ----------
    name
        Plugin name.
    """
    warnings.warn(
        f"{name} is deprecated. Use native Hamilton modules instead. "
        "See: codeintel.build.hamilton.native.graphs",
        DeprecationWarning,
        stacklevel=3,
    )


GOID_BUILDER_METADATA = CorePluginMetadata(
    name="graphs.goid_builder.stub",
    version="3.0.0",
    description="[DEPRECATED] Use native goids.py instead.",
    domain=PluginDomain.ANALYTICS,
    kind="builder",
    stage="graph",
    provides=("core.goids", "core.goid_crosswalk"),
    requires=("core.modules",),
    produces_tables=("core.goids", "core.goid_crosswalk"),
    consumes_tables=("core.modules",),
)


class GoidBuilderPlugin(MetadataPlugin):
    """Stub for GoidBuilderPlugin.

    .. deprecated::
        Use native Hamilton module instead:
        ``codeintel.build.hamilton.native.graphs.goids``
    """

    _core_metadata: ClassVar[CorePluginMetadata] = GOID_BUILDER_METADATA

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Return a skipped result indicating plugin migration.

        Parameters
        ----------
        ctx
            Execution context (unused).

        Returns
        -------
        TargetResult
            Skipped result indicating migration.
        """
        _deprecation_warning(self.__class__.__name__)
        return TargetResult.skipped()


CALL_GRAPH_METADATA = CorePluginMetadata(
    name="graphs.call_graph.stub",
    version="3.0.0",
    description="[DEPRECATED] Use native call_graph.py instead.",
    domain=PluginDomain.ANALYTICS,
    kind="builder",
    stage="graph",
    provides=("graph.call_graph_nodes", "graph.call_graph_edges"),
    requires=("core.goids",),
    produces_tables=("graph.call_graph_nodes", "graph.call_graph_edges"),
    consumes_tables=("core.goids",),
)


class CallGraphPlugin(MetadataPlugin):
    """Stub for CallGraphPlugin.

    .. deprecated::
        Use native Hamilton module instead:
        ``codeintel.build.hamilton.native.graphs.call_graph``
    """

    _core_metadata: ClassVar[CorePluginMetadata] = CALL_GRAPH_METADATA

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Return a skipped result indicating plugin migration.

        Parameters
        ----------
        ctx
            Execution context (unused).

        Returns
        -------
        TargetResult
            Skipped result indicating migration.
        """
        _deprecation_warning(self.__class__.__name__)
        return TargetResult.skipped()


IMPORT_GRAPH_METADATA = CorePluginMetadata(
    name="graphs.import_graph.stub",
    version="3.0.0",
    description="[DEPRECATED] Use native import_graph.py instead.",
    domain=PluginDomain.ANALYTICS,
    kind="builder",
    stage="graph",
    provides=("graph.import_modules", "graph.import_graph_edges"),
    requires=("core.modules",),
    produces_tables=("graph.import_modules", "graph.import_graph_edges"),
    consumes_tables=("core.modules",),
)


class ImportGraphPlugin(MetadataPlugin):
    """Stub for ImportGraphPlugin.

    .. deprecated::
        Use native Hamilton module instead:
        ``codeintel.build.hamilton.native.graphs.import_graph``
    """

    _core_metadata: ClassVar[CorePluginMetadata] = IMPORT_GRAPH_METADATA

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Return a skipped result indicating plugin migration.

        Parameters
        ----------
        ctx
            Execution context (unused).

        Returns
        -------
        TargetResult
            Skipped result indicating migration.
        """
        _deprecation_warning(self.__class__.__name__)
        return TargetResult.skipped()


SYMBOL_USES_METADATA = CorePluginMetadata(
    name="graphs.symbol_uses.stub",
    version="3.0.0",
    description="[DEPRECATED] Use native symbol_uses.py instead.",
    domain=PluginDomain.ANALYTICS,
    kind="builder",
    stage="graph",
    provides=("graph.symbol_use_edges",),
    requires=("core.scip_occurrences", "core.modules", "core.goids"),
    produces_tables=("graph.symbol_use_edges",),
    consumes_tables=("core.scip_occurrences", "core.modules", "core.goids"),
)


class SymbolUsesPlugin(MetadataPlugin):
    """Stub for SymbolUsesPlugin.

    .. deprecated::
        Use native Hamilton module instead:
        ``codeintel.build.hamilton.native.graphs.symbol_uses``
    """

    _core_metadata: ClassVar[CorePluginMetadata] = SYMBOL_USES_METADATA

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Return a skipped result indicating plugin migration.

        Parameters
        ----------
        ctx
            Execution context (unused).

        Returns
        -------
        TargetResult
            Skipped result indicating migration.
        """
        _deprecation_warning(self.__class__.__name__)
        return TargetResult.skipped()


CFG_DFG_METADATA = CorePluginMetadata(
    name="graphs.cfg_dfg.stub",
    version="3.0.0",
    description="[DEPRECATED] Use native cfg_dfg.py instead.",
    domain=PluginDomain.ANALYTICS,
    kind="builder",
    stage="graph",
    provides=("graph.cfg_blocks", "graph.cfg_edges", "graph.dfg_edges"),
    requires=("core.goids",),
    produces_tables=("graph.cfg_blocks", "graph.cfg_edges", "graph.dfg_edges"),
    consumes_tables=("core.goids",),
)


class CfgDfgPlugin(MetadataPlugin):
    """Stub for CfgDfgPlugin.

    .. deprecated::
        Use native Hamilton module instead:
        ``codeintel.build.hamilton.native.graphs.cfg_dfg``
    """

    _core_metadata: ClassVar[CorePluginMetadata] = CFG_DFG_METADATA

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Return a skipped result indicating plugin migration.

        Parameters
        ----------
        ctx
            Execution context (unused).

        Returns
        -------
        TargetResult
            Skipped result indicating migration.
        """
        _deprecation_warning(self.__class__.__name__)
        return TargetResult.skipped()


GRAPH_VALIDATION_METADATA = CorePluginMetadata(
    name="graphs.graph_validation.stub",
    version="3.0.0",
    description="[DEPRECATED] Use native graph_validation.py instead.",
    domain=PluginDomain.ANALYTICS,
    kind="validation",
    stage="graph",
    provides=("analytics.graph_validation",),
    requires=("graph.call_graph_edges", "graph.import_graph_edges"),
    produces_tables=(),
    consumes_tables=(
        "graph.call_graph_edges",
        "graph.call_graph_nodes",
        "graph.import_graph_edges",
        "graph.import_modules",
        "graph.cfg_edges",
        "graph.cfg_blocks",
    ),
)


class GraphValidationPlugin(MetadataPlugin):
    """Stub for GraphValidationPlugin.

    .. deprecated::
        Use native Hamilton module instead:
        ``codeintel.build.hamilton.native.graphs.graph_validation``
    """

    _core_metadata: ClassVar[CorePluginMetadata] = GRAPH_VALIDATION_METADATA

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Return a skipped result indicating plugin migration.

        Parameters
        ----------
        ctx
            Execution context (unused).

        Returns
        -------
        TargetResult
            Skipped result indicating migration.
        """
        _deprecation_warning(self.__class__.__name__)
        return TargetResult.skipped()


CORE_METRICS_METADATA = CorePluginMetadata(
    name="graphs.core_metrics.stub",
    version="3.0.0",
    description="[DEPRECATED] Use native graph_metrics.py instead.",
    domain=PluginDomain.ANALYTICS,
    kind="metric",
    stage="graph",
    provides=(
        "analytics.graph_metrics_functions",
        "analytics.graph_metrics_modules",
        "analytics.graph_metrics_functions_ext",
        "analytics.graph_metrics_modules_ext",
        "analytics.graph_stats",
    ),
    requires=("graph.call_graph_edges",),
    produces_tables=(
        "analytics.graph_metrics_functions",
        "analytics.graph_metrics_modules",
        "analytics.graph_metrics_functions_ext",
        "analytics.graph_metrics_modules_ext",
        "analytics.graph_stats",
    ),
    consumes_tables=("graph.call_graph_edges", "graph.call_graph_nodes"),
)


class CoreMetricsPlugin(MetadataPlugin):
    """Stub for CoreMetricsPlugin.

    .. deprecated::
        Use native Hamilton module instead:
        ``codeintel.build.hamilton.native.graphs.graph_metrics``
    """

    _core_metadata: ClassVar[CorePluginMetadata] = CORE_METRICS_METADATA

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Return a skipped result indicating plugin migration.

        Parameters
        ----------
        ctx
            Execution context (unused).

        Returns
        -------
        TargetResult
            Skipped result indicating migration.
        """
        _deprecation_warning(self.__class__.__name__)
        return TargetResult.skipped()


__all__ = [
    "CallGraphPlugin",
    "CfgDfgPlugin",
    "CoreMetricsPlugin",
    "GoidBuilderPlugin",
    "GraphValidationPlugin",
    "ImportGraphPlugin",
    "SymbolUsesPlugin",
]
