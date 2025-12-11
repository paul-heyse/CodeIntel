"""Function metrics plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

from codeintel.analytics.functions import (
    FunctionAnalyticsOptions,
    compute_function_metrics_and_types,
)
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config.steps_analytics import FunctionAnalyticsStepConfig
from codeintel.core.plugins.execution.options import PluginOptionsResolver
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.core.plugins.types.protocol import PluginKind, PluginMetadata, PluginStage

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.build.context import TargetExecutionContext


FUNCTION_METRICS_METADATA = CorePluginMetadata(
    name="analytics.function_metrics",
    version="3.0.0",
    description="Compute function complexity and type coverage metrics.",
    domain=PluginDomain.ANALYTICS,
    kind="metric",
    stage="function",
    provides=(
        "analytics.function_metrics",
        "analytics.function_types",
    ),
    requires=("core.goids",),
    produces_tables=(
        "analytics.function_metrics",
        "analytics.function_types",
    ),
    consumes_tables=("core.goids",),
    supports_incremental=False,
    scope_aware=False,
    options_model=FunctionAnalyticsOptions,
    resource_hints={"max_memory_mb": 512},
)


def _to_plugin_metadata(core: CorePluginMetadata) -> PluginMetadata:
    """Convert CorePluginMetadata to PluginMetadata for protocol compliance.

    Returns
    -------
    PluginMetadata
        Protocol-compatible metadata instance.
    """
    return PluginMetadata(
        name=core.name,
        version=core.version,
        description=core.description,
        kind=cast("PluginKind", core.kind),
        stage=cast("PluginStage", core.stage or "function"),
        provides=core.provides,
        requires=core.requires,
        produces_tables=core.produces_tables,
    )


class FunctionMetricsPlugin(TargetPlugin):
    """Compute function complexity and type coverage metrics.

    Output Tables
    -------------
    - analytics.function_metrics: Complexity and size metrics
    - analytics.function_types: Type annotation data
    """

    plugin_name: ClassVar[str] = "function_metrics"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Compute function complexity and type coverage metrics."
    _core_metadata: ClassVar[CorePluginMetadata] = FUNCTION_METRICS_METADATA

    def __init__(self, *, options_resolver: PluginOptionsResolver | None = None) -> None:
        self._options_resolver = options_resolver

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata.

        Returns
        -------
        PluginMetadata
            Protocol-compatible metadata.
        """
        return _to_plugin_metadata(self._core_metadata)

    @property
    def core_metadata(self) -> CorePluginMetadata:
        """Return full core metadata.

        Returns
        -------
        CorePluginMetadata
            Canonical metadata definition.
        """
        return self._core_metadata

    def resolve_options(
        self,
        *,
        dynamic_overrides: Mapping[str, Any] | None = None,
    ) -> FunctionAnalyticsOptions:
        """Resolve typed options from configuration.

        Returns
        -------
        FunctionAnalyticsOptions
            Resolved options instance.
        """
        if self._options_resolver is None:
            if dynamic_overrides:
                return FunctionAnalyticsOptions(**dynamic_overrides)
            return FunctionAnalyticsOptions()

        return self._options_resolver.get_options(
            self._core_metadata,
            FunctionAnalyticsOptions,
            dynamic_overrides=dynamic_overrides,
        )

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute function metrics computation.

        Parameters
        ----------
        ctx
            Execution context providing gateway and parameters.

        Returns
        -------
        TargetResult
            Success result with row counts.
        """
        _ = self  # Protocol method requires instance

        # Get AST data from catalog if available
        function_ast_map = None
        missing_function_goids: set[int] = set()

        # Note: Direct resource access not yet implemented in build context
        # The FunctionCatalogProvider doesn't have get_resource()
        # This will be populated when the build executor provides resources

        opts = self.resolve_options(
            dynamic_overrides={
                "function_ast_map": function_ast_map,
                "missing_function_goids": missing_function_goids,
            }
        )

        # Build config from parameters
        cfg = FunctionAnalyticsStepConfig(
            snapshot=ctx.snapshot,
        )

        result = compute_function_metrics_and_types(ctx.gateway, cfg, options=opts)

        return TargetResult.succeeded(
            row_counts={
                "analytics.function_metrics": result.get("metrics_rows", 0),
                "analytics.function_types": result.get("types_rows", 0),
            }
        )


__all__ = [
    "FUNCTION_METRICS_METADATA",
    "FunctionMetricsPlugin",
]
