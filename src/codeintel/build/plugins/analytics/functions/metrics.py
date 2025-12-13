"""Function metrics plugin.

This plugin computes function complexity and type coverage metrics.
It is schema-aware and validates output data against registered Pandera schemas.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.functions import (
    FunctionAnalyticsOptions,
    compute_function_metrics_and_types,
)
from codeintel.build.context import TargetResult
from codeintel.build.plugin import MetadataPlugin
from codeintel.config.datasets.schema_registry import SCHEMA_REGISTRY
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext

log = logging.getLogger(__name__)


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


class FunctionMetricsPlugin(MetadataPlugin):
    """Compute function complexity and type coverage metrics.

    Output Tables
    -------------
    - analytics.function_metrics: Complexity and size metrics
    - analytics.function_types: Type annotation data
    """

    _core_metadata: ClassVar[CorePluginMetadata] = FUNCTION_METRICS_METADATA

    def check_schemas_available(self) -> dict[str, bool]:
        """Check if output table schemas are registered.

        Returns
        -------
        dict[str, bool]
            Mapping of table names to availability status.
        """
        return {
            table: SCHEMA_REGISTRY.get(table) is not None
            for table in self._core_metadata.produces_tables
        }

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

        Notes
        -----
        This plugin validates output data against Pandera schemas when available.
        Schema validation occurs in the persistence layer via `codeintel.config.datasets.validation.validate_df`.
        """
        _ = self

        schema_status = self.check_schemas_available()
        for table, available in schema_status.items():
            if not available:
                log.warning("No schema registered for output table: %s", table)
            else:
                log.debug("Schema available for validation: %s", table)

        function_ast_map = None
        missing_function_goids: set[int] = set()

        opts = self.resolve_options(
            FunctionAnalyticsOptions,
            dynamic_overrides={
                "function_ast_map": function_ast_map,
                "missing_function_goids": missing_function_goids,
            },
        )

        result = compute_function_metrics_and_types(ctx.gateway, ctx.snapshot, options=opts)

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
