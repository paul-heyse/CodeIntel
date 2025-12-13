"""Coverage functions plugin.

This plugin aggregates line coverage to function-level metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.compute.coverage.functions import compute_coverage_functions
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.build.plugins.analytics._metadata import to_plugin_metadata
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.core.plugins.types.protocol import PluginMetadata


FUNCTION_COVERAGE_METADATA = CorePluginMetadata(
    name="analytics.coverage_functions",
    version="3.0.0",
    description="Aggregate line coverage to function-level metrics.",
    domain=PluginDomain.ANALYTICS,
    kind="metric",
    stage="coverage",
    provides=("analytics.coverage_functions",),
    requires=("core.goids", "analytics.coverage_lines"),
    produces_tables=("analytics.coverage_functions",),
    consumes_tables=("core.goids", "analytics.coverage_lines"),
)


class CoverageFunctionsPlugin(TargetPlugin):
    """Aggregate line coverage to function-level metrics.

    Analyzes code coverage data to compute:
    - Function-level coverage percentages
    - Covered/uncovered line counts per function
    - Coverage quality metrics

    Outputs
    -------
    - analytics.coverage_functions: Function-level coverage metrics
    """

    plugin_name: ClassVar[str] = "coverage_functions"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Aggregate line coverage to function-level metrics."
    _core_metadata: ClassVar[CorePluginMetadata] = FUNCTION_COVERAGE_METADATA

    @property
    def metadata(self) -> PluginMetadata:
        """Return protocol-compatible metadata."""
        return to_plugin_metadata(self._core_metadata)

    @property
    def core_metadata(self) -> CorePluginMetadata:
        """Return canonical metadata."""
        return self._core_metadata

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute the coverage functions computation.

        Parameters
        ----------
        ctx
            Execution context with gateway and parameters.

        Returns
        -------
        TargetResult
            Success result with row counts.
        """
        _ = self

        try:
            compute_coverage_functions(ctx.gateway, ctx.snapshot)
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Coverage functions computation failed: {e}")

        return TargetResult.succeeded()


__all__ = ["FUNCTION_COVERAGE_METADATA", "CoverageFunctionsPlugin"]
