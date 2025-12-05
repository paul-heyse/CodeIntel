"""Entrypoints plugin.

This plugin detects HTTP/CLI/job entrypoints and maps them to handlers and tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.entrypoints import build_entrypoints
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config.steps_analytics import EntryPointsStepConfig

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext


class EntrypointsPlugin(TargetPlugin):
    """Detect HTTP/CLI/job entrypoints and map them to handlers and tests.

    Identifies and maps:
    - HTTP endpoints (Flask, FastAPI, etc.)
    - CLI commands
    - Background job handlers
    - Tests covering each entrypoint

    Outputs
    -------
    - analytics.entrypoints: Detected entrypoints
    - analytics.entrypoint_tests: Tests covering entrypoints
    """

    plugin_name: ClassVar[str] = "entrypoints.build"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = (
        "Detect HTTP/CLI/job entrypoints and map them to handlers and tests."
    )

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute the entrypoints detection.

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

        cfg = EntryPointsStepConfig(
            snapshot=ctx.snapshot,
            paths=ctx.paths,
        )

        catalog = ctx.resources.catalog
        if catalog is None:
            return TargetResult.failed("CatalogProvider is required")

        module_map = catalog.get_resource("ModuleMapProvider")
        if module_map is None:
            return TargetResult.failed("ModuleMapProvider is required")

        features_map = catalog.get_resource("FeaturesProvider")
        if features_map is None:
            return TargetResult.failed("FeaturesProvider is required")

        try:
            build_entrypoints(
                ctx.gateway,
                cfg,
                catalog_provider=catalog,
                module_map=module_map,
                features_map=features_map,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Entrypoints build failed: {e}")

        return TargetResult.succeeded()


__all__ = ["EntrypointsPlugin"]
