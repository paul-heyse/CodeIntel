"""Entrypoints plugin.

This plugin detects HTTP/CLI/job entrypoints and maps them to handlers and tests.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.entrypoints import build_entrypoints
from codeintel.analytics.resources.features import FeaturesProvider
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.build.plugins.analytics._metadata import to_plugin_metadata
from codeintel.config.steps_analytics import EntryPointsStepConfig
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.core.plugins.types.protocol import PluginMetadata

log = logging.getLogger(__name__)


ENTRYPOINTS_METADATA = CorePluginMetadata(
    name="analytics.entrypoints",
    version="3.0.0",
    description="Detect HTTP/CLI/job entrypoints and map them to handlers and tests.",
    domain=PluginDomain.ANALYTICS,
    kind="metric",
    stage="entrypoints",
    provides=("analytics.entrypoints", "analytics.entrypoint_tests"),
    requires=("core.modules", "analytics.function_ast_features"),
    produces_tables=("analytics.entrypoints", "analytics.entrypoint_tests"),
    consumes_tables=("core.modules", "analytics.function_ast_features"),
)


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

    plugin_name: ClassVar[str] = "entrypoints"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = (
        "Detect HTTP/CLI/job entrypoints and map them to handlers and tests."
    )
    _core_metadata: ClassVar[CorePluginMetadata] = ENTRYPOINTS_METADATA

    @property
    def metadata(self) -> PluginMetadata:
        """Return protocol-compatible metadata."""
        return to_plugin_metadata(self._core_metadata)

    @property
    def core_metadata(self) -> CorePluginMetadata:
        """Return canonical metadata."""
        return self._core_metadata

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
        _ = self

        cfg = EntryPointsStepConfig(
            snapshot=ctx.snapshot,
        )

        catalog = ctx.resources.catalog
        if catalog is None:
            return TargetResult.failed("CatalogProvider is required")

        rows = ctx.gateway.execute(
            """
            SELECT path, module
            FROM core.modules
            WHERE repo = ? AND commit = ?
            """,
            [ctx.snapshot.repo, ctx.snapshot.commit],
        ).fetchall()
        module_map = {str(row[0]): str(row[1]) for row in rows}

        try:
            provider = FeaturesProvider(
                gateway=ctx.gateway,
                snapshot=ctx.snapshot,
                catalog_provider=catalog,
            )
            features_map = provider.get()
        except (RuntimeError, ValueError, OSError) as e:
            log.warning("Failed to compute function features: %s", e)
            features_map = {}

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


__all__ = ["ENTRYPOINTS_METADATA", "EntrypointsPlugin"]
