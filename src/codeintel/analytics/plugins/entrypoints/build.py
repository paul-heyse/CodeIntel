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
from codeintel.config.steps_analytics import EntryPointsStepConfig

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext

log = logging.getLogger(__name__)


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
        )

        catalog = ctx.resources.catalog
        if catalog is None:
            return TargetResult.failed("CatalogProvider is required")

        # Load module map from database
        rows = ctx.gateway.con.execute(
            """
            SELECT path, module
            FROM core.modules
            WHERE repo = ? AND commit = ?
            """,
            [ctx.snapshot.repo, ctx.snapshot.commit],
        ).fetchall()
        module_map = {str(row[0]): str(row[1]) for row in rows}

        # Load features from FeaturesProvider
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


__all__ = ["EntrypointsPlugin"]
