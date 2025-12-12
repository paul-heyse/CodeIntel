"""Profiles plugin.

This plugin builds aggregated profiles for functions, files, and modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.plugins._metadata import to_plugin_metadata
from codeintel.analytics.profiles import (
    build_file_profile,
    build_function_profile,
    build_module_profile,
)
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config.steps_analytics import ProfilesAnalyticsStepConfig
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.core.plugins.types.protocol import PluginMetadata


PROFILES_METADATA = CorePluginMetadata(
    name="analytics.profiles",
    version="3.0.0",
    description="Build aggregated profiles for functions, files, and modules.",
    domain=PluginDomain.ANALYTICS,
    kind="metric",
    stage="profiles",
    provides=(
        "analytics.function_profile",
        "analytics.file_profile",
        "analytics.module_profile",
    ),
    requires=("graph.call_graph_edges", "graph.symbol_use_edges"),
    produces_tables=(
        "analytics.function_profile",
        "analytics.file_profile",
        "analytics.module_profile",
    ),
    consumes_tables=("graph.call_graph_edges", "graph.symbol_use_edges"),
)


class ProfilesPlugin(TargetPlugin):
    """Build aggregated profiles for functions, files, and modules.

    Creates comprehensive profiles for:
    - Functions (combining metrics, effects, contracts, etc.)
    - Files (aggregating function profiles)
    - Modules (aggregating file profiles)

    Outputs
    -------
    - analytics.function_profile: Function profiles
    - analytics.file_profile: File profiles
    - analytics.module_profile: Module profiles
    """

    plugin_name: ClassVar[str] = "profiles"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = (
        "Build aggregated profiles for functions, files, and modules."
    )
    _core_metadata: ClassVar[CorePluginMetadata] = PROFILES_METADATA

    @property
    def metadata(self) -> PluginMetadata:
        """Return protocol-compatible metadata."""
        return to_plugin_metadata(self._core_metadata)

    @property
    def core_metadata(self) -> CorePluginMetadata:
        """Return canonical metadata."""
        return self._core_metadata

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute the plugin.

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

        cfg = ProfilesAnalyticsStepConfig(
            snapshot=ctx.snapshot,
        )

        catalog_provider = ctx.resources.catalog
        module_map = None

        try:
            build_function_profile(
                ctx.gateway,
                cfg,
                catalog_provider=catalog_provider,
                module_map=module_map,
            )
            build_file_profile(
                ctx.gateway,
                cfg,
                catalog_provider=catalog_provider,
            )
            build_module_profile(
                ctx.gateway,
                cfg,
                catalog_provider=catalog_provider,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Profiles build failed: {e}")

        return TargetResult.succeeded()


__all__ = ["PROFILES_METADATA", "ProfilesPlugin"]
