"""Profiles plugin.

This plugin builds aggregated profiles for functions, files, and modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.profiles import (
    build_file_profile,
    build_function_profile,
    build_module_profile,
)
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config.steps_analytics import ProfilesAnalyticsStepConfig

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext


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

    plugin_name: ClassVar[str] = "profiles.build"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = (
        "Build aggregated profiles for functions, files, and modules."
    )

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
        _ = self  # Protocol method requires instance

        # Build config from context
        include_ownership = ctx.parameters.get("include_ownership", bool, default=True)

        cfg = ProfilesAnalyticsStepConfig(
            snapshot=ctx.snapshot,
            paths=ctx.paths,
            include_ownership=include_ownership,
        )

        # Get optional resources
        catalog_provider = ctx.resources.catalog
        module_map = None  # Will be loaded from catalog if available

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


__all__ = ["ProfilesPlugin"]
