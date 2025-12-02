"""Profiles plugin using the new protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from codeintel.analytics.resources.catalog import CatalogProvider
    from codeintel.analytics.resources.module_map import ModuleMapProvider

from codeintel.analytics.core.execution_context import PluginExecutionContext
from codeintel.analytics.core.plugin_protocol import (
    PluginCapability,
    PluginInputSpec,
    PluginMetadata,
    PluginOutputSpec,
    PluginResourceHints,
    PluginResult,
    ValidationResult,
)
from codeintel.analytics.profiles import (
    build_file_profile,
    build_function_profile,
    build_module_profile,
)
from codeintel.config.steps_analytics import ProfilesAnalyticsStepConfig


@dataclass
class ProfilesPlugin:
    """Plugin for building aggregated profiles.

    Creates comprehensive profiles for:
    - Functions (combining metrics, effects, contracts, etc.)
    - Files (aggregating function profiles)
    - Modules (aggregating file profiles)
    """

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="profiles.build",
            description="Build aggregated profiles for functions, files, and modules.",
            stage="profiles",
            version="3.0.0",
            enabled_by_default=True,
            severity="fatal",
            inputs=(
                PluginInputSpec(
                    name="profiles_cfg",
                    type_ref="ProfilesAnalyticsStepConfig",
                    required=True,
                    source="config",
                ),
            ),
            outputs=(
                PluginOutputSpec(name="function_profile", tables=("analytics.function_profile",)),
                PluginOutputSpec(name="file_profile", tables=("analytics.file_profile",)),
                PluginOutputSpec(name="module_profile", tables=("analytics.module_profile",)),
            ),
            capabilities_provided=(
                PluginCapability(name="analytics.function_profile", kind="dataset"),
                PluginCapability(name="analytics.file_profile", kind="dataset"),
                PluginCapability(name="analytics.module_profile", kind="dataset"),
            ),
            capabilities_required=(
                PluginCapability(name="analytics.goid_risk_factors", kind="dataset"),
            ),
            depends_on=(
                "risk_factors.build",
                "callgraph",
                "import_graph",
                "functions.effects",
                "functions.contracts",
                "semantic.roles",
                "functions.history",
            ),
            resource_hints=PluginResourceHints(
                max_runtime_ms=120_000,
                priority=70,
            ),
            tags=("profiles", "aggregation"),
        )

    def validate_inputs(self, ctx: PluginExecutionContext) -> ValidationResult:
        """Validate required inputs.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        ValidationResult
            Validation result.
        """
        _ = self.metadata
        if not ctx.has_config(ProfilesAnalyticsStepConfig):
            return ValidationResult.failure(("ProfilesAnalyticsStepConfig is required",))
        return ValidationResult.success()

    def execute(self, ctx: PluginExecutionContext) -> PluginResult:
        """Execute the plugin.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        PluginResult
            Execution result.
        """
        _ = self.metadata
        try:
            cfg = ctx.get_config(ProfilesAnalyticsStepConfig)
        except ValueError as e:
            return PluginResult.fail(str(e))

        # Get catalog from CatalogProvider (optional)
        catalog_provider = None
        if ctx.has_resource_by_name("CatalogProvider"):
            cat_prov = cast("CatalogProvider", ctx.require_by_name("CatalogProvider"))
            catalog_provider = cat_prov.get()

        # Get module map from ModuleMapProvider (optional)
        module_map = None
        if ctx.has_resource_by_name("ModuleMapProvider"):
            mm_prov = cast("ModuleMapProvider", ctx.require_by_name("ModuleMapProvider"))
            module_map = mm_prov.get()

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
            return PluginResult.fail(f"Profiles build failed: {e}")

        return PluginResult.ok()


__all__ = ["ProfilesPlugin"]
