"""External dependencies plugin using the new protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from codeintel.analytics.resources.asts import AstProvider
    from codeintel.analytics.resources.catalog import CatalogProvider
    from codeintel.analytics.resources.features import FeaturesProvider
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
from codeintel.analytics.dependencies import (
    build_external_dependencies,
    build_external_dependency_calls,
)
from codeintel.analytics.dependencies.core import ExternalDependencyInputs
from codeintel.config.steps_graphs import ExternalDependenciesStepConfig


@dataclass
class ExternalDepsPlugin:
    """Plugin for identifying external dependency usage.

    Analyzes and builds:
    - External dependency calls per function
    - Aggregated dependency usage patterns
    - Third-party library integration points
    """

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="deps.external",
            description="Identify external dependency usage across functions.",
            stage="other",
            version="3.0.0",
            enabled_by_default=True,
            severity="fatal",
            inputs=(
                PluginInputSpec(
                    name="external_deps_cfg",
                    type_ref="ExternalDependenciesStepConfig",
                    required=True,
                    source="config",
                ),
            ),
            outputs=(
                PluginOutputSpec(
                    name="external_dependency_calls",
                    tables=("analytics.external_dependency_calls",),
                ),
                PluginOutputSpec(
                    name="external_dependencies",
                    tables=("analytics.external_dependencies",),
                ),
            ),
            capabilities_provided=(
                PluginCapability(name="analytics.external_dependency_calls", kind="dataset"),
                PluginCapability(name="analytics.external_dependencies", kind="dataset"),
            ),
            capabilities_required=(PluginCapability(name="core.goids", kind="dataset"),),
            depends_on=("goids", "config_ingest"),
            resource_hints=PluginResourceHints(
                max_runtime_ms=90_000,
                priority=50,
            ),
            tags=("dependencies", "external", "third-party"),
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
        if not ctx.has_config(ExternalDependenciesStepConfig):
            return ValidationResult.failure(("ExternalDependenciesStepConfig is required",))
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
            cfg = ctx.get_config(ExternalDependenciesStepConfig)
        except ValueError as e:
            return PluginResult.fail(str(e))

        # Get catalog from CatalogProvider
        catalog_provider = None
        if ctx.has_resource_by_name("CatalogProvider"):
            cat_prov = cast("CatalogProvider", ctx.require_by_name("CatalogProvider"))
            catalog_provider = cat_prov.get()

        # Get module map from ModuleMapProvider
        module_map: dict[str, str] = {}
        if ctx.has_resource_by_name("ModuleMapProvider"):
            mm_prov = cast("ModuleMapProvider", ctx.require_by_name("ModuleMapProvider"))
            module_map = mm_prov.get().module_by_path

        # Get AST data from AstProvider
        ast_by_goid: dict[int, object] = {}
        missing_goids: set[int] = set()
        if ctx.has_resource_by_name("AstProvider"):
            ast_prov = cast("AstProvider", ctx.require_by_name("AstProvider"))
            ast_data = ast_prov.get()
            ast_by_goid = ast_data.ast_by_goid
            missing_goids = ast_data.missing_goids

        # Get features from FeaturesProvider
        features_map: dict[int, object] = {}
        if ctx.has_resource_by_name("FeaturesProvider"):
            feat_prov = cast("FeaturesProvider", ctx.require_by_name("FeaturesProvider"))
            features_map = feat_prov.get().features_by_goid

        try:
            inputs = ExternalDependencyInputs(
                catalog_provider=catalog_provider,
                module_map=module_map,
                ast_by_goid=ast_by_goid,  # type: ignore[arg-type]
                features_map=features_map,  # type: ignore[arg-type]
                missing_goids=missing_goids,
            )
            build_external_dependency_calls(ctx.gateway, cfg, inputs=inputs)
            build_external_dependencies(ctx.gateway, cfg)
        except (RuntimeError, ValueError, OSError) as e:
            return PluginResult.fail(f"External dependencies build failed: {e}")

        return PluginResult.ok()


__all__ = ["ExternalDepsPlugin"]
