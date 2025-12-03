"""Semantic roles plugin using the new protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from codeintel.analytics.ast_features.model import FunctionAstFeatures
    from codeintel.analytics.function_ast_cache import FunctionAst
    from codeintel.analytics.resources.asts import AstResourceData
    from codeintel.graphs.catalog import FunctionCatalogProvider

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
from codeintel.analytics.semantic_roles import compute_semantic_roles
from codeintel.config.steps_analytics import SemanticRolesStepConfig


@dataclass
class SemanticRolesPlugin:
    """Plugin for computing semantic roles.

    Classifies functions and calls by:
    - Semantic role (producer, consumer, transformer, etc.)
    - Data flow patterns
    - Behavioral classification
    """

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="semantic.roles",
            description="Compute semantic roles for functions and calls.",
            stage="semantic",
            version="3.0.0",
            enabled_by_default=True,
            severity="fatal",
            inputs=(
                PluginInputSpec(
                    name="semantic_roles_cfg",
                    type_ref="SemanticRolesStepConfig",
                    required=True,
                    source="config",
                ),
            ),
            outputs=(
                PluginOutputSpec(name="semantic_roles", tables=("analytics.semantic_roles",)),
            ),
            capabilities_provided=(
                PluginCapability(name="analytics.semantic_roles", kind="dataset"),
            ),
            capabilities_required=(PluginCapability(name="core.goids", kind="dataset"),),
            depends_on=("callgraph",),
            resource_hints=PluginResourceHints(
                max_runtime_ms=90_000,
                priority=50,
            ),
            tags=("semantic", "roles", "classification"),
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
        if not ctx.has_config(SemanticRolesStepConfig):
            return ValidationResult.failure(("SemanticRolesStepConfig is required",))
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
            cfg = ctx.get_config(SemanticRolesStepConfig)
        except ValueError as e:
            return PluginResult.fail(str(e))

        # Get module_by_path from CatalogProvider
        # Note: require_by_name already calls .get() internally
        module_by_path: dict[str, str] = {}
        if ctx.has_resource_by_name("CatalogProvider"):
            catalog_provider = cast(
                "FunctionCatalogProvider", ctx.require_by_name("CatalogProvider")
            )
            module_by_path = catalog_provider.catalog().module_by_path

        # Get ast_map from AstProvider
        # Note: require_by_name returns the loaded resource (AstResourceData), not provider
        ast_map: dict[int, FunctionAst] = {}
        if ctx.has_resource_by_name("AstProvider"):
            ast_data = cast("AstResourceData", ctx.require_by_name("AstProvider"))
            ast_map = ast_data.function_ast_map

        # Get features from FeaturesProvider
        # Note: require_by_name already calls .get() internally
        features_map: dict[int, FunctionAstFeatures] = {}
        if ctx.has_resource_by_name("FeaturesProvider"):
            features_map = cast(
                "dict[int, FunctionAstFeatures]", ctx.require_by_name("FeaturesProvider")
            )

        try:
            compute_semantic_roles(
                ctx.gateway,
                cfg,
                module_by_path=module_by_path,
                ast_map=ast_map,
                features_map=features_map,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return PluginResult.fail(f"Semantic roles computation failed: {e}")

        return PluginResult.ok()


__all__ = ["SemanticRolesPlugin"]
