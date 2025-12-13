"""Semantic roles plugin.

This plugin computes semantic roles for functions and calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

from codeintel.analytics.semantic_roles import compute_semantic_roles
from codeintel.build.context import TargetResult
from codeintel.build.plugin import MetadataPlugin
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain

if TYPE_CHECKING:
    from codeintel.analytics.ast_features.model import FunctionAstFeatures
    from codeintel.analytics.parsing.ast_cache import FunctionAst
    from codeintel.build.context import TargetExecutionContext


SEMANTIC_ROLES_METADATA = CorePluginMetadata(
    name="analytics.semantic_roles",
    version="3.0.0",
    description="Compute semantic roles for functions and calls.",
    domain=PluginDomain.ANALYTICS,
    kind="metric",
    stage="semantic",
    provides=("analytics.semantic_roles",),
    requires=("core.modules", "analytics.function_ast_features"),
    produces_tables=("analytics.semantic_roles",),
    consumes_tables=("core.modules", "analytics.function_ast_features"),
)


class SemanticRolesPlugin(MetadataPlugin):
    """Compute semantic roles for functions and calls.

    Classifies functions and calls by:
    - Semantic role (producer, consumer, transformer, etc.)
    - Data flow patterns
    - Behavioral classification

    Outputs
    -------
    - analytics.semantic_roles: Semantic role classifications
    """

    _core_metadata: ClassVar[CorePluginMetadata] = SEMANTIC_ROLES_METADATA

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

        catalog = ctx.resources.catalog
        module_by_path: dict[str, str] = {}
        ast_map: dict[int, FunctionAst] = {}
        features_map: dict[int, FunctionAstFeatures] = {}

        if catalog is not None:
            if hasattr(catalog, "catalog"):
                module_by_path = catalog.catalog().module_by_path

            ast_data = None
            if ast_data is not None:
                ast_map = ast_data.function_ast_map

            features = None
            if features is not None:
                features_map = cast("dict[int, FunctionAstFeatures]", features)

        try:
            compute_semantic_roles(
                ctx.gateway,
                ctx.snapshot,
                module_by_path=module_by_path,
                ast_map=ast_map,
                features_map=features_map,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Semantic roles computation failed: {e}")

        return TargetResult.succeeded()


__all__ = ["SEMANTIC_ROLES_METADATA", "SemanticRolesPlugin"]
