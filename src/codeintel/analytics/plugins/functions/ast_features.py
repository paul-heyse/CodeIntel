"""Function AST features plugin.

This plugin computes AST-derived semantic features for functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

from codeintel.analytics.adapters.base import DeleteScope
from codeintel.analytics.ast_features.persist import features_to_row
from codeintel.analytics.utilities.datasets import (
    get_function_ast_features_contract,
    insert_analytics_rows,
)
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin

if TYPE_CHECKING:
    from codeintel.analytics.ast_features.model import FunctionAstFeatures
    from codeintel.build.context import TargetExecutionContext


class FunctionAstFeaturesPlugin(TargetPlugin):
    """Compute AST-derived semantic features for each function.

    Extracts structural features from function ASTs including:
    - Control flow patterns
    - Statement types and distribution
    - Expression complexity

    Outputs
    -------
    - analytics.function_ast_features: Semantic features per function
    """

    plugin_name: ClassVar[str] = "functions.ast_features"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Compute AST-derived semantic features for each function."

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

        # Get catalog for resources
        catalog = ctx.resources.catalog
        if catalog is None:
            return TargetResult.failed("CatalogProvider is required")

        # Get features from FeaturesProvider
        features_map = catalog.get_resource("FeaturesProvider")
        if features_map is None:
            return TargetResult.failed("FeaturesProvider is required")

        features_map = cast("dict[int, FunctionAstFeatures]", features_map)

        # Get AST stats for metadata (available for debugging/logging)
        ast_data = catalog.get_resource("AstProvider")
        _ = ast_data  # May be used for logging in future

        try:
            rows = [
                features_to_row(
                    repo=ctx.repo,
                    commit=ctx.commit,
                    features=features,
                )
                for features in features_map.values()
            ]

            contract = get_function_ast_features_contract(ctx.gateway)
            delete_scope = DeleteScope(repo=ctx.repo, commit=ctx.commit)
            insert_analytics_rows(
                ctx.gateway,
                contract,
                rows,
                delete_scope=delete_scope,
                scope=f"{ctx.repo}@{ctx.commit}",
            )
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Function AST features computation failed: {e}")

        return TargetResult.succeeded(
            row_counts={"analytics.function_ast_features": len(rows)},
        )


__all__ = ["FunctionAstFeaturesPlugin"]
