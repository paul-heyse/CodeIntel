"""Function AST features plugin.

This plugin computes AST-derived semantic features for functions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.adapters.base import DeleteScope
from codeintel.analytics.ast_features.persist import features_to_row
from codeintel.analytics.resources.features import FeaturesProvider
from codeintel.analytics.utilities.datasets import (
    get_function_ast_features_contract,
    insert_analytics_rows,
)
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext

log = logging.getLogger(__name__)


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

    plugin_name: ClassVar[str] = "function_ast_features"
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

        # Create FeaturesProvider and compute features
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

        if not features_map:
            log.info("No function features computed for %s@%s", ctx.repo, ctx.commit)
            return TargetResult.succeeded(
                row_counts={"analytics.function_ast_features": 0},
            )

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
