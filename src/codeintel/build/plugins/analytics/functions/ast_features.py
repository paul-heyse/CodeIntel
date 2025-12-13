"""Function AST features plugin.

This plugin computes AST-derived semantic features for functions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.ast_features.persist import features_to_row
from codeintel.analytics.resources.features import FeaturesProvider
from codeintel.analytics.utilities.datasets import (
    get_function_ast_features_contract,
    insert_analytics_rows,
)
from codeintel.analytics.utilities.persistence import DeleteScope
from codeintel.build.context import TargetResult
from codeintel.build.plugin import MetadataPlugin
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext

log = logging.getLogger(__name__)


FUNCTION_AST_FEATURES_METADATA = CorePluginMetadata(
    name="analytics.function_ast_features",
    version="3.0.0",
    description="Compute AST-derived semantic features for each function.",
    domain=PluginDomain.ANALYTICS,
    kind="metric",
    stage="function",
    provides=("analytics.function_ast_features",),
    requires=("core.goids", "core.modules"),
    produces_tables=("analytics.function_ast_features",),
    consumes_tables=("core.goids", "core.modules"),
)


class FunctionAstFeaturesPlugin(MetadataPlugin):
    """Compute AST-derived semantic features for each function.

    Extracts structural features from function ASTs including:
    - Control flow patterns
    - Statement types and distribution
    - Expression complexity

    Outputs
    -------
    - analytics.function_ast_features: Semantic features per function
    """

    _core_metadata: ClassVar[CorePluginMetadata] = FUNCTION_AST_FEATURES_METADATA

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
        if catalog is None:
            return TargetResult.failed("CatalogProvider is required")

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


__all__ = ["FUNCTION_AST_FEATURES_METADATA", "FunctionAstFeaturesPlugin"]
