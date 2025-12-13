"""Function contracts plugin.

This plugin infers pre/postconditions and nullability contracts.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.functions import compute_function_contracts
from codeintel.analytics.parsing.ast_cache import FunctionAstLoadRequest, load_function_asts
from codeintel.build.context import TargetResult
from codeintel.build.plugin import MetadataPlugin
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext

log = logging.getLogger(__name__)


FUNCTION_CONTRACTS_METADATA = CorePluginMetadata(
    name="analytics.function_contracts",
    version="3.0.0",
    description="Infer pre/postconditions and nullability contracts for functions.",
    domain=PluginDomain.ANALYTICS,
    kind="metric",
    stage="function",
    provides=("analytics.function_contracts",),
    requires=("core.goids",),
    produces_tables=("analytics.function_contracts",),
    consumes_tables=("core.goids",),
)


class FunctionContractsPlugin(MetadataPlugin):
    """Infer pre/postconditions and nullability contracts for functions.

    Analyzes functions to infer:
    - Preconditions (required input states)
    - Postconditions (guaranteed output states)
    - Nullability contracts

    Outputs
    -------
    - analytics.function_contracts: Contract information
    """

    _core_metadata: ClassVar[CorePluginMetadata] = FUNCTION_CONTRACTS_METADATA

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
            function_ast_map, _missing = load_function_asts(
                ctx.gateway,
                FunctionAstLoadRequest(
                    repo=ctx.snapshot.repo,
                    commit=ctx.snapshot.commit,
                    repo_root=ctx.snapshot.repo_root,
                    catalog_provider=catalog,
                ),
            )
        except (RuntimeError, ValueError, OSError) as e:
            log.warning("Failed to load function ASTs: %s", e)
            function_ast_map = {}

        try:
            compute_function_contracts(
                ctx.gateway,
                ctx.snapshot,
                function_ast_map=function_ast_map,
                catalog=catalog,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Function contracts computation failed: {e}")

        return TargetResult.succeeded()


__all__ = ["FUNCTION_CONTRACTS_METADATA", "FunctionContractsPlugin"]
