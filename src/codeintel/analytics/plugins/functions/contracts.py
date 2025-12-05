"""Function contracts plugin.

This plugin infers pre/postconditions and nullability contracts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.functions import compute_function_contracts
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config.steps_analytics import FunctionContractsStepConfig

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext


class FunctionContractsPlugin(TargetPlugin):
    """Infer pre/postconditions and nullability contracts for functions.

    Analyzes functions to infer:
    - Preconditions (required input states)
    - Postconditions (guaranteed output states)
    - Nullability contracts

    Outputs
    -------
    - analytics.function_contracts: Contract information
    """

    plugin_name: ClassVar[str] = "functions.contracts"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = (
        "Infer pre/postconditions and nullability contracts for functions."
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
        cfg = FunctionContractsStepConfig(
            snapshot=ctx.snapshot,
            paths=ctx.paths,
        )

        # Get catalog provider
        catalog = ctx.resources.catalog
        if catalog is None:
            return TargetResult.failed("CatalogProvider is required")

        # Get AST data
        ast_data = catalog.get_resource("AstProvider")
        if ast_data is None:
            return TargetResult.failed("AstProvider is required")

        function_ast_map = ast_data.function_ast_map

        try:
            compute_function_contracts(
                ctx.gateway,
                cfg,
                function_ast_map=function_ast_map,
                catalog=catalog,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Function contracts computation failed: {e}")

        return TargetResult.succeeded()


__all__ = ["FunctionContractsPlugin"]
