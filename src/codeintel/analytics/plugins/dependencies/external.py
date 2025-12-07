"""External dependencies plugin.

This plugin identifies external dependency usage across functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.dependencies import (
    build_external_dependencies,
    build_external_dependency_calls,
)
from codeintel.analytics.dependencies.core import ExternalDependencyInputs
from codeintel.analytics.parsing.ast_cache import FunctionAstLoadRequest, load_function_asts
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config.steps_graphs import ExternalDependenciesStepConfig

if TYPE_CHECKING:
    from codeintel.analytics.ast_features.model import FunctionAstFeatures
    from codeintel.build.context import TargetExecutionContext


class ExternalDepsPlugin(TargetPlugin):
    """Identify external dependency usage across functions.

    Analyzes and builds:
    - External dependency calls per function
    - Aggregated dependency usage patterns
    - Third-party library integration points

    Outputs
    -------
    - analytics.external_dependency_calls: External dependency calls
    - analytics.external_dependencies: Aggregated dependencies
    """

    plugin_name: ClassVar[str] = "external_deps"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Identify external dependency usage across functions."

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

        cfg = ExternalDependenciesStepConfig(
            snapshot=ctx.snapshot,
        )

        catalog = ctx.resources.catalog
        if catalog is None:
            return TargetResult.failed("CatalogProvider is required")

        # Get resources from catalog
        module_map: dict[str, str] = dict(catalog.catalog().module_by_path)
        missing_goids: set[int] = set()
        features_map: dict[int, FunctionAstFeatures] = {}

        try:
            ast_by_goid, missing_goids = load_function_asts(
                ctx.gateway,
                FunctionAstLoadRequest(
                    repo=ctx.snapshot.repo,
                    commit=ctx.snapshot.commit,
                    repo_root=ctx.snapshot.repo_root,
                    catalog_provider=catalog,
                ),
            )
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"External dependency AST load failed: {e}")

        for rel_path in {func_ast.rel_path for func_ast in ast_by_goid.values()}:
            module = catalog.module_for_path(rel_path)
            if module is None:
                module = rel_path.replace("/", ".").removesuffix(".py")
            module_map[rel_path] = module

        try:
            inputs = ExternalDependencyInputs(
                catalog_provider=catalog,
                module_map=module_map,
                ast_by_goid=ast_by_goid,
                features_map=features_map,
                missing_goids=missing_goids,
            )
            build_external_dependency_calls(ctx.gateway, cfg, inputs=inputs)
            build_external_dependencies(ctx.gateway, cfg)
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"External dependencies build failed: {e}")

        return TargetResult.succeeded()


__all__ = ["ExternalDepsPlugin"]
