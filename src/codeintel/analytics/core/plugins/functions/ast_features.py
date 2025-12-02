"""Function AST features plugin using the new protocol.

This module provides the function AST features plugin migrated to the
new unified plugin protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from codeintel.analytics.resources.analytics_context import AnalyticsContextProvider

from codeintel.analytics.ast_features.persist import features_to_row
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
from codeintel.analytics.datasets import (
    DeleteScope,
    get_function_ast_features_contract,
    insert_analytics_rows,
)
from codeintel.config.steps_analytics import FunctionAnalyticsStepConfig


@dataclass
class FunctionAstFeaturesPlugin:
    """Plugin for computing AST-derived semantic features for functions.

    Extracts structural features from function ASTs including:
    - Control flow patterns
    - Statement types and distribution
    - Expression complexity
    """

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="functions.ast_features",
            description="Compute AST-derived semantic features for each function.",
            stage="function",
            version="2.0.0",
            enabled_by_default=True,
            severity="fatal",
            inputs=(
                PluginInputSpec(
                    name="function_cfg",
                    type_ref="FunctionAnalyticsStepConfig",
                    required=True,
                    source="config",
                ),
            ),
            outputs=(
                PluginOutputSpec(
                    name="function_ast_features",
                    tables=("analytics.function_ast_features",),
                ),
            ),
            capabilities_provided=(
                PluginCapability(name="analytics.function_ast_features", kind="dataset"),
            ),
            capabilities_required=(PluginCapability(name="core.goids", kind="dataset"),),
            resource_hints=PluginResourceHints(
                max_runtime_ms=120_000,
                requires_gpu=False,
                priority=12,
            ),
            tags=("functions", "ast", "features"),
        )

    def validate_inputs(self, ctx: PluginExecutionContext) -> ValidationResult:
        """Validate that required inputs are available.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        ValidationResult
            Validation result.
        """
        _ = self.metadata  # Access self for protocol compliance
        errors: list[str] = []

        if not ctx.has_config(FunctionAnalyticsStepConfig):
            errors.append("FunctionAnalyticsStepConfig is required")

        if errors:
            return ValidationResult.failure(tuple(errors))
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
        _ = self.metadata  # Access self for protocol compliance
        try:
            cfg = ctx.get_config(FunctionAnalyticsStepConfig)
        except ValueError as e:
            return PluginResult.fail(str(e))

        # Get required analytics context
        if not ctx.has_resource_by_name("AnalyticsContextProvider"):
            return PluginResult.fail("AnalyticsContextProvider is required")
        analytics_provider = cast(
            "AnalyticsContextProvider", ctx.require_by_name("AnalyticsContextProvider")
        )
        analytics_ctx = analytics_provider.get()

        try:
            rows = [
                features_to_row(
                    repo=cfg.repo,
                    commit=cfg.commit,
                    features=features,
                )
                for features in analytics_ctx.function_features_map.values()
            ]

            contract = get_function_ast_features_contract(ctx.gateway)
            delete_scope = DeleteScope(params=[cfg.repo, cfg.commit])
            insert_analytics_rows(
                ctx.gateway,
                contract,
                rows,
                delete_scope=delete_scope,
                scope=f"{cfg.repo}@{cfg.commit}",
            )
        except (RuntimeError, ValueError, OSError) as e:
            return PluginResult.fail(f"Function AST features computation failed: {e}")

        return PluginResult.ok(
            row_counts={"analytics.function_ast_features": len(rows)},
            meta={
                "functions_seen": len(analytics_ctx.function_ast_map),
                "functions_missing": len(analytics_ctx.missing_function_goids),
            },
        )


__all__ = ["FunctionAstFeaturesPlugin"]
