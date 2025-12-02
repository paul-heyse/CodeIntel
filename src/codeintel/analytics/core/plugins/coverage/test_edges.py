"""Coverage test edges plugin using the new protocol.

This module provides the coverage test edges plugin migrated to the
new unified plugin protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from codeintel.analytics.resources.catalog import CatalogProvider

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
from codeintel.analytics.tests import compute_test_coverage_edges
from codeintel.config.steps_analytics import TestCoverageStepConfig


@dataclass
class CoverageTestEdgesPlugin:
    """Plugin for building test-to-function coverage edges.

    Analyzes coverage context data to build:
    - Test to function mapping
    - Coverage relationship edges
    - Test impact analysis foundation
    """

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="coverage.test_edges",
            description="Build test-to-function coverage edges from coverage contexts.",
            stage="coverage",
            version="2.0.0",
            enabled_by_default=True,
            severity="fatal",
            inputs=(
                PluginInputSpec(
                    name="test_coverage_cfg",
                    type_ref="TestCoverageStepConfig",
                    required=True,
                    source="config",
                ),
            ),
            outputs=(
                PluginOutputSpec(
                    name="test_edges",
                    tables=("coverage.test_edges",),
                ),
            ),
            capabilities_provided=(PluginCapability(name="coverage.test_edges", kind="dataset"),),
            capabilities_required=(PluginCapability(name="coverage.lines", kind="dataset"),),
            depends_on=("coverage_ingest", "tests_ingest", "goids"),
            resource_hints=PluginResourceHints(
                max_runtime_ms=60_000,
                priority=40,
            ),
            tags=("coverage", "tests", "edges"),
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

        if not ctx.has_config(TestCoverageStepConfig):
            errors.append("TestCoverageStepConfig is required")

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
            cfg = ctx.get_config(TestCoverageStepConfig)
        except ValueError as e:
            return PluginResult.fail(str(e))

        catalog_provider = None
        if ctx.has_resource_by_name("CatalogProvider"):
            cat_resource = cast("CatalogProvider", ctx.require_by_name("CatalogProvider"))
            catalog_provider = cat_resource.get()

        try:
            compute_test_coverage_edges(
                ctx.gateway,
                cfg,
                catalog_provider=catalog_provider,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return PluginResult.fail(f"Coverage test edges computation failed: {e}")

        return PluginResult.ok()


__all__ = ["CoverageTestEdgesPlugin"]
