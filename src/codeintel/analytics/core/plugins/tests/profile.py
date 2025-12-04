"""Test profile plugin using the new protocol.

This module provides the test profile plugin migrated to the
new unified plugin protocol.
"""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.analytics.core.execution_context import PluginExecutionContext
from codeintel.analytics.core.plugin_protocol import (
    PluginInputSpec,
    PluginMetadata,
    PluginOutputSpec,
    PluginResourceHints,
    PluginResult,
    ValidationResult,
)
from codeintel.analytics.testing.profiles.builder import build_test_profile
from codeintel.config.steps_analytics import TestProfileStepConfig


@dataclass
class TestProfilePlugin:
    """Plugin for building per-test profiles.

    Creates comprehensive test profiles including:
    - Coverage context for each test
    - Subsystem associations
    - Test metadata aggregation
    """

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="tests.profile",
            description="Build per-test profiles with coverage and subsystem context.",
            kind="analytics",
            stage="test",
            version="2.0.0",
            enabled_by_default=True,
            severity="fatal",
            inputs=(
                PluginInputSpec(
                    name="test_profile_cfg",
                    type_ref="TestProfileStepConfig",
                    required=True,
                    source="config",
                ),
            ),
            outputs=(
                PluginOutputSpec(
                    name="test_profile",
                    tables=("analytics.test_profile",),
                ),
            ),
            provides=(
                "analytics.test_profile",
            ),
            requires=(
                "core.goids",
                "coverage.test_edges",
            ),
            resource_hints=PluginResourceHints(
                max_runtime_ms=60_000,
                requires_gpu=False,
                priority=20,
            ),
            tags=("tests", "profile"),
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

        if not ctx.has_config(TestProfileStepConfig):
            errors.append("TestProfileStepConfig is required")

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
            cfg = ctx.get_config(TestProfileStepConfig)
        except ValueError as e:
            return PluginResult.fail(str(e))

        try:
            build_test_profile(ctx.gateway, cfg)
        except (RuntimeError, ValueError, OSError) as e:
            return PluginResult.fail(f"Test profile build failed: {e}")

        # Count rows written
        row = ctx.gateway.con.execute(
            """
            SELECT COUNT(*) FROM analytics.test_profile
            WHERE repo = ? AND commit = ?
            """,
            [cfg.repo, cfg.commit],
        ).fetchone()
        row_count = int(row[0]) if row else 0

        return PluginResult.ok(
            row_counts={"analytics.test_profile": row_count},
            meta={"profile_rows": row_count},
        )


__all__ = ["TestProfilePlugin"]
