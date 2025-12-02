"""Behavioral coverage plugin using the new protocol.

This module provides the behavioral coverage plugin migrated to the
new unified plugin protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

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
from codeintel.analytics.tests.profiles import build_behavioral_coverage
from codeintel.analytics.tests_profiles.types import BehavioralLLMRunner
from codeintel.config.steps_analytics import BehavioralCoverageStepConfig


@dataclass
class BehavioralCoveragePlugin:
    """Plugin for assigning behavioral tags to tests.

    Classifies tests into categories:
    - Unit tests vs integration tests
    - Behavioral patterns and coverage types
    - Optional LLM-assisted classification
    """

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="tests.behavioral_coverage",
            description="Assign heuristic behavior tags to tests (unit/integration/etc.).",
            stage="test",
            version="2.0.0",
            enabled_by_default=True,
            severity="fatal",
            inputs=(
                PluginInputSpec(
                    name="behavioral_cfg",
                    type_ref="BehavioralCoverageStepConfig",
                    required=True,
                    source="config",
                ),
            ),
            outputs=(
                PluginOutputSpec(
                    name="behavioral_coverage",
                    tables=("analytics.behavioral_coverage",),
                ),
            ),
            capabilities_provided=(
                PluginCapability(name="analytics.behavioral_coverage", kind="dataset"),
            ),
            capabilities_required=(
                PluginCapability(name="analytics.test_profile", kind="dataset"),
            ),
            resource_hints=PluginResourceHints(
                max_runtime_ms=120_000,
                requires_gpu=False,
                priority=30,
            ),
            tags=("tests", "behavioral", "classification"),
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

        if not ctx.has_config(BehavioralCoverageStepConfig):
            errors.append("BehavioralCoverageStepConfig is required")

        # Validate llm runner if provided
        llm_runner_raw = ctx.extra.get("behavioral_llm_runner")
        if llm_runner_raw is not None and not callable(llm_runner_raw):
            errors.append("behavioral_llm_runner must be callable or None")

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
            cfg = ctx.get_config(BehavioralCoverageStepConfig)
        except ValueError as e:
            return PluginResult.fail(str(e))

        # Get optional LLM runner
        llm_runner: BehavioralLLMRunner | None = None
        llm_runner_raw = ctx.extra.get("behavioral_llm_runner")
        if llm_runner_raw is not None:
            if not callable(llm_runner_raw):
                return PluginResult.fail("behavioral_llm_runner must be callable")
            llm_runner = cast("BehavioralLLMRunner", llm_runner_raw)

        try:
            build_behavioral_coverage(
                ctx.gateway,
                cfg,
                llm_runner=llm_runner,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return PluginResult.fail(f"Behavioral coverage build failed: {e}")

        # Count rows written
        row = ctx.gateway.con.execute(
            """
            SELECT COUNT(*) FROM analytics.behavioral_coverage
            WHERE repo = ? AND commit = ?
            """,
            [cfg.repo, cfg.commit],
        ).fetchone()
        row_count = int(row[0]) if row else 0

        return PluginResult.ok(
            row_counts={"analytics.behavioral_coverage": row_count},
            meta={"behavior_rows": row_count},
        )


__all__ = ["BehavioralCoveragePlugin"]
