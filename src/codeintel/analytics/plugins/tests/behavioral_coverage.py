"""Behavioral coverage plugin.

This plugin assigns heuristic behavior tags to tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

from codeintel.analytics.testing.profiles.builder import build_behavioral_coverage
from codeintel.analytics.testing.profiles.types import BehavioralLLMRunner
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config.steps_analytics import BehavioralCoverageStepConfig

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext


class BehavioralCoveragePlugin(TargetPlugin):
    """Assign heuristic behavior tags to tests (unit/integration/etc.).

    Classifies tests into categories:
    - Unit tests vs integration tests
    - Behavioral patterns and coverage types
    - Optional LLM-assisted classification

    Outputs
    -------
    - analytics.behavioral_coverage: Test behavioral classifications
    """

    plugin_name: ClassVar[str] = "behavioral_coverage"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = (
        "Assign heuristic behavior tags to tests (unit/integration/etc.)."
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

        cfg = BehavioralCoverageStepConfig(
            snapshot=ctx.snapshot,
        )

        # Get optional LLM runner from parameters
        llm_runner: BehavioralLLMRunner | None = None
        llm_runner_raw = ctx.parameters.get_optional("behavioral_llm_runner", object)
        if llm_runner_raw is not None and callable(llm_runner_raw):
            llm_runner = cast("BehavioralLLMRunner", llm_runner_raw)

        try:
            build_behavioral_coverage(
                ctx.gateway,
                cfg,
                llm_runner=llm_runner,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Behavioral coverage build failed: {e}")

        # Count rows written
        row = ctx.gateway.con.execute(
            """
            SELECT COUNT(*) FROM analytics.behavioral_coverage
            WHERE repo = ? AND commit = ?
            """,
            [ctx.repo, ctx.commit],
        ).fetchone()
        row_count = int(row[0]) if row else 0

        return TargetResult.succeeded(
            row_counts={"analytics.behavioral_coverage": row_count},
        )


__all__ = ["BehavioralCoveragePlugin"]
