"""Behavioral coverage plugin.

This plugin assigns heuristic behavior tags to tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

from codeintel.analytics.testing.profiles.builder import build_behavioral_coverage
from codeintel.build.context import TargetResult
from codeintel.build.plugin import MetadataPlugin
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain

if TYPE_CHECKING:
    from codeintel.analytics.testing.profiles.types import BehavioralLLMRunner
    from codeintel.build.context import TargetExecutionContext


BEHAVIORAL_COVERAGE_METADATA = CorePluginMetadata(
    name="analytics.behavioral_coverage",
    version="3.0.0",
    description="Assign heuristic behavior tags to tests (unit/integration/etc.).",
    domain=PluginDomain.ANALYTICS,
    kind="metric",
    stage="test",
    provides=("analytics.behavioral_coverage",),
    requires=("analytics.test_profile",),
    produces_tables=("analytics.behavioral_coverage",),
    consumes_tables=("analytics.test_profile",),
)


class BehavioralCoveragePlugin(MetadataPlugin):
    """Assign heuristic behavior tags to tests (unit/integration/etc.).

    Classifies tests into categories:
    - Unit tests vs integration tests
    - Behavioral patterns and coverage types
    - Optional LLM-assisted classification

    Outputs
    -------
    - analytics.behavioral_coverage: Test behavioral classifications
    """

    _core_metadata: ClassVar[CorePluginMetadata] = BEHAVIORAL_COVERAGE_METADATA

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

        llm_runner: BehavioralLLMRunner | None = None
        llm_runner_raw = ctx.parameters.get_optional("behavioral_llm_runner", object)
        if llm_runner_raw is not None and callable(llm_runner_raw):
            llm_runner = cast("BehavioralLLMRunner", llm_runner_raw)

        try:
            build_behavioral_coverage(
                ctx.gateway,
                ctx.snapshot,
                llm_runner=llm_runner,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Behavioral coverage build failed: {e}")

        row = ctx.gateway.execute(
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


__all__ = ["BEHAVIORAL_COVERAGE_METADATA", "BehavioralCoveragePlugin"]
