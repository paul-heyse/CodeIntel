"""Behavioral coverage plugin.

This plugin assigns heuristic behavior tags to tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

from codeintel.analytics.plugins._metadata import to_plugin_metadata
from codeintel.analytics.testing.profiles.builder import build_behavioral_coverage
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config.steps_analytics import BehavioralCoverageStepConfig
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain

if TYPE_CHECKING:
    from codeintel.analytics.testing.profiles.types import BehavioralLLMRunner
    from codeintel.build.context import TargetExecutionContext
    from codeintel.core.plugins.types.protocol import PluginMetadata


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
    _core_metadata: ClassVar[CorePluginMetadata] = BEHAVIORAL_COVERAGE_METADATA

    @property
    def metadata(self) -> PluginMetadata:
        """Return protocol-compatible metadata."""
        return to_plugin_metadata(self._core_metadata)

    @property
    def core_metadata(self) -> CorePluginMetadata:
        """Return canonical metadata."""
        return self._core_metadata

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


__all__ = ["BEHAVIORAL_COVERAGE_METADATA", "BehavioralCoveragePlugin"]
