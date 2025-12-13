"""Protocol-aligned ingestion/build fakes for realistic test wiring.

These fakes implement the public interfaces used in production so tests can
exercise real DuckDB-backed storage and build orchestration without resorting
to monkeypatching. They record calls for assertions while delegating to the
real implementations when provided.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.build.executor import BuildResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.build.executor import BuildExecutor
    from codeintel.build.plan import BuildPlan


@dataclass
class RecordingBuildExecutor:
    """BuildExecutor stand-in that records execute calls and optionally delegates.

    Parameters
    ----------
    delegate
        Optional real BuildExecutor to delegate execution to for realism.
    run_id_factory
        Optional callable to generate deterministic run_ids for tests.
    """

    delegate: BuildExecutor | None = None
    run_id_factory: Callable[[], str] | None = None
    executions: list[tuple[BuildPlan, bool]] = field(default_factory=list)

    def execute(self, plan: BuildPlan, *, dry_run: bool = False) -> BuildResult:
        """Record execution and delegate or return a minimal success result.

        Returns
        -------
        BuildResult
            Execution result from the delegate or a minimal success result.
        """
        self.executions.append((plan, dry_run))
        if self.delegate is not None:
            return self.delegate.execute(plan, dry_run=dry_run)

        run_id = self.run_id_factory() if self.run_id_factory is not None else "recording"
        duration_ms = 0.0
        return BuildResult(
            run_id=run_id,
            plan=plan,
            status="succeeded",
            completed_targets=tuple(step.target for stage in plan.stages for step in stage.steps),
            failed_targets=(),
            skipped_targets=(),
            duration_ms=duration_ms,
            error_summary=None,
        )


__all__ = [
    "RecordingBuildExecutor",
]
