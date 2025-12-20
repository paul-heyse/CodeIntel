"""Execution policy consolidation for build targets."""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.build.hamilton.execution_options import BuildExecutionOptions
from codeintel.build.resources import IsolationKind, TargetExecution


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    """Resolved execution policy for a target and run configuration."""

    run_options: BuildExecutionOptions
    target_execution: TargetExecution

    @property
    def isolation(self) -> IsolationKind:
        """Return the isolation level for target execution.

        Returns
        -------
        IsolationKind
            Isolation level for the target execution.
        """
        return self.target_execution.isolation

    @property
    def max_runtime_ms(self) -> int:
        """Return the max runtime for this target.

        Returns
        -------
        int
            Max runtime in milliseconds.
        """
        return self.target_execution.max_runtime_ms

    @property
    def supports_incremental(self) -> bool:
        """Return True if the target supports incremental execution.

        Returns
        -------
        bool
            True when the target supports incremental execution.
        """
        return self.target_execution.supports_incremental

    def effective_max_workers(self) -> int | None:
        """Resolve the effective max workers for this target.

        Returns
        -------
        int | None
            Effective worker limit after combining run and target caps.
        """
        run_limit = self.run_options.max_workers
        target_limit = self.target_execution.max_parallelism
        if run_limit is None:
            return target_limit
        if target_limit is None:
            return run_limit
        return min(run_limit, target_limit)


__all__ = ["ExecutionPolicy"]
