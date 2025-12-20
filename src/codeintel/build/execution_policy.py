"""Execution policy consolidation for build targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.hamilton.execution_options import BuildExecutionOptions
from codeintel.build.resources import IsolationKind, TargetExecution

if TYPE_CHECKING:
    from codeintel.build.targets import TargetGraph


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


def effective_max_workers_for_graph(
    *,
    run_options: BuildExecutionOptions,
    graph: TargetGraph,
) -> int | None:
    """Return the effective max workers for a target graph.

    Parameters
    ----------
    run_options
        Run-level execution options.
    graph
        Target graph whose per-target execution limits are evaluated.

    Returns
    -------
    int | None
        Effective maximum worker count for the run.
    """
    limits: list[int] = []
    for target in graph.all_targets:
        policy = ExecutionPolicy(run_options=run_options, target_execution=target.execution)
        max_workers = policy.effective_max_workers()
        if max_workers is not None:
            limits.append(max_workers)
    if not limits:
        return run_options.max_workers
    return min(limits)


__all__ = ["ExecutionPolicy", "effective_max_workers_for_graph"]
