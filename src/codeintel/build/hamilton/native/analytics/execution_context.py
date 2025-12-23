"""ExecutionContext accessors for Hamilton targets."""

from __future__ import annotations

from codeintel.build.hamilton.env import BuildEnv
from codeintel.core.execution import ExecutionContext, RunContext


def execution_context(env: BuildEnv) -> ExecutionContext:
    """Expose the unified ExecutionContext to Hamilton targets.

    Parameters
    ----------
    env
        Build environment input passed to Hamilton.

    Returns
    -------
    ExecutionContext
        Unified execution context for the run.

    Raises
    ------
    ValueError
        If the execution context is missing from the build environment.
    """
    if env.execution_context is None:
        msg = "ExecutionContext is required for Hamilton execution"
        raise ValueError(msg)
    return env.execution_context


def run_context(execution_context: ExecutionContext) -> RunContext:
    """Expose the RunContext for Hamilton targets.

    Parameters
    ----------
    execution_context
        Unified execution context for the run.

    Returns
    -------
    RunContext
        Run metadata for the Hamilton target.
    """
    return execution_context.run


__all__ = ["execution_context", "run_context"]
