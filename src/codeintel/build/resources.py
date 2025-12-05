"""Resource and execution configuration for build targets.

This module defines types that specify what resources a target needs
and how it should be executed. These replace the scattered ClassVars
previously defined on plugin classes.

Example
-------
>>> from codeintel.build.resources import TargetResources, TargetExecution
>>> resources = TargetResources(
...     tracker=True,
...     modules=True,
...     tools=("scip-python", "scip"),
... )
>>> execution = TargetExecution(
...     cpu_intensive=True,
...     max_runtime_ms=300000,
...     isolation="process",
... )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "IsolationKind",
    "TargetExecution",
    "TargetResources",
]


IsolationKind = Literal["none", "thread", "process"]
"""Isolation level for target execution.

- "none": Run in the main process/thread (fast, shared state)
- "thread": Run in a separate thread (concurrent, shared memory)
- "process": Run in a subprocess (isolated, no shared state)
"""


@dataclass(frozen=True)
class TargetResources:
    """Resources required by a target for execution.

    This replaces the scattered `requires`, `tool_dependencies`, and
    `tracker_required` ClassVars previously defined on plugins.

    Parameters
    ----------
    tracker
        Whether this target needs the change tracker for incremental builds.
    modules
        Whether this target needs access to the module list.
    gateway
        Whether this target needs database access. Almost always True.
    tools
        External tools required (e.g., "scip-python", "pyright").
        Build system validates tool availability at planning time.

    Examples
    --------
    >>> resources = TargetResources(
    ...     tracker=True,
    ...     modules=True,
    ...     tools=("scip-python", "scip"),
    ... )
    """

    tracker: bool = False
    modules: bool = False
    gateway: bool = True
    tools: tuple[str, ...] = ()

    def requires_any_tool(self) -> bool:
        """Check if this target requires any external tools.

        Returns
        -------
        bool
            True if tools tuple is non-empty.
        """
        return len(self.tools) > 0


@dataclass(frozen=True)
class TargetExecution:
    """Execution hints and constraints for a target.

    This replaces `resource_hints`, `isolation_kind`, and
    `supports_incremental` ClassVars previously defined on plugins.

    Parameters
    ----------
    cpu_intensive
        Whether this target is CPU-bound (affects parallelization).
    io_intensive
        Whether this target is I/O-bound (affects scheduling).
    memory_intensive
        Whether this target requires significant memory.
    max_runtime_ms
        Maximum allowed execution time in milliseconds.
        Targets exceeding this will be terminated with TimeoutError.
    isolation
        Isolation level for execution (none, thread, process).
    supports_incremental
        Whether this target can do incremental computation.
        If True, the change tracker can be used to skip unchanged files.
    max_parallelism
        Maximum number of parallel workers for this target.
        None means use system default.

    Examples
    --------
    >>> execution = TargetExecution(
    ...     cpu_intensive=True,
    ...     io_intensive=True,
    ...     max_runtime_ms=300000,  # 5 minutes
    ...     isolation="process",
    ...     supports_incremental=True,
    ... )
    """

    cpu_intensive: bool = False
    io_intensive: bool = False
    memory_intensive: bool = False
    max_runtime_ms: int = 60000  # 1 minute default
    isolation: IsolationKind = "thread"
    supports_incremental: bool = True
    max_parallelism: int | None = None

    def estimated_duration_ms(self) -> int:
        """Estimate execution duration for planning.

        Returns a rough estimate based on execution characteristics.
        This is used for build planning and progress reporting.

        Returns
        -------
        int
            Estimated duration in milliseconds.
        """
        base = 5000  # 5 second base
        if self.cpu_intensive:
            base *= 4
        if self.io_intensive:
            base *= 2
        if self.memory_intensive:
            base *= 2
        return min(base, self.max_runtime_ms)


# Default instances for common patterns
DEFAULT_RESOURCES = TargetResources()
DEFAULT_EXECUTION = TargetExecution()

# For CPU-intensive targets like AST parsing
CPU_INTENSIVE_EXECUTION = TargetExecution(
    cpu_intensive=True,
    isolation="process",
    max_runtime_ms=120000,
)

# For I/O-intensive targets like git history
IO_INTENSIVE_EXECUTION = TargetExecution(
    io_intensive=True,
    max_runtime_ms=180000,
)

# For external tool targets like SCIP
TOOL_EXECUTION = TargetExecution(
    cpu_intensive=True,
    io_intensive=True,
    isolation="process",
    max_runtime_ms=300000,
)
