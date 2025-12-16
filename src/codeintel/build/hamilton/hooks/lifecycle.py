"""Extended lifecycle hooks for Hamilton build execution.

This module provides additional lifecycle hooks beyond the core
manifest, telemetry, and contract hooks:

- ProgressBar: Visual progress indicator using tqdm
- BuildTimingHook: Detailed timing metrics for optimization
- ConditionalHook: Conditional execution wrapper

These hooks integrate with Hamilton's lifecycle API and can be composed
via Builder.with_adapters().

Examples
--------
Using ProgressBar for visual feedback:

>>> from codeintel.build.hamilton.hooks.lifecycle import ProgressBarHook
>>> dr = (
...     driver.Builder()
...     .with_modules(modules)
...     .with_adapters(ProgressBarHook(desc="Building targets"))
...     .build()
... )

Using BuildTimingHook for performance analysis:

>>> from codeintel.build.hamilton.hooks.lifecycle import BuildTimingHook
>>> timing_hook = BuildTimingHook()
>>> dr = driver.Builder().with_modules(modules).with_adapters(timing_hook).build()
>>> # After execution:
>>> print(timing_hook.get_slowest_nodes(n=10))
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hamilton.lifecycle import NodeExecutionHook

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "BuildTimingHook",
    "ConditionalHook",
    "NodeTimingRecord",
    "ProgressBarHook",
    "create_progress_hook",
]

log = logging.getLogger(__name__)


@dataclass
class NodeTimingRecord:
    """Timing record for a single node execution.

    Attributes
    ----------
    node_name
        Name of the executed node.
    duration_seconds
        Execution time in seconds.
    start_time
        Unix timestamp when execution started.
    task_id
        Optional task identifier for parallel execution.
    """

    node_name: str
    duration_seconds: float
    start_time: float
    task_id: str | None = None


class ProgressBarHook(NodeExecutionHook):
    """Progress bar hook using tqdm.

    Provides visual feedback during build execution by showing
    a progress bar for node execution.

    Parameters
    ----------
    desc
        Description to show in the progress bar.
    max_node_name_width
        Maximum width for node names in display.
    disable
        If True, disable the progress bar (useful for CI).
    tqdm_kwargs
        Additional keyword arguments passed to tqdm.

    Examples
    --------
    >>> hook = ProgressBarHook(desc="Building analytics")
    >>> dr = Builder().with_adapters(hook).build()
    """

    def __init__(
        self,
        desc: str = "Graph execution",
        max_node_name_width: int = 50,
        disable: bool = False,
        **tqdm_kwargs: Any,
    ) -> None:
        """Initialize the progress bar hook."""
        self.desc = desc
        self.max_node_name_width = max_node_name_width
        self.disable = disable
        self.tqdm_kwargs = tqdm_kwargs
        self._delegate: NodeExecutionHook | None = None

    def _ensure_delegate(self) -> NodeExecutionHook | None:
        """Lazily create the delegate progress bar hook.

        Returns
        -------
        NodeExecutionHook | None
            TQDM-based progress hook when available, otherwise None.
        """
        if self._delegate is not None:
            return self._delegate

        if self.disable:
            return None

        try:
            from hamilton.plugins.h_tqdm import ProgressBar

            self._delegate = ProgressBar(
                desc=self.desc,
                max_node_name_width=self.max_node_name_width,
                **self.tqdm_kwargs,
            )
        except ImportError:
            log.warning(
                "tqdm not available, progress bar disabled. Install with: pip install tqdm",
            )
        return self._delegate

    def run_before_node_execution(
        self,
        *,
        node_name: str,
        node_tags: dict[str, Any],
        node_kwargs: dict[str, Any],
        task_id: str | None,
        **future_kwargs: Any,
    ) -> dict[str, Any] | None:
        """Execute before each node runs.

        Returns
        -------
        dict[str, Any] | None
            Optional context forwarded to the delegate hook.
        """
        delegate = self._ensure_delegate()
        if delegate is not None:
            return delegate.run_before_node_execution(
                node_name=node_name,
                node_tags=node_tags,
                node_kwargs=node_kwargs,
                task_id=task_id,
                **future_kwargs,
            )
        return None

    def run_after_node_execution(
        self,
        *,
        node_name: str,
        node_tags: dict[str, Any],
        node_kwargs: dict[str, Any],
        node_return_type: type,
        result: Any,
        error: Exception | None,
        success: bool,
        task_id: str | None,
        **future_kwargs: Any,
    ) -> dict[str, Any] | None:
        """Execute after each node completes.

        Returns
        -------
        dict[str, Any] | None
            Optional context forwarded to the delegate hook.
        """
        delegate = self._ensure_delegate()
        if delegate is not None:
            return delegate.run_after_node_execution(
                node_name=node_name,
                node_tags=node_tags,
                node_kwargs=node_kwargs,
                node_return_type=node_return_type,
                result=result,
                error=error,
                success=success,
                task_id=task_id,
                **future_kwargs,
            )
        return None


class BuildTimingHook(NodeExecutionHook):
    """Hook for collecting detailed node execution timing.

    Collects timing information for each node execution, useful for
    identifying bottlenecks and optimizing build performance.

    Parameters
    ----------
    min_duration_to_log
        Minimum duration in seconds to log individual nodes.

    Examples
    --------
    >>> timing_hook = BuildTimingHook()
    >>> dr = Builder().with_adapters(timing_hook).build()
    >>> dr.execute(["output"])
    >>> slowest = timing_hook.get_slowest_nodes(n=5)
    >>> for record in slowest:
    ...     print(f"{record.node_name}: {record.duration_seconds:.3f}s")
    """

    def __init__(self, min_duration_to_log: float = 1.0) -> None:
        """Initialize the timing hook."""
        self.min_duration_to_log = min_duration_to_log
        self._timings: dict[tuple[str, str | None], float] = {}
        self._records: list[NodeTimingRecord] = []

    def run_before_node_execution(
        self,
        *,
        node_name: str,
        node_tags: dict[str, Any],  # noqa: ARG002
        node_kwargs: dict[str, Any],  # noqa: ARG002
        task_id: str | None,
        **future_kwargs: Any,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Record start time before node execution.

        Returns
        -------
        dict[str, Any] | None
            Optional context for downstream hooks (unused).
        """
        key = (node_name, task_id)
        self._timings[key] = time.perf_counter()
        return None

    def run_after_node_execution(
        self,
        *,
        node_name: str,
        node_tags: dict[str, Any],  # noqa: ARG002
        node_kwargs: dict[str, Any],  # noqa: ARG002
        node_return_type: type,  # noqa: ARG002
        result: Any,  # noqa: ARG002
        error: Exception | None,  # noqa: ARG002
        success: bool,  # noqa: ARG002
        task_id: str | None,
        **future_kwargs: Any,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Record duration after node execution.

        Returns
        -------
        dict[str, Any] | None
            Optional context for downstream hooks (unused).
        """
        key = (node_name, task_id)
        start_time = self._timings.pop(key, None)
        if start_time is None:
            return None

        duration = time.perf_counter() - start_time
        record = NodeTimingRecord(
            node_name=node_name,
            duration_seconds=duration,
            start_time=start_time,
            task_id=task_id,
        )
        self._records.append(record)

        if duration >= self.min_duration_to_log:
            log.info("Node %s took %.3fs", node_name, duration)

        return None

    def get_records(self) -> list[NodeTimingRecord]:
        """Get all timing records.

        Returns
        -------
        list[NodeTimingRecord]
            All recorded node timings.
        """
        return list(self._records)

    def get_slowest_nodes(self, n: int = 10) -> list[NodeTimingRecord]:
        """Get the N slowest nodes.

        Parameters
        ----------
        n
            Number of nodes to return.

        Returns
        -------
        list[NodeTimingRecord]
            Top N slowest nodes sorted by duration.
        """
        sorted_records = sorted(
            self._records,
            key=lambda r: r.duration_seconds,
            reverse=True,
        )
        return sorted_records[:n]

    def total_duration(self) -> float:
        """Get total execution time across all nodes.

        Returns
        -------
        float
            Sum of all node durations in seconds.
        """
        return sum(r.duration_seconds for r in self._records)

    def reset(self) -> None:
        """Clear all timing records."""
        self._timings.clear()
        self._records.clear()


@dataclass
class ConditionalHook:
    """Wrapper to conditionally enable a hook.

    Enables a hook based on a predicate function, allowing dynamic
    hook configuration based on environment or config.

    Parameters
    ----------
    hook
        The hook to conditionally enable.
    condition
        Function that returns True if the hook should be enabled.

    Examples
    --------
    >>> import os
    >>> hook = ConditionalHook(
    ...     ProgressBarHook(),
    ...     condition=lambda: os.getenv("CI") != "true",
    ... )
    """

    hook: NodeExecutionHook
    condition: Callable[[], bool]
    _enabled: bool | None = field(default=None, init=False)

    def _is_enabled(self) -> bool:
        """Check if hook is enabled (cached).

        Returns
        -------
        bool
            True when the underlying hook should be executed.
        """
        if self._enabled is None:
            self._enabled = self.condition()
        return self._enabled

    def run_before_node_execution(
        self,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Execute before hook if enabled.

        Returns
        -------
        dict[str, Any] | None
            Delegate result when enabled, otherwise None.
        """
        if self._is_enabled():
            return self.hook.run_before_node_execution(**kwargs)
        return None

    def run_after_node_execution(
        self,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Execute after hook if enabled.

        Returns
        -------
        dict[str, Any] | None
            Delegate result when enabled, otherwise None.
        """
        if self._is_enabled():
            return self.hook.run_after_node_execution(**kwargs)
        return None


def create_progress_hook(
    desc: str = "Building",
    *,
    disable_in_ci: bool = True,
) -> ProgressBarHook:
    """Create a progress bar hook with sensible defaults.

    Factory function for creating progress hooks with common
    configurations.

    Parameters
    ----------
    desc
        Description for the progress bar.
    disable_in_ci
        If True, disable progress bar when CI env var is set.

    Returns
    -------
    ProgressBarHook
        Configured progress bar hook.

    Examples
    --------
    >>> hook = create_progress_hook("Building targets", disable_in_ci=True)
    """
    import os

    disable = False
    if disable_in_ci and os.getenv("CI") == "true":
        disable = True

    return ProgressBarHook(desc=desc, disable=disable)
