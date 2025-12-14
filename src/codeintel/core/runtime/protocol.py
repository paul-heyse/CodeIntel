"""Runtime protocol definitions.

This module provides protocols for executors and runtimes,
enabling dependency injection and testability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.core.runtime.tracking import StepResult


@runtime_checkable
class ExecutorProtocol(Protocol):
    """Protocol for step execution.

    Implementations execute a step/task and return structured results.

    Examples
    --------
    >>> class MyExecutor:
    ...     def execute(self, step_name: str, **kwargs: object) -> StepResult:
    ...         # Execute the step
    ...         return StepResult.success(step_name)
    """

    def execute(self, step_name: str, **kwargs: object) -> StepResult:
        """Execute a named step.

        Parameters
        ----------
        step_name
            Name of the step to execute.
        **kwargs
            Step-specific configuration.

        Returns
        -------
        StepResult
            Structured result of execution.
        """
        ...

    @property
    def completed_steps(self) -> list[str]:
        """Return names of completed steps.

        Returns
        -------
        list[str]
            Names of successfully completed steps.
        """
        ...


@runtime_checkable
class RuntimeProtocol(Protocol):
    """Protocol for runtime configuration access.

    Implementations provide access to resolved runtime configuration
    with caching and lazy resolution.

    Examples
    --------
    >>> class MyRuntime:
    ...     @property
    ...     def config(self) -> Mapping[str, object]:
    ...         return self._config
    ...
    ...     def refresh(self) -> None:
    ...         self._config = self._resolve_config()
    """

    @property
    def config(self) -> Mapping[str, object]:
        """Return runtime configuration.

        Returns
        -------
        Mapping[str, object]
            Configuration mapping.
        """
        ...

    def refresh(self) -> None:
        """Refresh runtime configuration.

        Invalidate any cached values and re-resolve configuration.
        """
        ...


__all__ = [
    "ExecutorProtocol",
    "RuntimeProtocol",
]
