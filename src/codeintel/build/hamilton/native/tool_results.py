"""Shared tool step result containers for native targets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from codeintel.build.hamilton.execution_result import ExecutionResult


class HasExecutionResult(Protocol):
    """Protocol for tool outputs carrying an ExecutionResult."""

    @property
    def result(self) -> ExecutionResult:
        """Return the execution status."""
        ...


@dataclass(frozen=True)
class ToolStepOutput:
    """Standard tool step output with status and named output paths.

    Attributes
    ----------
    result
        Execution status for the tool step.
    outputs
        Mapping of output identifiers to filesystem paths.
    """

    result: ExecutionResult
    outputs: Mapping[str, Path] = field(default_factory=dict)

    def path_for(self, name: str) -> Path | None:
        """Return the path for a named output.

        Parameters
        ----------
        name
            Output identifier to resolve.

        Returns
        -------
        Path | None
            Output path if present.
        """
        return self.outputs.get(name)


__all__ = [
    "HasExecutionResult",
    "ToolStepOutput",
]
