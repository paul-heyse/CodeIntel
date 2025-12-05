"""Pipeline step base types and protocols.

This module provides the foundational types for pipeline step implementations:

- PipelineStep: Protocol for step implementations
- StepPhase: Enum classifying step phases
- StepMetadata: Dataclass for step metadata

These types are used by the step registry and all step implementations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from codeintel.pipeline.execution.context import PipelineContext


class StepPhase(Enum):
    """Classification of pipeline step phases.

    Attributes
    ----------
    INGESTION
        Steps that scan and index source code.
    GRAPHS
        Steps that build call graphs, import graphs, etc.
    ANALYTICS
        Steps that compute metrics, profiles, and insights.
    EXPORT
        Steps that export data to external formats.
    """

    INGESTION = "ingestion"
    GRAPHS = "graphs"
    ANALYTICS = "analytics"
    EXPORT = "export"


@dataclass(frozen=True)
class StepMetadata:
    """Machine-readable metadata for a pipeline step.

    Parameters
    ----------
    name
        Unique step identifier.
    description
        Human-readable description of what the step does.
    phase
        Pipeline phase this step belongs to.
    deps
        Names of steps this step depends on.
    """

    name: str
    description: str
    phase: StepPhase
    deps: tuple[str, ...]


class PipelineStep(Protocol):
    """Contract for pipeline steps.

    Each step must define:

    - name: Unique identifier for the step.
    - description: Human-readable description of the step's purpose.
    - phase: The pipeline phase this step belongs to.
    - deps: Sequence of step names this step depends on.
    - run(): Method to execute the step with a PipelineContext.

    Examples
    --------
    >>> @dataclass
    ... class MyStep:
    ...     name: str = "my.step"
    ...     description: str = "Does something useful"
    ...     phase: StepPhase = StepPhase.ANALYTICS
    ...     deps: tuple[str, ...] = ()
    ...
    ...     def run(self, ctx: PipelineContext) -> None:
    ...         pass
    """

    name: str
    description: str
    phase: StepPhase
    deps: Sequence[str]

    def run(self, ctx: PipelineContext) -> None:
        """Execute the step using shared context.

        Parameters
        ----------
        ctx
            Pipeline context providing access to gateway, tools, and configuration.
        """


__all__ = [
    "PipelineStep",
    "StepMetadata",
    "StepPhase",
]
