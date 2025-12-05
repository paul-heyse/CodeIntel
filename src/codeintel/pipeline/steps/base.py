"""Pipeline step base types and protocols.

This module provides the foundational types for pipeline step implementations:

- PipelineStep: Protocol for step implementations
- StepPhase: Enum classifying step phases
- StepMetadata: Dataclass for step metadata
- step_to_plugin_metadata: Helper to convert step attributes to PluginMetadata

These types are used by the step registry and all step implementations.
The protocol is designed to be compatible with the unified RegistrablePlugin
protocol from core.plugins.registry.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

from codeintel.core.plugins.types.protocol import PluginMetadata, PluginStage

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


# Mapping from StepPhase to PluginStage for registry compatibility
STEP_PHASE_TO_PLUGIN_STAGE: dict[StepPhase, PluginStage] = {
    StepPhase.INGESTION: "pipeline_ingestion",
    StepPhase.GRAPHS: "pipeline_graphs",
    StepPhase.ANALYTICS: "pipeline_analytics",
    StepPhase.EXPORT: "pipeline_export",
}


def step_to_plugin_metadata(
    name: str,
    description: str,
    phase: StepPhase,
    deps: Sequence[str],
) -> PluginMetadata:
    """Convert pipeline step attributes to PluginMetadata.

    This helper enables pipeline steps to participate in the unified
    plugin registry system by providing compatible metadata.

    Parameters
    ----------
    name
        Step identifier.
    description
        Human-readable description.
    phase
        Pipeline phase the step belongs to.
    deps
        Step dependencies.

    Returns
    -------
    PluginMetadata
        Metadata compatible with BasePluginRegistry.
    """
    return PluginMetadata(
        name=name,
        description=description,
        kind="analytics",  # Steps are treated as analytics-like plugins
        stage=STEP_PHASE_TO_PLUGIN_STAGE[phase],
        depends_on=tuple(deps),
        enabled_by_default=True,
    )


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
    - metadata: Property returning PluginMetadata for registry compatibility.
    - run(): Method to execute the step with a PipelineContext.

    The metadata property enables steps to be registered in the unified
    plugin registry system alongside graph and analytics plugins.

    Examples
    --------
    >>> @dataclass
    ... class MyStep:
    ...     name: str = "my.step"
    ...     description: str = "Does something useful"
    ...     phase: StepPhase = StepPhase.ANALYTICS
    ...     deps: tuple[str, ...] = ()
    ...
    ...     @property
    ...     def metadata(self) -> PluginMetadata:
    ...         return step_to_plugin_metadata(self.name, self.description, self.phase, self.deps)
    ...
    ...     def run(self, ctx: PipelineContext) -> None:
    ...         pass
    """

    name: str
    description: str
    phase: StepPhase
    deps: Sequence[str]

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata for registry compatibility.

        Returns
        -------
        PluginMetadata
            Metadata enabling this step to participate in the unified
            plugin registry system.
        """
        ...

    def run(self, ctx: PipelineContext) -> None:
        """Execute the step using shared context.

        Parameters
        ----------
        ctx
            Pipeline context providing access to gateway, tools, and configuration.
        """


__all__ = [
    "STEP_PHASE_TO_PLUGIN_STAGE",
    "PipelineStep",
    "StepMetadata",
    "StepPhase",
    "step_to_plugin_metadata",
]
