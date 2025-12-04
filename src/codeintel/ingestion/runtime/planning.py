"""Planning utilities for ingestion plugin execution.

This module provides execution planning for ingestion plugins, including
dependency resolution, topological ordering, and plan creation.
Analogous to graphs/runtime/planning.py for structural alignment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import uuid4

from codeintel.ingestion.plugins.protocol import (
    IngestPluginPlan,
    IngestPluginProtocol,
    IngestPluginSkip,
)
from codeintel.ingestion.plugins.registry import PlanOptions

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.ingestion.plugins.registry import IngestPluginRegistry


@dataclass(frozen=True)
class PluginExecutionPlan:
    """Execution plan for a batch of ingestion plugins.

    Attributes
    ----------
    plan_id
        Unique plan identifier.
    plugins
        Ordered list of plugins to execute.
    skipped
        Plugins that were skipped with reasons.
    snapshot
        Snapshot reference for the execution.
    """

    plan_id: str = field(default_factory=lambda: uuid4().hex)
    plugins: tuple[IngestPluginProtocol, ...] = ()
    skipped: tuple[IngestPluginSkip, ...] = ()
    snapshot: SnapshotRef | None = None

    @property
    def plugin_names(self) -> tuple[str, ...]:
        """Return names of plugins in the plan."""
        return tuple(p.metadata.name for p in self.plugins)

    @property
    def skipped_names(self) -> tuple[str, ...]:
        """Return names of skipped plugins."""
        return tuple(s.name for s in self.skipped)


@dataclass
class IngestPlanContext:
    """Context for creating an execution plan.

    Attributes
    ----------
    snapshot
        Snapshot reference.
    registry
        Plugin registry.
    options
        Planning options.
    """

    snapshot: SnapshotRef
    registry: IngestPluginRegistry
    options: PlanOptions = field(default_factory=PlanOptions)


def plan_ingest_plugins(
    context: IngestPlanContext,
) -> PluginExecutionPlan:
    """Create an execution plan for ingestion plugins.

    Parameters
    ----------
    context
        Planning context with registry and options.

    Returns
    -------
    PluginExecutionPlan
        Execution plan with ordered plugins.
    """
    registry = context.registry
    options = context.options

    # Build the plan using registry
    plan: IngestPluginPlan = registry.plan(options)

    return PluginExecutionPlan(
        plugins=tuple(plan.plugins),
        skipped=tuple(plan.skipped_plugins),
        snapshot=context.snapshot,
    )


def resolve_plugin_order(
    plugins: Sequence[IngestPluginProtocol],
) -> list[IngestPluginProtocol]:
    """Resolve plugin execution order based on dependencies.

    Performs topological sort to ensure dependencies are executed
    before dependents.

    Parameters
    ----------
    plugins
        Plugins to order.

    Returns
    -------
    list[IngestPluginProtocol]
        Plugins in dependency order.

    Raises
    ------
    ValueError
        If circular dependencies are detected.
    """
    # Build dependency graph
    name_to_plugin = {p.metadata.name: p for p in plugins}
    in_degree: dict[str, int] = {p.metadata.name: 0 for p in plugins}

    for plugin in plugins:
        for dep in plugin.metadata.requires:
            if dep in name_to_plugin:
                in_degree[plugin.metadata.name] += 1

    # Topological sort (Kahn's algorithm)
    queue = [name for name, degree in in_degree.items() if degree == 0]
    result: list[IngestPluginProtocol] = []

    while queue:
        name = queue.pop(0)
        result.append(name_to_plugin[name])

        for plugin in plugins:
            if name in plugin.metadata.requires:
                in_degree[plugin.metadata.name] -= 1
                if in_degree[plugin.metadata.name] == 0:
                    queue.append(plugin.metadata.name)

    if len(result) != len(plugins):
        message = "Circular dependency detected in plugin graph"
        raise ValueError(message)

    return result


__all__ = [
    "IngestPlanContext",
    "PlanOptions",
    "PluginExecutionPlan",
    "plan_ingest_plugins",
    "resolve_plugin_order",
]
