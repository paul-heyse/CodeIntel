"""Graph handlers.

Handlers for graph plugin listing and execution planning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from codeintel.cli.context import CommandContext
from codeintel.cli.core import CliResult
from codeintel.cli.errors.results import fail_invalid_policy
from codeintel.cli.execution.registry import OperationSpec, register_operation
from codeintel.graphs.core.registry import (
    DependencyPolicy,
    PlanningOptions,
    SelectionPolicy,
    list_graph_plugins,
    plan_graph_plugins,
)

LOG = logging.getLogger(__name__)


class PlanMode(Enum):
    """Graph plugin listing mode."""

    LIST = "list"
    PLAN = "plan"


@dataclass(frozen=True)
class GraphPluginInfo:
    """Information about a single graph plugin.

    Parameters
    ----------
    name
        Plugin name.
    description
        Plugin description.
    stage
        Plugin execution stage.
    enabled_by_default
        Whether enabled by default.
    depends_on
        Dependencies.
    provides
        What the plugin provides.
    """

    name: str
    description: str
    stage: str
    enabled_by_default: bool
    depends_on: list[str]
    provides: list[str]

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "name": self.name,
            "description": self.description,
            "stage": self.stage,
            "enabled_by_default": self.enabled_by_default,
            "depends_on": self.depends_on,
            "provides": self.provides,
        }


@dataclass(frozen=True)
class GraphPluginsResult:
    """Result from listing graph plugins.

    Parameters
    ----------
    plugins
        List of plugin information.
    count
        Total count of plugins.
    """

    plugins: list[GraphPluginInfo]
    count: int

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "plugins": [p.to_dict() for p in self.plugins],
            "count": self.count,
        }


@dataclass(frozen=True)
class GraphPlanStage:
    """A stage in the graph execution plan.

    Parameters
    ----------
    stage
        Stage number.
    plugins
        Plugins to execute in this stage.
    """

    stage: int
    plugins: list[str]

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "stage": self.stage,
            "plugins": self.plugins,
        }


@dataclass(frozen=True)
class GraphPlanResult:
    """Result from planning graph execution.

    Parameters
    ----------
    stages
        List of execution stages.
    total_plugins
        Total number of plugins.
    disabled
        Plugins that were disabled.
    """

    stages: list[GraphPlanStage]
    total_plugins: int
    disabled: list[str]

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "stages": [s.to_dict() for s in self.stages],
            "total_plugins": self.total_plugins,
            "disabled": self.disabled,
        }


def graph_plugins_list_handler(
    ctx: CommandContext,
) -> CliResult[GraphPluginsResult]:
    """List registered graph plugins.

    Parameters
    ----------
    ctx
        Command context with params:
        - names: Optional plugin names to filter.
        - enable: Optional plugins to enable.
        - disable: Optional plugins to disable.
        - include_disabled: Whether to include disabled plugins.

    Returns
    -------
    CliResult[GraphPluginsResult]
        List of plugins.
    """
    names_tuple = ctx.params.get_tuple("names")
    names: tuple[str, ...] | None = names_tuple if names_tuple else None
    include_disabled = ctx.params.get_bool("include_disabled")

    LOG.info("Listing graph plugins (names=%s)", names)

    plugins = list_graph_plugins()

    if names:
        plugins = [p for p in plugins if p.metadata.name in names]

    if not include_disabled:
        plugins = [p for p in plugins if p.metadata.enabled_by_default]

    plugin_infos = [
        GraphPluginInfo(
            name=p.metadata.name,
            description=p.metadata.description,
            stage=p.metadata.stage,
            enabled_by_default=p.metadata.enabled_by_default,
            depends_on=list(p.metadata.depends_on),
            provides=list(p.metadata.provides),
        )
        for p in plugins
    ]

    return CliResult.ok(
        GraphPluginsResult(
            plugins=plugin_infos,
            count=len(plugin_infos),
        )
    )


def graph_plugins_plan_handler(
    ctx: CommandContext,
) -> CliResult[GraphPlanResult]:
    """Display an execution plan for graph plugins.

    Parameters
    ----------
    ctx
        Command context with params:
        - names: Optional plugin names to include.
        - enable: Optional plugins to enable.
        - disable: Optional plugins to disable.
        - selection_policy: Selection policy (explicit_only, include_defaults).
        - dependency_policy: Dependency policy (include_all, skip_satisfied).

    Returns
    -------
    CliResult[GraphPlanResult]
        Execution plan.
    """
    names_tuple = ctx.params.get_tuple("names")
    names: tuple[str, ...] | None = names_tuple if names_tuple else None
    enable_tuple = ctx.params.get_tuple("enable")
    enable: tuple[str, ...] | None = enable_tuple if enable_tuple else None
    disable = ctx.params.get_tuple("disable") or ()
    selection_policy_str = ctx.params.get_str("selection_policy", "lenient")
    dependency_policy_str = ctx.params.get_str("dependency_policy", "strict")

    # Parse policies
    try:
        selection_policy = SelectionPolicy(selection_policy_str)
    except ValueError:
        return fail_invalid_policy("selection", selection_policy_str or "")

    try:
        dependency_policy = DependencyPolicy(dependency_policy_str)
    except ValueError:
        return fail_invalid_policy("dependency", dependency_policy_str or "")

    LOG.info(
        "Planning graph plugins (names=%s, enable=%s, disable=%s)",
        names,
        enable,
        disable,
    )

    options = PlanningOptions(
        dependency_policy=dependency_policy,
        selection_policy=selection_policy,
    )

    # Call with the correct API signature
    plan = plan_graph_plugins(
        plugin_names=list(names) if names else None,
        enabled=list(enable) if enable else None,
        disabled=list(disable) if disable else None,
        plan_options=options,
    )

    # Convert plan to result format - GraphPluginPlan has `plugins` not `stages`
    # Simplified: treat all plugins as one stage
    plugin_names_list = [p.metadata.name for p in plan.plugins]
    stages = [
        GraphPlanStage(
            stage=1,
            plugins=plugin_names_list,
        )
    ]

    skipped_names = [skip.name for skip in plan.skipped_plugins]

    return CliResult.ok(
        GraphPlanResult(
            stages=stages,
            total_plugins=len(plan.plugins),
            disabled=skipped_names,
        )
    )


# -----------------------------------------------------------------------------
# Operation Registrations
# -----------------------------------------------------------------------------

register_operation(
    OperationSpec(
        operation_id="graphs.plugins.list",
        name="List Graph Plugins",
        description="List registered graph plugins",
        handler=graph_plugins_list_handler,
        group="graphs",
        require_runtime=False,
        require_gateway=False,
    )
)

register_operation(
    OperationSpec(
        operation_id="graphs.plugins.plan",
        name="Graph Plugins Plan",
        description="Display an execution plan for graph plugins",
        handler=graph_plugins_plan_handler,
        group="graphs",
        require_runtime=False,
        require_gateway=False,
    )
)

__all__ = [
    "GraphPlanResult",
    "GraphPlanStage",
    "GraphPluginInfo",
    "GraphPluginsResult",
    "PlanMode",
    "graph_plugins_list_handler",
    "graph_plugins_plan_handler",
]
