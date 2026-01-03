"""Graph analytics target commands.

Provide commands for listing graph build targets and their execution plans
using the handler-based operation registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from cyclopts import App

from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.graphs import (
    DependencyPolicy,
    SelectionPolicy,
    graph_plugins_handler,
    graph_targets_handler,
    graph_targets_list_handler,
    graph_targets_plan_handler,
)
from codeintel.cli.options.registry import (
    GRAPH_DEPENDENCY_POLICY,
    GRAPH_NAMES,
    GRAPH_PLAN,
    GRAPH_SELECTION_POLICY,
)
from codeintel.cli.options.shared_flags import SharedFlagsProtocol, shared_flags_field
from codeintel.cli.options.types import CommandPath, option_param
from codeintel.cli.rendering.types import OutputFormat

graphs_app = App(
    name="graph",
    help="Graph analytics target commands.",
)

_GRAPH_CONFIG = CommandConfig(require_runtime=True, require_gateway=True)

GRAPH_TARGETS_LIST_PATH: CommandPath = ("graph", "targets-list")
GRAPH_TARGETS_PLAN_PATH: CommandPath = ("graph", "targets-plan")
GRAPH_PLUGINS_PATH: CommandPath = ("graph", "plugins")
GRAPH_TARGETS_PATH: CommandPath = ("graph", "targets")

_GRAPH_TARGETS_LIST_FLAGS_FIELD = shared_flags_field(
    GRAPH_TARGETS_LIST_PATH,
    default_output_format=OutputFormat.JSONL,
)
_GRAPH_TARGETS_PLAN_FLAGS_FIELD = shared_flags_field(GRAPH_TARGETS_PLAN_PATH)
_GRAPH_PLUGINS_FLAGS_FIELD = shared_flags_field(GRAPH_PLUGINS_PATH)
_GRAPH_TARGETS_FLAGS_FIELD = shared_flags_field(GRAPH_TARGETS_PATH)


@cli_command("graph.targets.list", handler=graph_targets_list_handler, config=_GRAPH_CONFIG)
@graphs_app.command(name="targets-list")
@dataclass(frozen=True)
class GraphTargetsListCommand:
    """List graph build targets."""

    names: Annotated[
        list[str] | None,
        option_param(GRAPH_NAMES, command_path=GRAPH_TARGETS_LIST_PATH),
    ] = None
    flags: SharedFlagsProtocol = _GRAPH_TARGETS_LIST_FLAGS_FIELD


@cli_command("graph.targets.plan", handler=graph_targets_plan_handler, config=_GRAPH_CONFIG)
@graphs_app.command(name="targets-plan")
@dataclass(frozen=True)
class GraphTargetsPlanCommand:
    """Display an execution plan for graph targets in dependency order."""

    names: Annotated[
        list[str] | None,
        option_param(GRAPH_NAMES, command_path=GRAPH_TARGETS_PLAN_PATH),
    ] = None
    flags: SharedFlagsProtocol = _GRAPH_TARGETS_PLAN_FLAGS_FIELD


@cli_command("graph.plugins", handler=graph_plugins_handler, config=_GRAPH_CONFIG)
@graphs_app.command(name="plugins")
@dataclass(frozen=True)
class GraphPluginsCommand:
    """List graph plugins or show an execution plan; use --plan for ordering."""

    plan: Annotated[
        bool,
        option_param(GRAPH_PLAN, command_path=GRAPH_PLUGINS_PATH),
    ] = False
    names: Annotated[
        list[str] | None,
        option_param(GRAPH_NAMES, command_path=GRAPH_PLUGINS_PATH),
    ] = None
    selection_policy: Annotated[
        SelectionPolicy,
        option_param(GRAPH_SELECTION_POLICY, command_path=GRAPH_PLUGINS_PATH),
    ] = SelectionPolicy.LENIENT
    dependency_policy: Annotated[
        DependencyPolicy,
        option_param(GRAPH_DEPENDENCY_POLICY, command_path=GRAPH_PLUGINS_PATH),
    ] = DependencyPolicy.STRICT
    flags: SharedFlagsProtocol = _GRAPH_PLUGINS_FLAGS_FIELD


@cli_command("graph.targets", handler=graph_targets_handler, config=_GRAPH_CONFIG)
@graphs_app.command(name="targets")
@dataclass(frozen=True)
class GraphTargetsCommand:
    """List graph targets or show execution plan; use --plan for ordering."""

    plan: Annotated[
        bool,
        option_param(GRAPH_PLAN, command_path=GRAPH_TARGETS_PATH),
    ] = False
    names: Annotated[
        list[str] | None,
        option_param(GRAPH_NAMES, command_path=GRAPH_TARGETS_PATH),
    ] = None
    flags: SharedFlagsProtocol = _GRAPH_TARGETS_FLAGS_FIELD


__all__ = [
    "DependencyPolicy",
    "GraphPluginsCommand",
    "GraphTargetsCommand",
    "GraphTargetsListCommand",
    "GraphTargetsPlanCommand",
    "SelectionPolicy",
    "graphs_app",
]
