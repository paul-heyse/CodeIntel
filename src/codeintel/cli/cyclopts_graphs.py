"""Cyclopts wiring for graph commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.common_handlers import OutputFormat
from codeintel.cli.cyclopts_common import (
    OutputFormatCLI,
    OutputParam,
    resolve_output_format,
)
from codeintel.cli.graphs_handlers import (
    DependencyPolicy,
    GraphPluginsOptions,
    PlanMode,
    SelectionPolicy,
    graph_plugins_handler,
)

graphs_app = App(
    name="graph",
    help="Graph analytics plugin commands.",
)


@dataclass
class GraphPluginsCli:
    """CLI surface for graph plugin listing/planning."""

    plan: Annotated[
        bool,
        Parameter(
            name="--plan",
            help="Show planned execution order instead of a simple list.",
            negative=(),
        ),
    ] = False
    names: Annotated[
        list[str] | None,
        Parameter(
            name="--names",
            help="Explicit plugin names to plan/list (repeatable).",
        ),
    ] = None
    enable: Annotated[
        list[str] | None,
        Parameter(
            name="--enable",
            help="Ordered list of plugins to enable (overrides defaults).",
        ),
    ] = None
    disable: Annotated[
        list[str] | None,
        Parameter(
            name="--disable",
            help="Plugins to disable/filter out from the selected set.",
        ),
    ] = None
    selection_policy: Annotated[
        SelectionPolicy,
        Parameter(
            name="--selection-policy",
            help="How to handle unknown requested plugins.",
            show_default=True,
        ),
    ] = SelectionPolicy.LENIENT
    dependency_policy: Annotated[
        DependencyPolicy,
        Parameter(
            name="--dependency-policy",
            help="How to handle missing/disabled dependencies.",
            show_default=True,
        ),
    ] = DependencyPolicy.STRICT
    validate_plan: Annotated[
        bool,
        Parameter(
            name="--validate-plan",
            help="Validate plan strictly (selection/dependency strict).",
            negative=(),
        ),
    ] = False
    output: OutputParam = field(default_factory=OutputFormatCLI)


@graphs_app.command(name="plugins")
def plugins(
    cfg: Annotated[GraphPluginsCli, Parameter(name="*")] | None = None,
) -> None:
    """List registered graph plugins or display an execution plan."""
    cfg = cfg or GraphPluginsCli()
    output_format = resolve_output_format(
        json_flag=cfg.output.json,
        explicit=cfg.output.output_format,
        default=OutputFormat.TEXT,
    )

    options = GraphPluginsOptions(
        mode=PlanMode.PLAN if cfg.plan else PlanMode.LIST,
        names=tuple(cfg.names) if cfg.names else None,
        enable=tuple(cfg.enable) if cfg.enable else None,
        disable=tuple(cfg.disable) if cfg.disable else (),
        selection_policy=cfg.selection_policy,
        dependency_policy=cfg.dependency_policy,
        validation_mode=cfg.validate_plan,
        output_format=output_format,
    )
    graph_plugins_handler(options)


__all__ = ["graphs_app"]
