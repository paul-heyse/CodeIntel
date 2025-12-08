"""Graph analytics plugin commands for the CodeIntel CLI.

This module provides Typer commands for managing and introspecting graph
analytics plugins.

Commands
--------
- **plugins**: List graph metric plugins with metadata
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from textwrap import indent
from typing import Annotated, cast

import typer

from codeintel.cli.commands._common import JsonFlagOpt, JsonOutputOpt, OutputFormat
from codeintel.cli.commands._option_shim import OptionSpec, wrap_command
from codeintel.graphs.core.protocol import (
    DEFAULT_GRAPH_PLUGINS,
    GraphPluginPlan,
    GraphPluginProtocol,
)
from codeintel.graphs.core.registry import list_graph_plugins, plan_graph_plugins

LOG = logging.getLogger(__name__)

graphs_app = typer.Typer(
    name="graph",
    help="Graph analytics plugin commands.",
    no_args_is_help=True,
)


# -----------------------------------------------------------------------------
# Option Types
# -----------------------------------------------------------------------------


class PlanMode(Enum):
    """Graph plugin listing mode."""

    LIST = "list"
    PLAN = "plan"


PlanModeFlagOpt = Annotated[
    bool,
    typer.Option(
        "--plan",
        help="Show planned execution order plus dependency graph and metadata.",
        is_flag=True,
    ),
]

NamesOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--names",
        help="Explicit plugin names to plan/list (defaults to built-in defaults).",
    ),
]

EnableOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--enable",
        help="Ordered list of plugins to enable (overrides defaults when provided).",
    ),
]

DisableOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--disable",
        help="Plugins to disable/filter out from the selected set.",
    ),
]


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphPluginsOptions:
    """Options for graph plugin listing and planning."""

    mode: PlanMode
    names: tuple[str, ...] | None
    enable: tuple[str, ...] | None
    disable: tuple[str, ...]
    output_format: OutputFormat


@dataclass(frozen=True)
class ParsedOptions:
    """Parsed CLI options for graph plugin operations."""

    plan: bool
    names: tuple[str, ...] | None
    enable: tuple[str, ...] | None
    disable: tuple[str, ...]
    output_format: OutputFormat


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _build_plugin_metadata_map(
    plugins: Iterable[GraphPluginProtocol],
) -> dict[str, dict[str, object]]:
    """Construct a metadata map for plugins.

    Returns
    -------
    dict[str, dict[str, object]]
        Metadata keyed by plugin name.
    """
    metadata: dict[str, dict[str, object]] = {}
    for plugin in plugins:
        meta = plugin.metadata
        metadata[meta.name] = {
            "name": meta.name,
            "description": meta.description,
            "stage": meta.stage,
            "severity": meta.severity,
            "enabled_by_default": meta.enabled_by_default,
            "depends_on": list(meta.depends_on),
            "provides": list(meta.provides),
            "requires": list(meta.requires),
            "resource_hints": (
                {
                    "max_runtime_ms": meta.resource_hints.max_runtime_ms,
                    "max_memory_mb": meta.resource_hints.max_memory_mb,
                }
                if meta.resource_hints is not None
                else None
            ),
            "options_model": meta.options_model.__name__ if meta.options_model else None,
            "options_default": meta.options_default,
            "version_hash": meta.version_hash,
            "contract_checkers": len(meta.contract_checkers),
            "scope_aware": meta.scope_aware,
            "supported_scopes": list(meta.supported_scopes),
            "requires_isolation": meta.requires_isolation,
            "isolation_kind": meta.isolation_kind,
            "config_schema_ref": meta.config_schema_ref,
            "row_count_tables": list(meta.row_count_tables),
            "cache_populates": list(meta.cache_populates),
            "cache_consumes": list(meta.cache_consumes),
        }
    return metadata


def _configure_registry_logger(output_format: OutputFormat) -> tuple[logging.Logger, int | None]:
    registry_logger = logging.getLogger("codeintel.graphs.core.registry")
    previous_registry_level: int | None = None
    if output_format is OutputFormat.JSON:
        previous_registry_level = registry_logger.level
        registry_logger.setLevel(logging.ERROR)
    return registry_logger, previous_registry_level


def _restore_registry_logger(logger: logging.Logger, previous_level: int | None) -> None:
    if previous_level is not None:
        logger.setLevel(previous_level)


def _render_plan_json(plan_result: GraphPluginPlan) -> None:
    metadata_map = _build_plugin_metadata_map(list(plan_result.plugins))
    payload = {
        "plan_id": plan_result.plan_id,
        "ordered_plugins": list(plan_result.ordered_names),
        "skipped_plugins": [
            {"name": skipped.name, "reason": skipped.reason}
            for skipped in plan_result.skipped_plugins
        ],
        "dep_graph": {name: list(deps) for name, deps in plan_result.dep_graph.items()},
        "plugin_metadata": metadata_map,
    }
    sys.stdout.write(json.dumps(payload, indent=2))
    sys.stdout.write("\n")


def _render_plan_text(plan_result: GraphPluginPlan) -> None:
    sys.stdout.write(f"Plan ID: {plan_result.plan_id}\n")
    sys.stdout.write("Execution order (stage | severity | isolation | scope-aware):\n")
    for plugin in plan_result.plugins:
        meta = plugin.metadata
        isolation = meta.isolation_kind or ("yes" if meta.requires_isolation else "no")
        scope_flag = "yes" if meta.scope_aware else "no"
        sys.stdout.write(
            f"  - {meta.name} [{meta.stage} | {meta.severity} | {isolation} | {scope_flag}]\n"
        )
    if plan_result.skipped_plugins:
        sys.stdout.write("Skipped:\n")
        for skipped in plan_result.skipped_plugins:
            sys.stdout.write(f"  - {skipped.name} ({skipped.reason})\n")


def _render_fallback_plan(output_format: OutputFormat) -> None:
    fallback_plugins = list_graph_plugins()
    metadata_map = _build_plugin_metadata_map(list(fallback_plugins))
    if output_format is OutputFormat.JSON:
        payload = {
            "plan_id": "fallback",
            "ordered_plugins": [plugin.metadata.name for plugin in fallback_plugins],
            "skipped_plugins": [],
            "dep_graph": {},
            "plugin_metadata": metadata_map,
        }
        sys.stdout.write(json.dumps(payload, indent=2))
        sys.stdout.write("\n")
        return

    typer.secho(
        "Failed to compute plan; showing available plugins instead.",
        fg=typer.colors.YELLOW,
    )
    _render_list_text(fallback_plugins)


def _plan_plugins(options: GraphPluginsOptions) -> GraphPluginPlan | None:
    try:
        return plan_graph_plugins(
            plugin_names=options.names,
            enabled=options.enable,
            disabled=options.disable,
            defaults=DEFAULT_GRAPH_PLUGINS,
        )
    except ValueError:
        LOG.debug("Invalid graph plugin plan for names=%s", options.names)
        return None


def _render_plan(plan_result: GraphPluginPlan, output_format: OutputFormat) -> None:
    if output_format is OutputFormat.JSON:
        _render_plan_json(plan_result)
        return

    _render_plan_text(plan_result)


def _render_list_json(plugins: Iterable[GraphPluginProtocol]) -> None:
    plugin_list = list(plugins)
    metadata_map = _build_plugin_metadata_map(plugin_list)
    payload = {
        "count": len(plugin_list),
        "plugins": metadata_map,
    }
    sys.stdout.write(json.dumps(payload, indent=2))
    sys.stdout.write("\n")


def _render_list_text(plugins: Iterable[GraphPluginProtocol]) -> None:
    for plugin in plugins:
        sys.stdout.write(f"- {plugin.metadata.name} [{plugin.metadata.stage}]\n")
        sys.stdout.write(indent(plugin.metadata.description, "    "))
        sys.stdout.write("\n")


# -----------------------------------------------------------------------------
# Command
# -----------------------------------------------------------------------------


def graph_plugins_handler(options: GraphPluginsOptions) -> None:
    """List registered graph plugins or display an execution plan."""
    registry_logger, previous_registry_level = _configure_registry_logger(options.output_format)

    try:
        if options.mode is PlanMode.PLAN:
            plan_result = _plan_plugins(options)
            if plan_result is None:
                _render_fallback_plan(options.output_format)
                return
            _render_plan(plan_result, options.output_format)
            return

        plugins = list_graph_plugins()
        if options.output_format is OutputFormat.JSON:
            _render_list_json(plugins)
        else:
            _render_list_text(plugins)
    finally:
        _restore_registry_logger(registry_logger, previous_registry_level)


def _to_tuple(value: object | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(cast("list[str]", value))


def parse_graph_options(cli_kwargs: Mapping[str, object]) -> ParsedOptions:
    output_format = cast("OutputFormat", cli_kwargs.get("output_format", OutputFormat.TEXT))
    if bool(cli_kwargs.get("json", False)):
        output_format = OutputFormat.JSON
    return ParsedOptions(
        plan=bool(cli_kwargs.get("mode", False)),
        names=_to_tuple(cli_kwargs.get("names")),
        enable=_to_tuple(cli_kwargs.get("enable")),
        disable=_to_tuple(cli_kwargs.get("disable")) or (),
        output_format=output_format,
    )


def _bundle_graph_plugins(cli_kwargs: Mapping[str, object]) -> Mapping[str, object]:
    parsed = parse_graph_options(cli_kwargs)
    return {
        "options": GraphPluginsOptions(
            mode=PlanMode.PLAN if parsed.plan else PlanMode.LIST,
            names=parsed.names,
            enable=parsed.enable,
            disable=parsed.disable,
            output_format=parsed.output_format,
        )
    }


_GRAPH_PLUGIN_SPECS = [
    OptionSpec("mode", PlanModeFlagOpt, default=False),
    OptionSpec("names", NamesOpt, None),
    OptionSpec("enable", EnableOpt, None),
    OptionSpec("disable", DisableOpt, None),
    OptionSpec("json", JsonFlagOpt, default=False),
    OptionSpec("output_format", OutputFormat, JsonOutputOpt),
]

graphs_app.command("plugins")(
    wrap_command(
        graph_plugins_handler,
        _GRAPH_PLUGIN_SPECS,
        bundle=_bundle_graph_plugins,
        name="graph_plugins",
    )
)


__all__ = ["graphs_app"]
