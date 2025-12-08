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
from enum import Enum
from textwrap import indent
from typing import Annotated

import typer

from codeintel.cli.commands._common import JsonFlagOpt, JsonOutputOpt, OutputFormat
from codeintel.graphs.core.protocol import DEFAULT_GRAPH_PLUGINS
from codeintel.graphs.core.registry import list_graph_plugins, plan_graph_plugins

LOG = logging.getLogger(__name__)

graphs_app = typer.Typer(
    name="graph",
    help="Graph analytics plugin commands.",
    no_args_is_help=True,
)


# -----------------------------------------------------------------------------
# Option Type Aliases
# -----------------------------------------------------------------------------


class PlanMode(Enum):
    """Graph plugin listing mode."""

    LIST = "list"
    PLAN = "plan"


PlanFlagOpt = typer.Option(
    False,
    "--plan",
    help="Show planned execution order plus dependency graph and metadata.",
    is_flag=True,
)

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


def _build_plugin_metadata_map(plugins: list[object]) -> dict[str, dict[str, object]]:
    """Construct a metadata map for plugins."""
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


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------


@graphs_app.command("plugins")
def graph_plugins(
    plan: bool = PlanFlagOpt,
    names: NamesOpt = None,
    enable: EnableOpt = None,
    disable: DisableOpt = None,
    json_flag: bool = JsonFlagOpt,
    output_format: OutputFormat = JsonOutputOpt,
) -> None:
    """List registered graph metric plugins or show execution plan.

    Shows all registered graph plugins with their metadata. Use --plan to
    see the planned execution order including dependency resolution.

    Raises
    ------
    typer.Exit
        If planning fails due to invalid plugin selection.

    Examples
    --------
    .. code-block:: bash

        # List all plugins
        codeintel graph plugins

        # Show execution plan
        codeintel graph plugins --plan

        # Output as JSON with full metadata
        codeintel graph plugins --json
    """
    enabled = tuple(enable) if enable else None
    requested_names = tuple(names) if names else None
    disabled = tuple(disable) if disable else ()

    plan_requested = bool(plan)
    as_json = json_flag or output_format is OutputFormat.JSON
    registry_logger = logging.getLogger("codeintel.graphs.core.registry")
    previous_registry_level: int | None = None
    if as_json:
        previous_registry_level = registry_logger.level
        registry_logger.setLevel(logging.ERROR)

    try:
        if plan_requested:
            try:
                plan_result = plan_graph_plugins(
                    plugin_names=requested_names,
                    enabled=enabled,
                    disabled=disabled,
                    defaults=DEFAULT_GRAPH_PLUGINS,
                )
            except ValueError:
                if as_json:
                    LOG.debug("Invalid graph plugin plan for names=%s", requested_names)
                else:
                    LOG.warning("Invalid graph plugin plan for names=%s", requested_names)
                fallback_plugins = list_graph_plugins()
                metadata_map = _build_plugin_metadata_map(list(fallback_plugins))
                if as_json:
                    payload = {
                        "plan_id": "fallback",
                        "ordered_plugins": [plugin.metadata.name for plugin in fallback_plugins],
                        "skipped_plugins": [],
                        "dep_graph": {},
                        "plugin_metadata": metadata_map,
                    }
                    sys.stdout.write(json.dumps(payload, indent=2))
                    sys.stdout.write("\n")
                else:
                    typer.secho(
                        "Failed to compute plan; showing available plugins instead.",
                        fg=typer.colors.YELLOW,
                    )
                    for plugin in fallback_plugins:
                        meta = plugin.metadata
                        isolation = meta.isolation_kind or (
                            "yes" if meta.requires_isolation else "no"
                        )
                        scope_flag = "yes" if meta.scope_aware else "no"
                        sys.stdout.write(
                            f"  - {meta.name} [{meta.stage} | {meta.severity} | "
                            f"{isolation} | {scope_flag}]\n"
                        )
                return

            ordered = plan_result.ordered_names
            metadata_map = _build_plugin_metadata_map(list(plan_result.plugins))
            if as_json:
                payload = {
                    "plan_id": plan_result.plan_id,
                    "ordered_plugins": list(ordered),
                    "skipped_plugins": [
                        {"name": skipped.name, "reason": skipped.reason}
                        for skipped in plan_result.skipped_plugins
                    ],
                    "dep_graph": {name: list(deps) for name, deps in plan_result.dep_graph.items()},
                    "plugin_metadata": metadata_map,
                }
                sys.stdout.write(json.dumps(payload, indent=2))
                sys.stdout.write("\n")
            else:
                sys.stdout.write(f"Plan ID: {plan_result.plan_id}\n")
                sys.stdout.write("Execution order (stage | severity | isolation | scope-aware):\n")
                for plugin in plan_result.plugins:
                    meta = plugin.metadata
                    isolation = meta.isolation_kind or ("yes" if meta.requires_isolation else "no")
                    scope_flag = "yes" if meta.scope_aware else "no"
                    sys.stdout.write(
                        f"  - {meta.name} [{meta.stage} | {meta.severity} | "
                        f"{isolation} | {scope_flag}]\n"
                    )
                if plan_result.skipped_plugins:
                    sys.stdout.write("Skipped:\n")
                    for skipped in plan_result.skipped_plugins:
                        sys.stdout.write(f"  - {skipped.name} ({skipped.reason})\n")
            return

        plugins = list_graph_plugins()
        metadata_map = _build_plugin_metadata_map(list(plugins))
        if as_json:
            payload = {
                "count": len(plugins),
                "plugins": metadata_map,
            }
            sys.stdout.write(json.dumps(payload, indent=2))
            sys.stdout.write("\n")
            return

        for plugin in plugins:
            sys.stdout.write(f"- {plugin.metadata.name} [{plugin.metadata.stage}]\n")
            sys.stdout.write(indent(plugin.metadata.description, "    "))
            sys.stdout.write("\n")
    finally:
        if previous_registry_level is not None:
            registry_logger.setLevel(previous_registry_level)


__all__ = ["graphs_app"]
