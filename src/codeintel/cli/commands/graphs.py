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
from textwrap import indent
from typing import Annotated

import typer

from codeintel.cli.commands._common import JsonOutputOpt
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

PlanOpt = Annotated[
    bool,
    typer.Option(
        "--plan",
        is_flag=True,
        help="Show planned execution order plus dependency graph and metadata.",
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
# Commands
# -----------------------------------------------------------------------------


@graphs_app.command("plugins")
def graph_plugins(
    plan: PlanOpt = False,
    names: NamesOpt = None,
    enable: EnableOpt = None,
    disable: DisableOpt = None,
    json_output: JsonOutputOpt = False,
) -> None:
    """List registered graph metric plugins or show execution plan.

    Shows all registered graph plugins with their metadata. Use --plan to
    see the planned execution order including dependency resolution.

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
    requested = requested_names if requested_names is not None else DEFAULT_GRAPH_PLUGINS

    if plan:
        try:
            plan_result = plan_graph_plugins(
                plugin_names=requested if enabled is None else None,
                enabled=enabled,
                disabled=disabled,
                defaults=DEFAULT_GRAPH_PLUGINS,
            )
        except ValueError:
            LOG.exception("Invalid graph plugin plan for names=%s", requested)
            raise typer.Exit(code=1) from None

        ordered = plan_result.ordered_names
        if json_output:
            payload = {
                "plan_id": plan_result.plan_id,
                "ordered_plugins": list(ordered),
                "skipped_plugins": [
                    {"name": skipped.name, "reason": skipped.reason}
                    for skipped in plan_result.skipped_plugins
                ],
                "dep_graph": {name: list(deps) for name, deps in plan_result.dep_graph.items()},
                "plugin_metadata": {
                    plugin.metadata.name: {
                        "stage": plugin.metadata.stage,
                        "severity": plugin.metadata.severity,
                        "requires_isolation": plugin.metadata.requires_isolation,
                        "isolation_kind": plugin.metadata.isolation_kind,
                        "scope_aware": plugin.metadata.scope_aware,
                        "supported_scopes": list(plugin.metadata.supported_scopes),
                        "cache_populates": list(plugin.metadata.cache_populates),
                        "cache_consumes": list(plugin.metadata.cache_consumes),
                    }
                    for plugin in plan_result.plugins
                },
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
    if json_output:
        payload = {
            "count": len(plugins),
            "plugins": {
                plugin.metadata.name: {
                    "name": plugin.metadata.name,
                    "description": plugin.metadata.description,
                    "stage": plugin.metadata.stage,
                    "severity": plugin.metadata.severity,
                    "enabled_by_default": plugin.metadata.enabled_by_default,
                    "depends_on": list(plugin.metadata.depends_on),
                    "provides": list(plugin.metadata.provides),
                    "requires": list(plugin.metadata.requires),
                    "resource_hints": (
                        {
                            "max_runtime_ms": plugin.metadata.resource_hints.max_runtime_ms,
                            "memory_mb_hint": plugin.metadata.resource_hints.memory_mb_hint,
                        }
                        if plugin.metadata.resource_hints is not None
                        else None
                    ),
                    "options_model": plugin.metadata.options_model.__name__
                    if plugin.metadata.options_model
                    else None,
                    "options_default": plugin.metadata.options_default,
                    "version_hash": plugin.metadata.version_hash,
                    "contract_checkers": len(plugin.metadata.contract_checkers),
                    "scope_aware": plugin.metadata.scope_aware,
                    "supported_scopes": list(plugin.metadata.supported_scopes),
                    "requires_isolation": plugin.metadata.requires_isolation,
                    "isolation_kind": plugin.metadata.isolation_kind,
                    "config_schema_ref": plugin.metadata.config_schema_ref,
                    "row_count_tables": list(plugin.metadata.row_count_tables),
                    "cache_populates": list(plugin.metadata.cache_populates),
                    "cache_consumes": list(plugin.metadata.cache_consumes),
                }
                for plugin in plugins
            },
        }
        sys.stdout.write(json.dumps(payload, indent=2))
        sys.stdout.write("\n")
        return

    for plugin in plugins:
        sys.stdout.write(f"- {plugin.metadata.name} [{plugin.metadata.stage}]\n")
        sys.stdout.write(indent(plugin.metadata.description, "    "))
        sys.stdout.write("\n")


__all__ = ["graphs_app"]
