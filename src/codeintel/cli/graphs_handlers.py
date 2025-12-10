"""Typer-free handlers for graph analytics plugin commands.

These helpers keep operational logic while allowing Cyclopts to invoke
them without importing Typer. All user-facing errors surface as
:class:`~codeintel.cli.cli_errors.ValidationError`.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from textwrap import indent
from typing import TYPE_CHECKING, cast

from codeintel.cli.cli_types import OutputFormat
from codeintel.cli.result_types import GraphPlanResult, GraphPluginInfo, GraphPluginsResult
from codeintel.cli.results import CliResult
from codeintel.graphs.core.protocol import (
    DEFAULT_GRAPH_PLUGINS,
    GraphPluginPlan,
    GraphPluginProtocol,
)
from codeintel.graphs.core.registry import (
    DependencyPolicy,
    PlanningOptions,
    SelectionPolicy,
    list_graph_plugins,
    plan_graph_plugins,
)

if TYPE_CHECKING:
    from codeintel.cli.execution.context import ExecutionContext

LOG = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------


class PlanMode(Enum):
    """Graph plugin listing mode."""

    LIST = "list"
    PLAN = "plan"


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphPluginsOptions:
    """Options for graph plugin listing and planning."""

    mode: PlanMode
    names: tuple[str, ...] | None
    enable: tuple[str, ...] | None
    disable: tuple[str, ...]
    selection_policy: SelectionPolicy
    dependency_policy: DependencyPolicy
    validation_mode: bool
    output_format: OutputFormat


@dataclass(frozen=True)
class ParsedOptions:
    """Parsed CLI options for graph plugin operations."""

    plan: bool
    names: tuple[str, ...] | None
    enable: tuple[str, ...] | None
    disable: tuple[str, ...]
    selection_policy: SelectionPolicy
    dependency_policy: DependencyPolicy
    validation_mode: bool
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
    """Configure registry logger for output format.

    Returns
    -------
    tuple[logging.Logger, int | None]
        Logger and previous level.
    """
    registry_logger = logging.getLogger("codeintel.graphs.core.registry")
    previous_registry_level: int | None = None
    if output_format is OutputFormat.JSON:
        previous_registry_level = registry_logger.level
        registry_logger.setLevel(logging.ERROR)
    return registry_logger, previous_registry_level


def _restore_registry_logger(logger: logging.Logger, previous_level: int | None) -> None:
    """Restore registry logger level.

    Parameters
    ----------
    logger
        Logger to restore.
    previous_level
        Previous level to restore.
    """
    if previous_level is not None:
        logger.setLevel(previous_level)


def _render_plan_json(plan_result: GraphPluginPlan) -> None:
    """Render plan as JSON.

    Parameters
    ----------
    plan_result
        Plan to render.
    """
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
    """Render plan as text.

    Parameters
    ----------
    plan_result
        Plan to render.
    """
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
    """Render fallback plugin list when plan fails.

    Parameters
    ----------
    output_format
        Output format.
    """
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

    sys.stderr.write("Failed to compute plan; showing available plugins instead.\n")
    _render_list_text(fallback_plugins)


def _plan_plugins(options: GraphPluginsOptions) -> GraphPluginPlan | None:
    """Plan plugin execution order.

    Parameters
    ----------
    options
        Plugin options.

    Returns
    -------
    GraphPluginPlan | None
        Plan or None if planning fails.
    """
    if options.validation_mode:
        plan_opts = PlanningOptions(
            selection_policy=SelectionPolicy.STRICT,
            dependency_policy=DependencyPolicy.STRICT,
            use_stubs=False,
            allow_missing_dependencies=False,
        )
    else:
        plan_opts = PlanningOptions(
            selection_policy=options.selection_policy,
            dependency_policy=options.dependency_policy,
            requested_required=False,
        )
    try:
        return plan_graph_plugins(
            plugin_names=options.names,
            enabled=options.enable,
            disabled=options.disable,
            defaults=DEFAULT_GRAPH_PLUGINS,
            plan_options=plan_opts,
        )
    except ValueError:
        LOG.debug("Invalid graph plugin plan for names=%s", options.names)
        return None


def _render_plan(plan_result: GraphPluginPlan, output_format: OutputFormat) -> None:
    """Render plan in requested format.

    Parameters
    ----------
    plan_result
        Plan to render.
    output_format
        Output format.
    """
    if output_format is OutputFormat.JSON:
        _render_plan_json(plan_result)
        return

    _render_plan_text(plan_result)


def _render_list_json(plugins: Iterable[GraphPluginProtocol]) -> None:
    """Render plugin list as JSON.

    Parameters
    ----------
    plugins
        Plugins to render.
    """
    plugin_list = list(plugins)
    metadata_map = _build_plugin_metadata_map(plugin_list)
    payload = {
        "count": len(plugin_list),
        "plugins": metadata_map,
    }
    sys.stdout.write(json.dumps(payload, indent=2))
    sys.stdout.write("\n")


def _render_list_text(plugins: Iterable[GraphPluginProtocol]) -> None:
    """Render plugin list as text.

    Parameters
    ----------
    plugins
        Plugins to render.
    """
    for plugin in plugins:
        sys.stdout.write(f"- {plugin.metadata.name} [{plugin.metadata.stage}]\n")
        sys.stdout.write(indent(plugin.metadata.description, "    "))
        sys.stdout.write("\n")


def _to_tuple(value: object | None) -> tuple[str, ...] | None:
    """Convert list to tuple or return None.

    Parameters
    ----------
    value
        Value to convert.

    Returns
    -------
    tuple[str, ...] | None
        Converted tuple.
    """
    if value is None:
        return None
    return tuple(cast("list[str]", value))


# -----------------------------------------------------------------------------
# Bundle Function
# -----------------------------------------------------------------------------


def parse_graph_options(cli_kwargs: Mapping[str, object]) -> ParsedOptions:
    """Parse raw CLI kwargs into typed options.

    Parameters
    ----------
    cli_kwargs
        Raw CLI keyword arguments.

    Returns
    -------
    ParsedOptions
        Parsed options.
    """
    output_format = cast("OutputFormat", cli_kwargs.get("output_format", OutputFormat.TEXT))
    if bool(cli_kwargs.get("json", False)):
        output_format = OutputFormat.JSON
    return ParsedOptions(
        plan=bool(cli_kwargs.get("mode", False)),
        names=_to_tuple(cli_kwargs.get("names")),
        enable=_to_tuple(cli_kwargs.get("enable")),
        disable=_to_tuple(cli_kwargs.get("disable")) or (),
        selection_policy=SelectionPolicy(
            cast(
                "str | SelectionPolicy", cli_kwargs.get("selection_policy", SelectionPolicy.LENIENT)
            )
        ),
        dependency_policy=DependencyPolicy(
            cast(
                "str | DependencyPolicy",
                cli_kwargs.get("dependency_policy", DependencyPolicy.STRICT),
            )
        ),
        validation_mode=bool(cli_kwargs.get("validate_plan", False)),
        output_format=output_format,
    )


def bundle_graph_plugins(cli_kwargs: Mapping[str, object]) -> Mapping[str, object]:
    """Bundle CLI arguments for graph plugins command.

    Parameters
    ----------
    cli_kwargs
        Raw CLI keyword arguments.

    Returns
    -------
    Mapping[str, object]
        Bundled arguments.
    """
    parsed = parse_graph_options(cli_kwargs)
    return {
        "options": GraphPluginsOptions(
            mode=PlanMode.PLAN if parsed.plan else PlanMode.LIST,
            names=parsed.names,
            enable=parsed.enable,
            disable=parsed.disable,
            selection_policy=parsed.selection_policy,
            dependency_policy=parsed.dependency_policy,
            validation_mode=parsed.validation_mode,
            output_format=parsed.output_format,
        )
    }


# -----------------------------------------------------------------------------
# Structured Handlers (return CliResult instead of printing)
# -----------------------------------------------------------------------------


def graph_plugins_handler_structured(
    options: GraphPluginsOptions,
) -> CliResult[GraphPluginsResult | GraphPlanResult]:
    """List registered graph plugins or display an execution plan (structured version).

    Parameters
    ----------
    options
        Graph plugin options.

    Returns
    -------
    CliResult[GraphPluginsResult | GraphPlanResult]
        Structured result with plugin list or execution plan.
    """
    registry_logger, previous_registry_level = _configure_registry_logger(options.output_format)

    try:
        if options.mode is PlanMode.PLAN:
            plan_result = _plan_plugins(options)
            if plan_result is None:
                # Return empty plan
                return CliResult.ok(
                    GraphPlanResult(
                        plan_id="empty",
                        plugins=[],
                        skipped=[],
                    )
                )

            # Convert plan to structured result
            plugins = [
                GraphPluginInfo(
                    name=p.metadata.name,
                    stage=p.metadata.stage,
                    output_tables=p.metadata.produces_tables,
                    enabled=True,
                )
                for p in plan_result.plugins
            ]
            skipped = [{"name": s.name, "reason": s.reason} for s in plan_result.skipped_plugins]
            return CliResult.ok(
                GraphPlanResult(
                    plan_id=plan_result.plan_id,
                    plugins=plugins,
                    skipped=skipped,
                )
            )

        # List mode
        all_plugins = list_graph_plugins()
        plugin_list: list[dict[str, object]] = [
            {
                "name": plugin.metadata.name,
                "stage": plugin.metadata.stage,
                "output_tables": list(plugin.metadata.produces_tables),
            }
            for plugin in all_plugins
        ]

        return CliResult.ok(
            GraphPluginsResult(
                plugins=plugin_list,
                count=len(plugin_list),
            )
        )
    finally:
        _restore_registry_logger(registry_logger, previous_registry_level)


# -----------------------------------------------------------------------------
# ExecutionContext-based Handler
# -----------------------------------------------------------------------------


def _build_graph_options_from_ctx(ctx: ExecutionContext) -> GraphPluginsOptions:
    """Build GraphPluginsOptions from ExecutionContext.

    Parameters
    ----------
    ctx
        Execution context.

    Returns
    -------
    GraphPluginsOptions
        Options for graph plugin operations.
    """
    output_format = ctx.output_format
    if ctx.get_bool_param("json"):
        output_format = OutputFormat.JSON

    names_raw = ctx.params.get("names")
    enable_raw = ctx.params.get("enable")
    disable_raw = ctx.params.get("disable")

    selection_policy_raw = ctx.params.get("selection_policy", SelectionPolicy.LENIENT)
    if isinstance(selection_policy_raw, SelectionPolicy):
        selection_policy = selection_policy_raw
    else:
        selection_policy = SelectionPolicy(str(selection_policy_raw))

    dependency_policy_raw = ctx.params.get("dependency_policy", DependencyPolicy.STRICT)
    if isinstance(dependency_policy_raw, DependencyPolicy):
        dependency_policy = dependency_policy_raw
    else:
        dependency_policy = DependencyPolicy(str(dependency_policy_raw))

    return GraphPluginsOptions(
        mode=PlanMode.PLAN if ctx.get_bool_param("plan") else PlanMode.LIST,
        names=tuple(names_raw) if names_raw else None,
        enable=tuple(enable_raw) if enable_raw else None,
        disable=tuple(disable_raw) if disable_raw else (),
        selection_policy=selection_policy,
        dependency_policy=dependency_policy,
        validation_mode=ctx.get_bool_param("validate_plan"),
        output_format=output_format,
    )


def graph_plugins_ctx(ctx: ExecutionContext) -> CliResult[GraphPluginsResult | GraphPlanResult]:
    """List registered graph plugins or display an execution plan.

    Parameters
    ----------
    ctx
        Execution context with params:
        - plan: Whether to show execution plan.
        - names: Specific plugin names to include.
        - enable: Plugins to enable.
        - disable: Plugins to disable.
        - selection_policy: Selection policy for plugins.
        - dependency_policy: Dependency resolution policy.
        - validate_plan: Whether to use strict validation.
        - json: Output in JSON format.

    Returns
    -------
    CliResult[GraphPluginsResult | GraphPlanResult]
        Structured result with plugin list or execution plan.
    """
    options = _build_graph_options_from_ctx(ctx)
    return graph_plugins_handler_structured(options)


__all__ = [
    "DependencyPolicy",
    "GraphPluginsOptions",
    "OutputFormat",
    "ParsedOptions",
    "PlanMode",
    "SelectionPolicy",
    "bundle_graph_plugins",
    "graph_plugins_ctx",
    "graph_plugins_handler_structured",
    "parse_graph_options",
]
