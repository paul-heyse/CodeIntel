"""Build system commands for minimal-work target computation.

This module wires Cyclopts command classes to unified handlers via @cli_command.

Note: Build commands require complex runtime/gateway/snapshot access that is not
yet fully supported by the Command[T] pattern's Deps abstraction. They use
the handler pattern for now.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from cyclopts import App

from codeintel.cli.commands.build_schema import build_schema_app
from codeintel.cli.commands.build_spec import build_spec_app
from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.build import (
    build_assets_handler,
    build_decision_trace_handler,
    build_diff_handler,
    build_explain_handler,
    build_graph_handler,
    build_history_handler,
    build_impact_handler,
    build_lineage_handler,
    build_plan_handler,
    build_promote_handler,
    build_resolve_handler,
    build_run_handler,
    build_status_handler,
)
from codeintel.cli.handlers.build_validate import build_validate_handler
from codeintel.cli.options.registry import (
    BUILD_ASSETS_ASSET,
    BUILD_ASSETS_FORMAT,
    BUILD_ASSETS_TARGET,
    BUILD_ASSETS_TYPE,
    BUILD_DECISION_TRACE_OUTPUT,
    BUILD_DECISION_TRACE_PATH,
    BUILD_DIFF_ASSET,
    BUILD_DIFF_FORMAT,
    BUILD_DIFF_FROM,
    BUILD_DIFF_TO,
    BUILD_EXPLAIN_FORCE,
    BUILD_EXPLAIN_IO_SURFACE,
    BUILD_EXPLAIN_TARGET,
    BUILD_GRAPH_ALL,
    BUILD_GRAPH_FORMAT,
    BUILD_GRAPH_MODULE,
    BUILD_GRAPH_OUTPUT,
    BUILD_GRAPH_TARGETS,
    BUILD_HISTORY_LIMIT,
    BUILD_HISTORY_RUN_ID,
    BUILD_IMPACT_ASSET_KEY,
    BUILD_IMPACT_ASSET_KIND,
    BUILD_IMPACT_FORMAT,
    BUILD_IMPACT_MAX_DEPTH,
    BUILD_IMPACT_SHOW_TARGETS,
    BUILD_IMPACT_VERSION_HASH,
    BUILD_LINEAGE_ASSET,
    BUILD_LINEAGE_DEPTH,
    BUILD_LINEAGE_DIRECTION,
    BUILD_LINEAGE_FORMAT,
    BUILD_PLAN_ALL,
    BUILD_PLAN_FORCE,
    BUILD_PLAN_MODULE,
    BUILD_PLAN_OUTPUT,
    BUILD_PLAN_TARGETS,
    BUILD_PROMOTE_ALIAS,
    BUILD_PROMOTE_ASSET,
    BUILD_PROMOTE_FORMAT,
    BUILD_PROMOTE_FROM_RUN,
    BUILD_PROMOTE_NOTE,
    BUILD_PROMOTE_VERSION_HASH,
    BUILD_RESOLVE_ALIAS,
    BUILD_RESOLVE_ASSET,
    BUILD_RESOLVE_FORMAT,
    BUILD_RUN_ALL_TARGETS,
    BUILD_RUN_CACHE_DIR,
    BUILD_RUN_CACHE_REPORT,
    BUILD_RUN_CLEAR_CACHE,
    BUILD_RUN_DRY_RUN,
    BUILD_RUN_ENABLE_CACHE,
    BUILD_RUN_FORCE,
    BUILD_RUN_MAX_WORKERS,
    BUILD_RUN_MODULE,
    BUILD_RUN_PARALLEL_BACKEND,
    BUILD_RUN_PROGRESS,
    BUILD_RUN_PUBLISH_SNAPSHOT,
    BUILD_RUN_STRICT_CONTRACTS,
    BUILD_RUN_TARGETS,
    BUILD_RUN_VALIDATE_OUTPUTS,
    BUILD_STATUS_MODULE,
    BUILD_VALIDATE_FORMAT,
)
from codeintel.cli.options.shared_flags import SharedFlags, shared_flags_field
from codeintel.cli.options.types import CommandPath, option_param

build_app = App(
    name="build",
    help="Build system commands for minimal-work target computation.",
)
build_app.command(build_schema_app, name="schema")
build_app.command(build_spec_app, name="spec")


_BUILD_CONFIG = CommandConfig(require_runtime=True, require_gateway=True)
_VALIDATE_CONFIG = CommandConfig(require_runtime=False, require_gateway=False)
_TRACE_CONFIG = CommandConfig(require_runtime=True, require_gateway=False)

BUILD_RUN_PATH: CommandPath = ("build", "run")
BUILD_STATUS_PATH: CommandPath = ("build", "status")
BUILD_HISTORY_PATH: CommandPath = ("build", "history")
BUILD_VALIDATE_PATH: CommandPath = ("build", "validate")
BUILD_PLAN_PATH: CommandPath = ("build", "plan")
BUILD_EXPLAIN_PATH: CommandPath = ("build", "explain")
BUILD_GRAPH_PATH: CommandPath = ("build", "graph")
BUILD_ASSETS_PATH: CommandPath = ("build", "assets")
BUILD_LINEAGE_PATH: CommandPath = ("build", "lineage")
BUILD_PROMOTE_PATH: CommandPath = ("build", "promote")
BUILD_RESOLVE_PATH: CommandPath = ("build", "resolve")
BUILD_DIFF_PATH: CommandPath = ("build", "diff")
BUILD_IMPACT_PATH: CommandPath = ("build", "impact")
BUILD_DECISION_TRACE_CMD_PATH: CommandPath = ("build", "decision-trace")


@cli_command("build.run", handler=build_run_handler, config=_BUILD_CONFIG)
@build_app.command(name="run")
@dataclass
class BuildRunCommand:
    """Build targets with automatic dependency resolution."""

    targets: Annotated[
        list[str] | None,
        option_param(BUILD_RUN_TARGETS, command_path=BUILD_RUN_PATH),
    ] = None
    module: Annotated[
        str | None,
        option_param(BUILD_RUN_MODULE, command_path=BUILD_RUN_PATH),
    ] = None
    all_targets: Annotated[
        bool,
        option_param(BUILD_RUN_ALL_TARGETS, command_path=BUILD_RUN_PATH),
    ] = False
    dry_run: Annotated[
        bool,
        option_param(BUILD_RUN_DRY_RUN, command_path=BUILD_RUN_PATH),
    ] = False
    force: Annotated[
        list[str] | None,
        option_param(BUILD_RUN_FORCE, command_path=BUILD_RUN_PATH),
    ] = None
    validate_outputs: Annotated[
        bool,
        option_param(BUILD_RUN_VALIDATE_OUTPUTS, command_path=BUILD_RUN_PATH),
    ] = False
    strict_contracts: Annotated[
        bool,
        option_param(BUILD_RUN_STRICT_CONTRACTS, command_path=BUILD_RUN_PATH),
    ] = False
    publish_serving_snapshot: Annotated[
        bool,
        option_param(BUILD_RUN_PUBLISH_SNAPSHOT, command_path=BUILD_RUN_PATH),
    ] = False
    parallel_backend: Annotated[
        str,
        option_param(BUILD_RUN_PARALLEL_BACKEND, command_path=BUILD_RUN_PATH),
    ] = "sequential"
    max_workers: Annotated[
        int | None,
        option_param(BUILD_RUN_MAX_WORKERS, command_path=BUILD_RUN_PATH),
    ] = None
    enable_cache: Annotated[
        bool,
        option_param(BUILD_RUN_ENABLE_CACHE, command_path=BUILD_RUN_PATH),
    ] = True
    cache_dir: Annotated[
        str | None,
        option_param(BUILD_RUN_CACHE_DIR, command_path=BUILD_RUN_PATH),
    ] = None
    clear_cache: Annotated[
        bool,
        option_param(BUILD_RUN_CLEAR_CACHE, command_path=BUILD_RUN_PATH),
    ] = False
    cache_report: Annotated[
        bool,
        option_param(BUILD_RUN_CACHE_REPORT, command_path=BUILD_RUN_PATH),
    ] = False
    enable_progress: Annotated[
        bool,
        option_param(BUILD_RUN_PROGRESS, command_path=BUILD_RUN_PATH),
    ] = False
    flags: SharedFlags = shared_flags_field(BUILD_RUN_PATH)


@cli_command("build.status", handler=build_status_handler, config=_BUILD_CONFIG)
@build_app.command(name="status")
@dataclass
class BuildStatusCommand:
    """Show current state of build targets."""

    module: Annotated[
        str | None,
        option_param(BUILD_STATUS_MODULE, command_path=BUILD_STATUS_PATH),
    ] = None
    flags: SharedFlags = shared_flags_field(BUILD_STATUS_PATH)


@cli_command("build.history", handler=build_history_handler, config=_BUILD_CONFIG)
@build_app.command(name="history")
@dataclass
class BuildHistoryCommand:
    """Show build run history and details."""

    run_id: Annotated[
        str | None,
        option_param(BUILD_HISTORY_RUN_ID, command_path=BUILD_HISTORY_PATH),
    ] = None
    limit: Annotated[
        int,
        option_param(BUILD_HISTORY_LIMIT, command_path=BUILD_HISTORY_PATH),
    ] = 10
    flags: SharedFlags = shared_flags_field(BUILD_HISTORY_PATH)


@cli_command("build.validate", handler=build_validate_handler, config=_VALIDATE_CONFIG)
@build_app.command(name="validate")
@dataclass
class BuildValidateCommand:
    """Validate Hamilton DAG invariants for DAG-first planning."""

    output_format: Annotated[
        str,
        option_param(BUILD_VALIDATE_FORMAT, command_path=BUILD_VALIDATE_PATH),
    ] = "json"
    flags: SharedFlags = shared_flags_field(BUILD_VALIDATE_PATH)


@cli_command("build.plan", handler=build_plan_handler, config=_BUILD_CONFIG)
@build_app.command(name="plan")
@dataclass
class BuildPlanCommand:
    """Show build plan with status and reason for each target."""

    targets: Annotated[
        list[str] | None,
        option_param(BUILD_PLAN_TARGETS, command_path=BUILD_PLAN_PATH),
    ] = None
    module: Annotated[
        str | None,
        option_param(BUILD_PLAN_MODULE, command_path=BUILD_PLAN_PATH),
    ] = None
    all_targets: Annotated[
        bool,
        option_param(BUILD_PLAN_ALL, command_path=BUILD_PLAN_PATH),
    ] = False
    force: Annotated[
        list[str] | None,
        option_param(BUILD_PLAN_FORCE, command_path=BUILD_PLAN_PATH),
    ] = None
    output_file: Annotated[
        str | None,
        option_param(BUILD_PLAN_OUTPUT, command_path=BUILD_PLAN_PATH),
    ] = None
    flags: SharedFlags = shared_flags_field(BUILD_PLAN_PATH)


@cli_command("build.explain", handler=build_explain_handler, config=_BUILD_CONFIG)
@build_app.command(name="explain")
@dataclass
class BuildExplainCommand:
    """Explain why a target is stale and what dependencies changed."""

    target: Annotated[
        str,
        option_param(BUILD_EXPLAIN_TARGET, command_path=BUILD_EXPLAIN_PATH),
    ]
    force: Annotated[
        list[str] | None,
        option_param(BUILD_EXPLAIN_FORCE, command_path=BUILD_EXPLAIN_PATH),
    ] = None
    io_surface: Annotated[
        bool,
        option_param(BUILD_EXPLAIN_IO_SURFACE, command_path=BUILD_EXPLAIN_PATH),
    ] = False
    flags: SharedFlags = shared_flags_field(BUILD_EXPLAIN_PATH)


@cli_command("build.graph", handler=build_graph_handler, config=_BUILD_CONFIG)
@build_app.command(name="graph")
@dataclass
class BuildGraphCommand:
    """Export Hamilton DAG for specified targets."""

    targets: Annotated[
        list[str] | None,
        option_param(BUILD_GRAPH_TARGETS, command_path=BUILD_GRAPH_PATH),
    ] = None
    module: Annotated[
        str | None,
        option_param(BUILD_GRAPH_MODULE, command_path=BUILD_GRAPH_PATH),
    ] = None
    all_targets: Annotated[
        bool,
        option_param(BUILD_GRAPH_ALL, command_path=BUILD_GRAPH_PATH),
    ] = False
    output_format: Annotated[
        str,
        option_param(BUILD_GRAPH_FORMAT, command_path=BUILD_GRAPH_PATH),
    ] = "json"
    output_file: Annotated[
        str | None,
        option_param(BUILD_GRAPH_OUTPUT, command_path=BUILD_GRAPH_PATH),
    ] = None
    flags: SharedFlags = shared_flags_field(BUILD_GRAPH_PATH)


@cli_command("build.assets", handler=build_assets_handler, config=_BUILD_CONFIG)
@build_app.command(name="assets")
@dataclass
class BuildAssetsCommand:
    """List materialized assets for the current snapshot."""

    asset: Annotated[
        str | None,
        option_param(BUILD_ASSETS_ASSET, command_path=BUILD_ASSETS_PATH),
    ] = None
    target: Annotated[
        str | None,
        option_param(BUILD_ASSETS_TARGET, command_path=BUILD_ASSETS_PATH),
    ] = None
    asset_type: Annotated[
        str | None,
        option_param(BUILD_ASSETS_TYPE, command_path=BUILD_ASSETS_PATH),
    ] = None
    output_format: Annotated[
        str,
        option_param(BUILD_ASSETS_FORMAT, command_path=BUILD_ASSETS_PATH),
    ] = "table"
    flags: SharedFlags = shared_flags_field(BUILD_ASSETS_PATH)


@cli_command("build.lineage", handler=build_lineage_handler, config=_BUILD_CONFIG)
@build_app.command(name="lineage")
@dataclass
class BuildLineageCommand:
    """Show asset lineage for the current snapshot."""

    asset: Annotated[
        str,
        option_param(BUILD_LINEAGE_ASSET, command_path=BUILD_LINEAGE_PATH),
    ]
    direction: Annotated[
        str,
        option_param(BUILD_LINEAGE_DIRECTION, command_path=BUILD_LINEAGE_PATH),
    ] = "up"
    depth: Annotated[
        int,
        option_param(BUILD_LINEAGE_DEPTH, command_path=BUILD_LINEAGE_PATH),
    ] = 1
    output_format: Annotated[
        str,
        option_param(BUILD_LINEAGE_FORMAT, command_path=BUILD_LINEAGE_PATH),
    ] = "json"
    flags: SharedFlags = shared_flags_field(BUILD_LINEAGE_PATH)


@cli_command("build.promote", handler=build_promote_handler, config=_BUILD_CONFIG)
@build_app.command(name="promote")
@dataclass
class BuildPromoteCommand:
    """Set an alias for an asset version."""

    asset: Annotated[
        str,
        option_param(BUILD_PROMOTE_ASSET, command_path=BUILD_PROMOTE_PATH),
    ]
    alias: Annotated[
        str,
        option_param(BUILD_PROMOTE_ALIAS, command_path=BUILD_PROMOTE_PATH),
    ]
    version_hash: Annotated[
        str | None,
        option_param(BUILD_PROMOTE_VERSION_HASH, command_path=BUILD_PROMOTE_PATH),
    ] = None
    from_run_id: Annotated[
        str | None,
        option_param(BUILD_PROMOTE_FROM_RUN, command_path=BUILD_PROMOTE_PATH),
    ] = None
    note: Annotated[
        str | None,
        option_param(BUILD_PROMOTE_NOTE, command_path=BUILD_PROMOTE_PATH),
    ] = None
    output_format: Annotated[
        str,
        option_param(BUILD_PROMOTE_FORMAT, command_path=BUILD_PROMOTE_PATH),
    ] = "json"
    flags: SharedFlags = shared_flags_field(BUILD_PROMOTE_PATH)


@cli_command("build.resolve", handler=build_resolve_handler, config=_BUILD_CONFIG)
@build_app.command(name="resolve")
@dataclass
class BuildResolveCommand:
    """Resolve an alias to a version hash."""

    asset: Annotated[
        str,
        option_param(BUILD_RESOLVE_ASSET, command_path=BUILD_RESOLVE_PATH),
    ]
    alias: Annotated[
        str,
        option_param(BUILD_RESOLVE_ALIAS, command_path=BUILD_RESOLVE_PATH),
    ]
    output_format: Annotated[
        str,
        option_param(BUILD_RESOLVE_FORMAT, command_path=BUILD_RESOLVE_PATH),
    ] = "json"
    flags: SharedFlags = shared_flags_field(BUILD_RESOLVE_PATH)


@cli_command("build.diff", handler=build_diff_handler, config=_BUILD_CONFIG)
@build_app.command(name="diff")
@dataclass
class BuildDiffCommand:
    """Diff two versions of an asset."""

    asset: Annotated[
        str,
        option_param(BUILD_DIFF_ASSET, command_path=BUILD_DIFF_PATH),
    ]
    from_spec: Annotated[
        str,
        option_param(BUILD_DIFF_FROM, command_path=BUILD_DIFF_PATH),
    ]
    to_spec: Annotated[
        str,
        option_param(BUILD_DIFF_TO, command_path=BUILD_DIFF_PATH),
    ]
    output_format: Annotated[
        str,
        option_param(BUILD_DIFF_FORMAT, command_path=BUILD_DIFF_PATH),
    ] = "json"
    flags: SharedFlags = shared_flags_field(BUILD_DIFF_PATH)


@cli_command("build.impact", handler=build_impact_handler, config=_BUILD_CONFIG)
@build_app.command(name="impact")
@dataclass
class BuildImpactCommand:
    """Analyze downstream impact of an asset change.

    Traverses the asset lineage graph to identify which assets and targets
    would be affected by changes to the specified asset.
    """

    asset_kind: Annotated[
        str,
        option_param(BUILD_IMPACT_ASSET_KIND, command_path=BUILD_IMPACT_PATH),
    ] = "table"
    asset_key: Annotated[
        str,
        option_param(BUILD_IMPACT_ASSET_KEY, command_path=BUILD_IMPACT_PATH),
    ] = ""
    version_hash: Annotated[
        str | None,
        option_param(BUILD_IMPACT_VERSION_HASH, command_path=BUILD_IMPACT_PATH),
    ] = None
    show_targets: Annotated[
        bool,
        option_param(BUILD_IMPACT_SHOW_TARGETS, command_path=BUILD_IMPACT_PATH),
    ] = False
    max_depth: Annotated[
        int,
        option_param(BUILD_IMPACT_MAX_DEPTH, command_path=BUILD_IMPACT_PATH),
    ] = 10
    output_format: Annotated[
        str,
        option_param(BUILD_IMPACT_FORMAT, command_path=BUILD_IMPACT_PATH),
    ] = "json"
    flags: SharedFlags = shared_flags_field(BUILD_IMPACT_PATH)


@cli_command("build.decision_trace", handler=build_decision_trace_handler, config=_TRACE_CONFIG)
@build_app.command(name="decision-trace")
@dataclass
class BuildDecisionTraceCommand:
    """Show or export the latest build decision trace."""

    input_file: Annotated[
        str | None,
        option_param(BUILD_DECISION_TRACE_PATH, command_path=BUILD_DECISION_TRACE_CMD_PATH),
    ] = None
    output_file: Annotated[
        str | None,
        option_param(BUILD_DECISION_TRACE_OUTPUT, command_path=BUILD_DECISION_TRACE_CMD_PATH),
    ] = None
    flags: SharedFlags = shared_flags_field(BUILD_DECISION_TRACE_CMD_PATH)


__all__ = ["build_app"]
