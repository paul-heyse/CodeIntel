"""Build system commands for minimal-work target computation.

This module wires Cyclopts command classes to unified handlers via @cli_command.

Note: Build commands require complex runtime/gateway/snapshot access that is not
yet fully supported by the Command[T] pattern's Deps abstraction. They use
the handler pattern for now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.commands._common import SHARED_FLAGS_METADATA, SharedFlags
from codeintel.cli.commands.build_schema import build_schema_app
from codeintel.cli.commands.build_spec import build_spec_app
from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.build import (
    build_assets_handler,
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

build_app = App(
    name="build",
    help="Build system commands for minimal-work target computation.",
)
build_app.command(build_schema_app, name="schema")
build_app.command(build_spec_app, name="spec")


_BUILD_CONFIG = CommandConfig(require_runtime=True, require_gateway=True)
_VALIDATE_CONFIG = CommandConfig(require_runtime=False, require_gateway=False)


@cli_command("build.run", handler=build_run_handler, config=_BUILD_CONFIG)
@build_app.command(name="run")
@dataclass
class BuildRunCommand:
    """Build targets with automatic dependency resolution."""

    targets: Annotated[
        list[str] | None,
        Parameter(
            name=None,
            help="Target names to build (e.g., function_metrics, call_graph).",
        ),
    ] = None
    module: Annotated[
        str | None,
        Parameter(
            name=["--module", "-m"],
            help="Build all targets in a module (ingestion, graphs, analytics, export).",
            show_choices=True,
        ),
    ] = None
    all_targets: Annotated[
        bool,
        Parameter(
            name=["--all", "-a"],
            help="Build all targets across all modules.",
            negative=(),
        ),
    ] = False
    dry_run: Annotated[
        bool,
        Parameter(
            name=["--dry-run", "-n"],
            help="Show build plan without executing.",
            negative=(),
        ),
    ] = False
    force: Annotated[
        list[str] | None,
        Parameter(
            name=["--force", "-f"],
            help="Force recompute of specific targets (repeatable).",
        ),
    ] = None
    validate_outputs: Annotated[
        bool,
        Parameter(
            name=["--validate-outputs"],
            help="Validate produced datasets against Pandera schemas after write.",
            negative=(),
        ),
    ] = False
    strict_contracts: Annotated[
        bool,
        Parameter(
            name=["--strict-contracts"],
            help="Fail if target writes outside declared contract.",
            negative=(),
        ),
    ] = False
    publish_serving_snapshot: Annotated[
        bool,
        Parameter(
            name=["--publish-serving-snapshot"],
            help="Publish an immutable serving snapshot (writes current.json and snapshot artifacts).",
            negative=(),
        ),
    ] = False
    parallel_backend: Annotated[
        str,
        Parameter(
            name=["--parallel-backend"],
            help=(
                "Parallel execution backend.\n\n"
                "Options: sequential (default, safest); threadpool (multi-threaded with write lock); "
                "auto (auto-select best backend).\n\n"
                "Example: --parallel-backend=threadpool --max-workers=4."
            ),
            show_choices=True,
        ),
    ] = "sequential"
    max_workers: Annotated[
        int | None,
        Parameter(
            name=["--max-workers", "--workers"],
            help="Maximum parallel workers for threadpool backend.",
        ),
    ] = None
    enable_cache: Annotated[
        bool,
        Parameter(
            name=["--cache"],
            help="Enable Hamilton on-disk caching for nodes decorated with @cache.",
            negative=("--no-cache",),
        ),
    ] = True
    cache_dir: Annotated[
        str | None,
        Parameter(
            name=["--cache-dir"],
            help="Directory for Hamilton cache (default: build/.hamilton_cache).",
        ),
    ] = None
    clear_cache: Annotated[
        bool,
        Parameter(
            name=["--clear-cache"],
            help="Clear the Hamilton cache directory before executing.",
            negative=(),
        ),
    ] = False
    cache_report: Annotated[
        bool,
        Parameter(
            name=["--cache-report"],
            help="Include a cache hit/miss report for nodes decorated with @cache.",
            negative=(),
        ),
    ] = False
    enable_progress: Annotated[
        bool,
        Parameter(
            name=["--progress"],
            help="Show progress bar during execution.",
            negative=("--no-progress",),
        ),
    ] = False
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


@cli_command("build.status", handler=build_status_handler, config=_BUILD_CONFIG)
@build_app.command(name="status")
@dataclass
class BuildStatusCommand:
    """Show current state of build targets."""

    module: Annotated[
        str | None,
        Parameter(
            name=["--module", "-m"],
            help="Filter status to a specific module (ingestion, graphs, analytics, export).",
            show_choices=True,
        ),
    ] = None
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


@cli_command("build.history", handler=build_history_handler, config=_BUILD_CONFIG)
@build_app.command(name="history")
@dataclass
class BuildHistoryCommand:
    """Show build run history and details."""

    run_id: Annotated[
        str | None,
        Parameter(
            name=["--run-id", "-i"],
            help="Specific run ID to show details for (prefix match supported).",
        ),
    ] = None
    limit: Annotated[
        int,
        Parameter(
            name=["--limit", "-n"],
            help="Number of recent runs to show.",
        ),
    ] = 10
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


@cli_command("build.validate", handler=build_validate_handler, config=_VALIDATE_CONFIG)
@build_app.command(name="validate")
@dataclass
class BuildValidateCommand:
    """Validate Hamilton DAG invariants for DAG-first planning."""

    output_format: Annotated[
        str,
        Parameter(
            name=["--format"],
            help="Output format: json (default).",
        ),
    ] = "json"
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


@cli_command("build.plan", handler=build_plan_handler, config=_BUILD_CONFIG)
@build_app.command(name="plan")
@dataclass
class BuildPlanCommand:
    """Show build plan with status and reason for each target."""

    targets: Annotated[
        list[str] | None,
        Parameter(
            name=None,
            help="Target names to plan (e.g., function_metrics, call_graph).",
        ),
    ] = None
    module: Annotated[
        str | None,
        Parameter(
            name=["--module", "-m"],
            help="Plan all targets in a module (ingestion, graphs, analytics, export).",
            show_choices=True,
        ),
    ] = None
    all_targets: Annotated[
        bool,
        Parameter(
            name=["--all", "-a"],
            help="Plan all targets across all modules.",
            negative=(),
        ),
    ] = False
    force: Annotated[
        list[str] | None,
        Parameter(
            name=["--force", "-f"],
            help="Mark specific targets as forced (repeatable).",
        ),
    ] = None
    output_file: Annotated[
        str | None,
        Parameter(
            name=["--output", "-o"],
            help="Output file path (stdout if not specified).",
        ),
    ] = None
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


@cli_command("build.explain", handler=build_explain_handler, config=_BUILD_CONFIG)
@build_app.command(name="explain")
@dataclass
class BuildExplainCommand:
    """Explain why a target is stale and what dependencies changed."""

    target: Annotated[
        str,
        Parameter(
            name=None,
            help="Target name to explain (e.g., function_metrics).",
        ),
    ]
    force: Annotated[
        list[str] | None,
        Parameter(
            name=["--force", "-f"],
            help="Mark specific targets as forced (repeatable).",
        ),
    ] = None
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


@cli_command("build.graph", handler=build_graph_handler, config=_BUILD_CONFIG)
@build_app.command(name="graph")
@dataclass
class BuildGraphCommand:
    """Export Hamilton DAG for specified targets."""

    targets: Annotated[
        list[str] | None,
        Parameter(
            name=None,
            help="Target names to show DAG for (e.g., function_metrics, call_graph).",
        ),
    ] = None
    module: Annotated[
        str | None,
        Parameter(
            name=["--module", "-m"],
            help="Show DAG for all targets in a module (ingestion, graphs, analytics, export).",
            show_choices=True,
        ),
    ] = None
    all_targets: Annotated[
        bool,
        Parameter(
            name=["--all", "-a"],
            help="Show DAG for all targets across all modules.",
            negative=(),
        ),
    ] = False
    output_format: Annotated[
        str,
        Parameter(
            name=["--format", "-f"],
            help="Output format: json (default), mermaid, or dot.",
        ),
    ] = "json"
    output_file: Annotated[
        str | None,
        Parameter(
            name=["--output", "-o"],
            help="Output file path (stdout if not specified).",
        ),
    ] = None
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


@cli_command("build.assets", handler=build_assets_handler, config=_BUILD_CONFIG)
@build_app.command(name="assets")
@dataclass
class BuildAssetsCommand:
    """List materialized assets for the current snapshot."""

    asset: Annotated[
        str | None,
        Parameter(
            name=["--asset"],
            help="Filter to a specific asset key (e.g., analytics.function_metrics).",
        ),
    ] = None
    target: Annotated[
        str | None,
        Parameter(
            name=["--target", "-t"],
            help="Filter to assets produced by a specific target.",
        ),
    ] = None
    versions: Annotated[
        bool,
        Parameter(
            name=["--versions"],
            help="Include asset versions from the Phase 4 catalog.",
            negative=(),
        ),
    ] = False
    asset_type: Annotated[
        str | None,
        Parameter(
            name=["--type"],
            help="Filter by asset type: table, view, or artifact.",
        ),
    ] = None
    output_format: Annotated[
        str,
        Parameter(
            name=["--format", "-f"],
            help="Output format: table (default), json, or csv.",
        ),
    ] = "table"
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


@cli_command("build.lineage", handler=build_lineage_handler, config=_BUILD_CONFIG)
@build_app.command(name="lineage")
@dataclass
class BuildLineageCommand:
    """Show asset lineage for the current snapshot."""

    asset: Annotated[
        str,
        Parameter(
            name=["--asset"],
            help="Asset key to traverse (e.g., analytics.goid_risk_factors).",
        ),
    ]
    direction: Annotated[
        str,
        Parameter(
            name=["--direction", "-d"],
            help="Traversal direction: up (dependencies) or down (dependents).",
            show_choices=True,
        ),
    ] = "up"
    depth: Annotated[
        int,
        Parameter(
            name=["--depth"],
            help="Traversal depth (number of hops).",
        ),
    ] = 1
    output_format: Annotated[
        str,
        Parameter(
            name=["--format", "-f"],
            help="Output format: json (default) or text.",
        ),
    ] = "json"
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


@cli_command("build.promote", handler=build_promote_handler, config=_BUILD_CONFIG)
@build_app.command(name="promote")
@dataclass
class BuildPromoteCommand:
    """Set an alias for an asset version."""

    asset: Annotated[
        str,
        Parameter(
            name=["--asset"],
            help="Asset key to promote.",
        ),
    ]
    alias: Annotated[
        str,
        Parameter(
            name=["--alias"],
            help="Alias to set (e.g., main, latest, release-2025.01).",
        ),
    ]
    version_hash: Annotated[
        str | None,
        Parameter(
            name=["--version-hash"],
            help="Version hash to pin (preferred).",
        ),
    ] = None
    from_run_id: Annotated[
        str | None,
        Parameter(
            name=["--from-run-id"],
            help="Use the version recorded for this run_id.",
        ),
    ] = None
    note: Annotated[
        str | None,
        Parameter(
            name=["--note"],
            help="Optional note describing the promotion.",
        ),
    ] = None
    output_format: Annotated[
        str,
        Parameter(
            name=["--format", "-f"],
            help="Output format: json (default) or text.",
        ),
    ] = "json"
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


@cli_command("build.resolve", handler=build_resolve_handler, config=_BUILD_CONFIG)
@build_app.command(name="resolve")
@dataclass
class BuildResolveCommand:
    """Resolve an alias to a version hash."""

    asset: Annotated[
        str,
        Parameter(
            name=["--asset"],
            help="Asset key to resolve.",
        ),
    ]
    alias: Annotated[
        str,
        Parameter(
            name=["--alias"],
            help="Alias to resolve (e.g., main, latest).",
        ),
    ]
    output_format: Annotated[
        str,
        Parameter(
            name=["--format", "-f"],
            help="Output format: json (default) or text.",
        ),
    ] = "json"
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


@cli_command("build.diff", handler=build_diff_handler, config=_BUILD_CONFIG)
@build_app.command(name="diff")
@dataclass
class BuildDiffCommand:
    """Diff two versions of an asset."""

    asset: Annotated[
        str,
        Parameter(
            name=["--asset"],
            help="Asset key to diff.",
        ),
    ]
    from_spec: Annotated[
        str,
        Parameter(
            name=["--from"],
            help="Baseline version spec (alias or version hash).",
        ),
    ]
    to_spec: Annotated[
        str,
        Parameter(
            name=["--to"],
            help="Target version spec (alias or version hash).",
        ),
    ]
    output_format: Annotated[
        str,
        Parameter(
            name=["--format", "-f"],
            help="Output format: json (default) or text.",
        ),
    ] = "json"
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


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
        Parameter(
            name=["--asset-kind"],
            help="Kind of source asset: table or artifact.",
        ),
    ] = "table"
    asset_key: Annotated[
        str,
        Parameter(
            name=["--asset-key"],
            help="Key of source asset (e.g., analytics.function_metrics).",
        ),
    ] = ""
    version_hash: Annotated[
        str | None,
        Parameter(
            name=["--version-hash"],
            help="Specific version hash to analyze (optional).",
        ),
    ] = None
    show_targets: Annotated[
        bool,
        Parameter(
            name=["--show-targets"],
            help="Include target names that would need to re-run.",
            negative=(),
        ),
    ] = False
    max_depth: Annotated[
        int,
        Parameter(
            name=["--max-depth"],
            help="Maximum traversal depth.",
        ),
    ] = 10
    output_format: Annotated[
        str,
        Parameter(
            name=["--format", "-f"],
            help="Output format: json (default) or text.",
        ),
    ] = "json"
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


__all__ = ["build_app"]
