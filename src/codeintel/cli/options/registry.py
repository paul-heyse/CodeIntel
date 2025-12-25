"""Canonical CLI option registry for CodeIntel commands."""

from __future__ import annotations

from codeintel.cli.options.types import OptionGroup, OptionSpec

# ---------------------------------------------------------------------------
# Shared flags
# ---------------------------------------------------------------------------

PROJECT_ROOT = OptionSpec(
    arg_name="project_root",
    names=("--root", "-r"),
    help="Explicit project root directory.",
    env_name="root",
)

OUTPUT_FORMAT = OptionSpec(
    arg_name="output_format",
    names=("--output-format",),
    help="Output format.",
    show_choices=True,
)

JSON_FLAG = OptionSpec(
    arg_name="json",
    names=("--json",),
    help="Alias for --output-format json.",
    negative=(),
)

VERBOSE = OptionSpec(
    arg_name="verbose",
    names=("--verbose", "-v"),
    help="Increase verbosity (0=warnings, 1=info, 2=debug).",
    count=True,
)

SHARED_FLAGS = OptionGroup(
    name="shared_flags",
    options={
        "project_root": PROJECT_ROOT,
        "output_format": OUTPUT_FORMAT,
        "json": JSON_FLAG,
        "verbose": VERBOSE,
    },
)


# ---------------------------------------------------------------------------
# Build command options
# ---------------------------------------------------------------------------

BUILD_RUN_TARGETS = OptionSpec(
    arg_name="targets",
    help="Target names to build (e.g., function_metrics, call_graph).",
)
BUILD_RUN_MODULE = OptionSpec(
    arg_name="module",
    names=("--module", "-m"),
    help="Build all targets in a module (ingestion, graphs, analytics, export).",
    show_choices=True,
)
BUILD_RUN_ALL_TARGETS = OptionSpec(
    arg_name="all_targets",
    names=("--all", "-a"),
    help="Build all targets across all modules.",
    negative=(),
    env_name="all",
)
BUILD_RUN_DRY_RUN = OptionSpec(
    arg_name="dry_run",
    names=("--dry-run", "-n"),
    help="Show build plan without executing.",
    negative=(),
)
BUILD_RUN_FORCE = OptionSpec(
    arg_name="force",
    names=("--force", "-f"),
    help="Force recompute of specific targets (repeatable).",
)
BUILD_RUN_VALIDATE_OUTPUTS = OptionSpec(
    arg_name="validate_outputs",
    names=("--validate-outputs",),
    help="Validate produced datasets against Pandera schemas after write.",
    negative=(),
)
BUILD_RUN_STRICT_CONTRACTS = OptionSpec(
    arg_name="strict_contracts",
    names=("--strict-contracts",),
    help="Fail if target writes outside declared contract.",
    negative=(),
)
BUILD_RUN_PUBLISH_SNAPSHOT = OptionSpec(
    arg_name="publish_serving_snapshot",
    names=("--publish-serving-snapshot",),
    help="Publish an immutable serving snapshot (writes current.json and snapshot artifacts).",
    negative=(),
)
BUILD_RUN_PARALLEL_BACKEND = OptionSpec(
    arg_name="parallel_backend",
    names=("--parallel-backend",),
    help=(
        "Parallel execution backend.\n\n"
        "Options: sequential (default, safest); threadpool (multi-threaded with write lock); "
        "auto (auto-select best backend).\n\n"
        "Example: --parallel-backend=threadpool --max-workers=4."
    ),
    show_choices=True,
)
BUILD_RUN_MAX_WORKERS = OptionSpec(
    arg_name="max_workers",
    names=("--max-workers", "--workers"),
    help="Maximum parallel workers for threadpool backend.",
)
BUILD_RUN_ENABLE_CACHE = OptionSpec(
    arg_name="enable_cache",
    names=("--cache",),
    help="Enable Hamilton on-disk caching for nodes decorated with @cache.",
    negative=("--no-cache",),
    env_name="cache",
)
BUILD_RUN_CACHE_DIR = OptionSpec(
    arg_name="cache_dir",
    names=("--cache-dir",),
    help="Directory for Hamilton cache (default: build/.hamilton_cache).",
)
BUILD_RUN_CLEAR_CACHE = OptionSpec(
    arg_name="clear_cache",
    names=("--clear-cache",),
    help="Clear the Hamilton cache directory before executing.",
    negative=(),
)
BUILD_RUN_CACHE_REPORT = OptionSpec(
    arg_name="cache_report",
    names=("--cache-report",),
    help="Include a cache hit/miss report for nodes decorated with @cache.",
    negative=(),
)
BUILD_RUN_PROGRESS = OptionSpec(
    arg_name="enable_progress",
    names=("--progress",),
    help="Show progress bar during execution.",
    negative=("--no-progress",),
    env_name="progress",
)

BUILD_STATUS_MODULE = OptionSpec(
    arg_name="module",
    names=("--module", "-m"),
    help="Filter status to a specific module (ingestion, graphs, analytics, export).",
    show_choices=True,
)

BUILD_HISTORY_RUN_ID = OptionSpec(
    arg_name="run_id",
    names=("--run-id", "-i"),
    help="Specific run ID to show details for (prefix match supported).",
)
BUILD_HISTORY_LIMIT = OptionSpec(
    arg_name="limit",
    names=("--limit", "-n"),
    help="Number of recent runs to show.",
)

BUILD_VALIDATE_FORMAT = OptionSpec(
    arg_name="output_format",
    names=("--format",),
    help="Output format: json (default).",
    env_name="format",
)

BUILD_PLAN_TARGETS = OptionSpec(
    arg_name="targets",
    help="Target names to plan (e.g., function_metrics, call_graph).",
)
BUILD_PLAN_MODULE = OptionSpec(
    arg_name="module",
    names=("--module", "-m"),
    help="Plan all targets in a module (ingestion, graphs, analytics, export).",
    show_choices=True,
)
BUILD_PLAN_ALL = OptionSpec(
    arg_name="all_targets",
    names=("--all", "-a"),
    help="Plan all targets across all modules.",
    negative=(),
    env_name="all",
)
BUILD_PLAN_FORCE = OptionSpec(
    arg_name="force",
    names=("--force", "-f"),
    help="Mark specific targets as forced (repeatable).",
)
BUILD_PLAN_OUTPUT = OptionSpec(
    arg_name="output_file",
    names=("--output", "-o"),
    help="Output file path (stdout if not specified).",
    env_name="output",
)

BUILD_EXPLAIN_TARGET = OptionSpec(
    arg_name="target",
    help="Target name to explain (e.g., function_metrics).",
)
BUILD_EXPLAIN_FORCE = OptionSpec(
    arg_name="force",
    names=("--force", "-f"),
    help="Mark specific targets as forced (repeatable).",
)
BUILD_EXPLAIN_IO_SURFACE = OptionSpec(
    arg_name="io_surface",
    names=("--io-surface",),
    help=(
        "Include a per-target IO surface (reads/writes) derived strictly from "
        "Hamilton DAG tags."
    ),
    negative=(),
)

BUILD_GRAPH_TARGETS = OptionSpec(
    arg_name="targets",
    help="Target names to show DAG for (e.g., function_metrics, call_graph).",
)
BUILD_GRAPH_MODULE = OptionSpec(
    arg_name="module",
    names=("--module", "-m"),
    help="Show DAG for all targets in a module (ingestion, graphs, analytics, export).",
    show_choices=True,
)
BUILD_GRAPH_ALL = OptionSpec(
    arg_name="all_targets",
    names=("--all", "-a"),
    help="Show DAG for all targets across all modules.",
    negative=(),
    env_name="all",
)
BUILD_GRAPH_FORMAT = OptionSpec(
    arg_name="output_format",
    names=("--format", "-f"),
    help="Output format: json (default), mermaid, or dot.",
    env_name="format",
)
BUILD_GRAPH_OUTPUT = OptionSpec(
    arg_name="output_file",
    names=("--output", "-o"),
    help="Output file path (stdout if not specified).",
    env_name="output",
)

BUILD_ASSETS_ASSET = OptionSpec(
    arg_name="asset",
    names=("--asset",),
    help="Filter to a specific asset key (e.g., analytics.function_metrics).",
)
BUILD_ASSETS_TARGET = OptionSpec(
    arg_name="target",
    names=("--target", "-t"),
    help="Filter to assets produced by a specific target.",
)
BUILD_ASSETS_TYPE = OptionSpec(
    arg_name="asset_type",
    names=("--type",),
    help="Filter by asset type: table, view, or artifact.",
    env_name="type",
)
BUILD_ASSETS_FORMAT = OptionSpec(
    arg_name="output_format",
    names=("--format", "-f"),
    help="Output format: table (default), json, or csv.",
    env_name="format",
)

BUILD_LINEAGE_ASSET = OptionSpec(
    arg_name="asset",
    names=("--asset",),
    help="Asset key to traverse (e.g., analytics.goid_risk_factors).",
)
BUILD_LINEAGE_DIRECTION = OptionSpec(
    arg_name="direction",
    names=("--direction", "-d"),
    help="Traversal direction: up (dependencies) or down (dependents).",
    show_choices=True,
)
BUILD_LINEAGE_DEPTH = OptionSpec(
    arg_name="depth",
    names=("--depth",),
    help="Traversal depth (number of hops).",
)
BUILD_LINEAGE_FORMAT = OptionSpec(
    arg_name="output_format",
    names=("--format", "-f"),
    help="Output format: json (default) or text.",
    env_name="format",
)

BUILD_PROMOTE_ASSET = OptionSpec(
    arg_name="asset",
    names=("--asset",),
    help="Asset key to promote.",
)
BUILD_PROMOTE_ALIAS = OptionSpec(
    arg_name="alias",
    names=("--alias",),
    help="Alias to set (e.g., main, latest, release-2025.01).",
)
BUILD_PROMOTE_VERSION_HASH = OptionSpec(
    arg_name="version_hash",
    names=("--version-hash",),
    help="Version hash to pin (preferred).",
)
BUILD_PROMOTE_FROM_RUN = OptionSpec(
    arg_name="from_run_id",
    names=("--from-run-id",),
    help="Use the version recorded for this run_id.",
)
BUILD_PROMOTE_NOTE = OptionSpec(
    arg_name="note",
    names=("--note",),
    help="Optional note describing the promotion.",
)
BUILD_PROMOTE_FORMAT = OptionSpec(
    arg_name="output_format",
    names=("--format", "-f"),
    help="Output format: json (default) or text.",
    env_name="format",
)

BUILD_RESOLVE_ASSET = OptionSpec(
    arg_name="asset",
    names=("--asset",),
    help="Asset key to resolve.",
)
BUILD_RESOLVE_ALIAS = OptionSpec(
    arg_name="alias",
    names=("--alias",),
    help="Alias to resolve (e.g., main, latest).",
)
BUILD_RESOLVE_FORMAT = OptionSpec(
    arg_name="output_format",
    names=("--format", "-f"),
    help="Output format: json (default) or text.",
    env_name="format",
)

BUILD_DIFF_ASSET = OptionSpec(
    arg_name="asset",
    names=("--asset",),
    help="Asset key to diff.",
)
BUILD_DIFF_FROM = OptionSpec(
    arg_name="from_spec",
    names=("--from",),
    help="Baseline version spec (alias or version hash).",
    env_name="from",
)
BUILD_DIFF_TO = OptionSpec(
    arg_name="to_spec",
    names=("--to",),
    help="Target version spec (alias or version hash).",
    env_name="to",
)
BUILD_DIFF_FORMAT = OptionSpec(
    arg_name="output_format",
    names=("--format", "-f"),
    help="Output format: json (default) or text.",
    env_name="format",
)

BUILD_IMPACT_ASSET_KIND = OptionSpec(
    arg_name="asset_kind",
    names=("--asset-kind",),
    help="Kind of source asset: table or artifact.",
)
BUILD_IMPACT_ASSET_KEY = OptionSpec(
    arg_name="asset_key",
    names=("--asset-key",),
    help="Key of source asset (e.g., analytics.function_metrics).",
)
BUILD_IMPACT_VERSION_HASH = OptionSpec(
    arg_name="version_hash",
    names=("--version-hash",),
    help="Specific version hash to analyze (optional).",
)
BUILD_IMPACT_SHOW_TARGETS = OptionSpec(
    arg_name="show_targets",
    names=("--show-targets",),
    help="Include target names that would need to re-run.",
    negative=(),
)
BUILD_IMPACT_MAX_DEPTH = OptionSpec(
    arg_name="max_depth",
    names=("--max-depth",),
    help="Maximum traversal depth.",
)
BUILD_IMPACT_FORMAT = OptionSpec(
    arg_name="output_format",
    names=("--format", "-f"),
    help="Output format: json (default) or text.",
    env_name="format",
)


__all__ = [
    "JSON_FLAG",
    "OUTPUT_FORMAT",
    "PROJECT_ROOT",
    "SHARED_FLAGS",
    "VERBOSE",
    "BUILD_ASSETS_ASSET",
    "BUILD_ASSETS_FORMAT",
    "BUILD_ASSETS_TARGET",
    "BUILD_ASSETS_TYPE",
    "BUILD_DIFF_ASSET",
    "BUILD_DIFF_FORMAT",
    "BUILD_DIFF_FROM",
    "BUILD_DIFF_TO",
    "BUILD_EXPLAIN_FORCE",
    "BUILD_EXPLAIN_IO_SURFACE",
    "BUILD_EXPLAIN_TARGET",
    "BUILD_GRAPH_ALL",
    "BUILD_GRAPH_FORMAT",
    "BUILD_GRAPH_MODULE",
    "BUILD_GRAPH_OUTPUT",
    "BUILD_GRAPH_TARGETS",
    "BUILD_HISTORY_LIMIT",
    "BUILD_HISTORY_RUN_ID",
    "BUILD_IMPACT_ASSET_KEY",
    "BUILD_IMPACT_ASSET_KIND",
    "BUILD_IMPACT_FORMAT",
    "BUILD_IMPACT_MAX_DEPTH",
    "BUILD_IMPACT_SHOW_TARGETS",
    "BUILD_IMPACT_VERSION_HASH",
    "BUILD_LINEAGE_ASSET",
    "BUILD_LINEAGE_DEPTH",
    "BUILD_LINEAGE_DIRECTION",
    "BUILD_LINEAGE_FORMAT",
    "BUILD_PLAN_ALL",
    "BUILD_PLAN_FORCE",
    "BUILD_PLAN_MODULE",
    "BUILD_PLAN_OUTPUT",
    "BUILD_PLAN_TARGETS",
    "BUILD_PROMOTE_ALIAS",
    "BUILD_PROMOTE_ASSET",
    "BUILD_PROMOTE_FORMAT",
    "BUILD_PROMOTE_FROM_RUN",
    "BUILD_PROMOTE_NOTE",
    "BUILD_PROMOTE_VERSION_HASH",
    "BUILD_RESOLVE_ALIAS",
    "BUILD_RESOLVE_ASSET",
    "BUILD_RESOLVE_FORMAT",
    "BUILD_RUN_ALL_TARGETS",
    "BUILD_RUN_CACHE_DIR",
    "BUILD_RUN_CACHE_REPORT",
    "BUILD_RUN_CLEAR_CACHE",
    "BUILD_RUN_DRY_RUN",
    "BUILD_RUN_ENABLE_CACHE",
    "BUILD_RUN_FORCE",
    "BUILD_RUN_MAX_WORKERS",
    "BUILD_RUN_MODULE",
    "BUILD_RUN_PARALLEL_BACKEND",
    "BUILD_RUN_PROGRESS",
    "BUILD_RUN_PUBLISH_SNAPSHOT",
    "BUILD_RUN_STRICT_CONTRACTS",
    "BUILD_RUN_TARGETS",
    "BUILD_RUN_VALIDATE_OUTPUTS",
    "BUILD_STATUS_MODULE",
    "BUILD_VALIDATE_FORMAT",
]
