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
BUILD_RUN_VALIDATION_MODE = OptionSpec(
    arg_name="validation_mode",
    names=("--validation-mode",),
    help="Contract validation mode: lenient, strict, or off.",
    show_choices=True,
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
        "Include a per-target IO surface (reads/writes) derived strictly from Hamilton DAG tags."
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


# ---------------------------------------------------------------------------
# Build decision trace options
# ---------------------------------------------------------------------------

BUILD_DECISION_TRACE_PATH = OptionSpec(
    arg_name="input_file",
    names=("--path",),
    help="Path to decision trace JSON (defaults to build/decision_trace.json).",
    env_name="path",
)
BUILD_DECISION_TRACE_OUTPUT = OptionSpec(
    arg_name="output_file",
    names=("--output", "-o"),
    help="Output file path (stdout if not specified).",
    env_name="output",
)


# ---------------------------------------------------------------------------
# Build schema command options
# ---------------------------------------------------------------------------

BUILD_SCHEMA_TARGETS = OptionSpec(
    arg_name="targets",
    help="Target names to include (defaults to all targets).",
)
BUILD_SCHEMA_MODULE = OptionSpec(
    arg_name="module",
    names=("--module", "-m"),
    help="Compile schemas for all targets in a module.",
    show_choices=True,
)
BUILD_SCHEMA_ALL = OptionSpec(
    arg_name="all_targets",
    names=("--all", "-a"),
    help="Compile schemas for all targets across all modules.",
    negative=(),
    env_name="all",
)
BUILD_SCHEMA_INFER_NATIVE = OptionSpec(
    arg_name="infer_native",
    names=("--infer-native", "--infer"),
    help="Infer schemas for inferable native targets.",
    negative=(),
)
BUILD_SCHEMA_STABLE = OptionSpec(
    arg_name="stable",
    names=("--stable",),
    help="Force deterministic ordering and canonicalized output.",
    negative=(),
)
BUILD_SCHEMA_FORMAT = OptionSpec(
    arg_name="output_format",
    names=("--format", "-f"),
    help="Output format: json (default).",
    env_name="format",
)
BUILD_SCHEMA_OUTPUT = OptionSpec(
    arg_name="output_file",
    names=("--output", "-o"),
    help="Output file path (stdout if not specified).",
    env_name="output",
)
BUILD_SCHEMA_INCLUDE_VIEWS = OptionSpec(
    arg_name="include_views",
    names=("--include-views",),
    help="Include DuckDB view schemas in the manifest (v2 format).",
    negative=(),
)
BUILD_SCHEMA_INCLUDE_ARTIFACTS = OptionSpec(
    arg_name="include_artifacts",
    names=("--include-artifacts",),
    help="Include export artifact metadata in the manifest (v2 format).",
    negative=(),
)
BUILD_SCHEMA_INCLUDE_PROVENANCE = OptionSpec(
    arg_name="include_provenance",
    names=("--include-provenance",),
    help="Include provenance metadata for schemas and artifacts (v2 format).",
    negative=(),
)
BUILD_SCHEMA_EXPECTED_FILE = OptionSpec(
    arg_name="expected_file",
    names=("--expected", "-e"),
    help="Path to an expected schema manifest JSON file.",
    env_name="expected",
)
BUILD_SCHEMA_FAIL_ON_BREAKING = OptionSpec(
    arg_name="fail_on_breaking",
    names=("--fail-on-breaking",),
    help="Exit with error if breaking changes detected (default: true).",
    negative=("--no-fail-on-breaking",),
)
BUILD_SCHEMA_FAIL_ON_ANY = OptionSpec(
    arg_name="fail_on_any",
    names=("--fail-on-any",),
    help="Exit with error on any schema drift, not just breaking changes.",
    negative=(),
)
BUILD_SCHEMA_DRY_RUN = OptionSpec(
    arg_name="dry_run",
    names=("--dry-run",),
    help="Show migration plan without writing changes (default: true).",
    negative=("--no-dry-run",),
)


# ---------------------------------------------------------------------------
# Build spec command options
# ---------------------------------------------------------------------------

BUILD_SPEC_INCLUDE_COLUMNS = OptionSpec(
    arg_name="include_columns",
    names=("--include-columns",),
    help="Include dataset column names in the compiled spec.",
    negative=("--no-include-columns",),
)
BUILD_SPEC_FORMAT = OptionSpec(
    arg_name="output_format",
    names=("--format",),
    help="Output format: json (default).",
    env_name="format",
)
BUILD_SPEC_OUTPUT = OptionSpec(
    arg_name="output_file",
    names=("--output", "-o"),
    help="Output file path (stdout if not specified).",
    env_name="output",
)


# ---------------------------------------------------------------------------
# Dataset ops command options
# ---------------------------------------------------------------------------

DATASET_TABLE_KEY = OptionSpec(
    arg_name="table_key",
    help="Dataset table key (e.g., 'core.goids').",
)


# ---------------------------------------------------------------------------
# Datasets command options
# ---------------------------------------------------------------------------

DATASETS_SCHEMA_DIR = OptionSpec(
    arg_name="schema_dir",
    names=("--schema-dir",),
    help="Directory containing export JSON Schemas.",
)
DATASETS_SAMPLING = OptionSpec(
    arg_name="sampling",
    names=("--sampling",),
    help="Sampling mode: enabled or disabled.",
)
DATASETS_DOCS_VIEW = OptionSpec(
    arg_name="docs_view",
    names=("--docs-view",),
    help='Docs view filter: "include", "exclude", or "only".',
)
DATASETS_READ_ONLY = OptionSpec(
    arg_name="read_only",
    names=("--read-only",),
    help='Read-only filter: "include", "exclude", or "only".',
)
DATASETS_MAX_DESCRIPTION = OptionSpec(
    arg_name="max_description",
    names=("--max-description",),
    help="Maximum description length before truncation.",
)
DATASETS_SNAPSHOT_OUTPUT = OptionSpec(
    arg_name="output",
    names=("--output",),
    help="Output file path for JSON dataset specs.",
)
DATASETS_DIFF_BASELINE = OptionSpec(
    arg_name="baseline",
    names=("--baseline",),
    help="Path to JSON baseline from `codeintel datasets snapshot`.",
)
DATASETS_DIFF_OUTPUT = OptionSpec(
    arg_name="output",
    names=("--output",),
    help="Optional output file path for writing current specs.",
)
DATASETS_DIFF_AGAINST_REF = OptionSpec(
    arg_name="against_ref",
    names=("--against-ref",),
    help="Git ref to diff against (e.g. HEAD~, main).",
)
DATASETS_DIFF_BASELINE_PATH = OptionSpec(
    arg_name="baseline_path",
    names=("--baseline-path",),
    help="Path of the snapshot file inside the git ref.",
)
DATASETS_SCAFFOLD_NAME = OptionSpec(
    arg_name="name",
    names=("name",),
    help="Dataset name to scaffold.",
)
DATASETS_SCAFFOLD_REGISTRY_CHECK = OptionSpec(
    arg_name="registry_check",
    names=("--registry-check",),
    help="Whether to fail when the dataset already exists.",
    show_default=True,
)
DATASETS_SCAFFOLD_DRY_RUN = OptionSpec(
    arg_name="dry_run",
    names=("--dry-run",),
    help="Perform validation only without writing files.",
    negative=("--no-dry-run",),
)


# ---------------------------------------------------------------------------
# Docs command options
# ---------------------------------------------------------------------------

DOCS_REPO = OptionSpec(
    arg_name="repo",
    names=("--repo",),
    help="Repository slug.",
)
DOCS_COMMIT = OptionSpec(
    arg_name="commit",
    names=("--commit",),
    help="Commit SHA.",
)
DOCS_DB_PATH = OptionSpec(
    arg_name="db_path",
    names=("--db-path",),
    help="Path to DuckDB database.",
)
DOCS_BUILD_DIR = OptionSpec(
    arg_name="build_dir",
    names=("--build-dir",),
    help="Build directory for docs export.",
)
DOCS_REPO_ROOT = OptionSpec(
    arg_name="repo_root",
    names=("--repo-root",),
    help="Repository root directory.",
)
DOCS_DOCUMENT_OUTPUT_DIR = OptionSpec(
    arg_name="document_output_dir",
    names=("--document-output-dir",),
    help="Document Output directory for emitted artifacts.",
)
DOCS_NX_BACKEND = OptionSpec(
    arg_name="nx_backend",
    names=("--nx-backend",),
    help="NetworkX backend selection: auto, cpu, or nx-cugraph.",
    show_choices=True,
)
DOCS_NX_GPU_MODE = OptionSpec(
    arg_name="nx_gpu_mode",
    names=("--nx-gpu-mode",),
    help="GPU backend preference: disabled, enabled, or strict.",
)
DOCS_VALIDATION_MODE = OptionSpec(
    arg_name="validation_mode",
    names=("--validation-mode",),
    help="Validation strategy: required or skip.",
    show_choices=True,
)
DOCS_VALIDATE = OptionSpec(
    arg_name="validate",
    names=("--validate",),
    help="Enable export validation.",
    negative=("--no-validate",),
)
DOCS_SKIP_PREREQS = OptionSpec(
    arg_name="skip_prereqs",
    names=("--skip-prereqs",),
    help="Skip prerequisite ingestion or build steps.",
    negative=("--run-prereqs",),
)
DOCS_SCHEMA = OptionSpec(
    arg_name="schemas",
    names=("--schema",),
    help="Table key to validate (repeatable).",
)
DOCS_DATASET = OptionSpec(
    arg_name="datasets",
    names=("--dataset",),
    help="Dataset name to export (repeatable).",
)
DOCS_RUN_MODE = OptionSpec(
    arg_name="run_mode",
    names=("--run-mode",),
    help="Execution mode for docs export.",
    show_choices=True,
)
DOCS_DRY_RUN = OptionSpec(
    arg_name="dry_run",
    names=("--dry-run",),
    help="Preview export without writing files.",
    negative=("--no-dry-run",),
)
DOCS_PREREQ_MODE = OptionSpec(
    arg_name="prereq_mode",
    names=("--prereq-mode",),
    help="Prerequisite execution mode.",
    show_choices=True,
)


# ---------------------------------------------------------------------------
# Graphs command options
# ---------------------------------------------------------------------------

GRAPH_NAMES = OptionSpec(
    arg_name="names",
    names=("--names",),
    help="Explicit target names to filter (repeatable).",
)
GRAPH_PLAN = OptionSpec(
    arg_name="plan",
    names=("--plan",),
    help="Display execution plan instead of listing.",
    negative=("--no-plan",),
)
GRAPH_SELECTION_POLICY = OptionSpec(
    arg_name="selection_policy",
    names=("--selection-policy",),
    help="How to handle unknown plugin names.",
    show_choices=True,
)
GRAPH_DEPENDENCY_POLICY = OptionSpec(
    arg_name="dependency_policy",
    names=("--dependency-policy",),
    help="How to handle missing plugin dependencies.",
    show_choices=True,
)


# ---------------------------------------------------------------------------
# History command options
# ---------------------------------------------------------------------------

HISTORY_REPO = OptionSpec(
    arg_name="repo",
    names=("--repo",),
    help="Repository slug (e.g., 'my-org/my-repo').",
)
HISTORY_COMMITS = OptionSpec(
    arg_name="commits",
    names=("--commits",),
    help="Commits to include in the timeseries (latest first).",
)
HISTORY_DB_DIR = OptionSpec(
    arg_name="db_dir",
    names=("--db-dir",),
    help="Directory with per-commit DuckDB snapshots.",
)
HISTORY_OUTPUT_DB = OptionSpec(
    arg_name="output_db",
    names=("--output-db",),
    help="Destination DuckDB for history_timeseries.",
)
HISTORY_ENTITY_KIND = OptionSpec(
    arg_name="entity_kind",
    names=("--entity-kind",),
    help="Entity kind to include: function, module, or both.",
)
HISTORY_MAX_ENTITIES = OptionSpec(
    arg_name="max_entities",
    names=("--max-entities",),
    help="Maximum entities to track (top-N by selection strategy).",
)
HISTORY_SELECTION_STRATEGY = OptionSpec(
    arg_name="selection_strategy",
    names=("--selection-strategy",),
    help="Selection strategy for picking entities (default: risk_score).",
)
HISTORY_REPO_ROOT = OptionSpec(
    arg_name="repo_root",
    names=("--repo-root",),
    help="Repository root directory.",
)


# ---------------------------------------------------------------------------
# Jobs command options
# ---------------------------------------------------------------------------

JOBS_STATUS_FILTER = OptionSpec(
    arg_name="status",
    names=("--status",),
    help="Filter by status.",
)
JOBS_LIMIT = OptionSpec(
    arg_name="limit",
    names=("--limit",),
    help="Maximum jobs to show.",
)
JOBS_JOB_ID = OptionSpec(
    arg_name="job_id",
    help="Job ID.",
)
JOBS_MAX_AGE_DAYS = OptionSpec(
    arg_name="max_age_days",
    names=("--max-age-days",),
    help="Maximum age in days.",
)


# ---------------------------------------------------------------------------
# Plugins command options
# ---------------------------------------------------------------------------

PLUGINS_NAME = OptionSpec(
    arg_name="name",
    help="Plugin name.",
)
PLUGINS_OUTPUT_DIR = OptionSpec(
    arg_name="output",
    names=("--output",),
    help="Output directory.",
)
PLUGINS_PATH = OptionSpec(
    arg_name="path",
    help="Plugin directory.",
)


# ---------------------------------------------------------------------------
# Serve command options
# ---------------------------------------------------------------------------

SERVE_HOST = OptionSpec(
    arg_name="host",
    names=("--host", "-h"),
    help="Host to bind to.",
)
SERVE_PORT = OptionSpec(
    arg_name="port",
    names=("--port", "-p"),
    help="Port to bind to.",
)
SERVE_RELOAD = OptionSpec(
    arg_name="reload",
    names=("--reload",),
    help="Enable auto-reload for development.",
    negative=(),
)


# ---------------------------------------------------------------------------
# Storage command options
# ---------------------------------------------------------------------------

STORAGE_DB_PATH = OptionSpec(
    arg_name="db_path",
    names=("--db-path",),
    help="Path to DuckDB database.",
)
STORAGE_VALIDATION_MODE = OptionSpec(
    arg_name="validation_mode",
    names=("--validation-mode",),
    help="Contract validation mode: lenient, strict, or off.",
    show_choices=True,
)
STORAGE_OUTPUT_DIR = OptionSpec(
    arg_name="output_dir",
    names=("--output-dir",),
    help="Output directory for profile report.",
)
STORAGE_INCLUDE_VIEWS = OptionSpec(
    arg_name="include_views",
    names=("--include-views",),
    help="Include views in profiling.",
    negative=(),
)
STORAGE_INPUT_DIR = OptionSpec(
    arg_name="input_dir",
    names=("--input-dir",),
    help="Directory containing a DuckDB EXPORT DATABASE dump.",
)


# ---------------------------------------------------------------------------
# Config command options
# ---------------------------------------------------------------------------

CONFIG_SOURCE = OptionSpec(
    arg_name="source",
    names=("--source",),
    help="Show only config from specific source.",
)
CONFIG_FORMAT = OptionSpec(
    arg_name="output_format",
    names=("--format",),
    help="Output format.",
    env_name="format",
)
CONFIG_TARGET = OptionSpec(
    arg_name="target",
    names=("--target",),
    help="Target path for config file.",
)


# ---------------------------------------------------------------------------
# Help commands options
# ---------------------------------------------------------------------------

HELP_OPERATION_ID = OptionSpec(
    arg_name="operation_id",
    help="Operation ID to describe.",
)
HELP_BY_GROUP = OptionSpec(
    arg_name="by_group",
    names=("--by-group",),
    help="Group operations by group.",
)
HELP_QUERY = OptionSpec(
    arg_name="query",
    help="Search query.",
)


# ---------------------------------------------------------------------------
# Completions command options
# ---------------------------------------------------------------------------

COMPLETIONS_SHELL = OptionSpec(
    arg_name="shell",
    help="Shell to install for.",
)


__all__ = [
    "BUILD_ASSETS_ASSET",
    "BUILD_ASSETS_FORMAT",
    "BUILD_ASSETS_TARGET",
    "BUILD_ASSETS_TYPE",
    "BUILD_DECISION_TRACE_OUTPUT",
    "BUILD_DECISION_TRACE_PATH",
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
    "BUILD_RUN_VALIDATION_MODE",
    "BUILD_SCHEMA_ALL",
    "BUILD_SCHEMA_DRY_RUN",
    "BUILD_SCHEMA_EXPECTED_FILE",
    "BUILD_SCHEMA_FAIL_ON_ANY",
    "BUILD_SCHEMA_FAIL_ON_BREAKING",
    "BUILD_SCHEMA_FORMAT",
    "BUILD_SCHEMA_INCLUDE_ARTIFACTS",
    "BUILD_SCHEMA_INCLUDE_PROVENANCE",
    "BUILD_SCHEMA_INCLUDE_VIEWS",
    "BUILD_SCHEMA_INFER_NATIVE",
    "BUILD_SCHEMA_MODULE",
    "BUILD_SCHEMA_OUTPUT",
    "BUILD_SCHEMA_STABLE",
    "BUILD_SCHEMA_TARGETS",
    "BUILD_SPEC_FORMAT",
    "BUILD_SPEC_INCLUDE_COLUMNS",
    "BUILD_SPEC_OUTPUT",
    "BUILD_STATUS_MODULE",
    "BUILD_VALIDATE_FORMAT",
    "COMPLETIONS_SHELL",
    "CONFIG_FORMAT",
    "CONFIG_SOURCE",
    "CONFIG_TARGET",
    "DATASETS_DIFF_AGAINST_REF",
    "DATASETS_DIFF_BASELINE",
    "DATASETS_DIFF_BASELINE_PATH",
    "DATASETS_DIFF_OUTPUT",
    "DATASETS_DOCS_VIEW",
    "DATASETS_MAX_DESCRIPTION",
    "DATASETS_READ_ONLY",
    "DATASETS_SAMPLING",
    "DATASETS_SCAFFOLD_DRY_RUN",
    "DATASETS_SCAFFOLD_NAME",
    "DATASETS_SCAFFOLD_REGISTRY_CHECK",
    "DATASETS_SCHEMA_DIR",
    "DATASETS_SNAPSHOT_OUTPUT",
    "DATASET_TABLE_KEY",
    "DOCS_BUILD_DIR",
    "DOCS_COMMIT",
    "DOCS_DATASET",
    "DOCS_DB_PATH",
    "DOCS_DOCUMENT_OUTPUT_DIR",
    "DOCS_DRY_RUN",
    "DOCS_NX_BACKEND",
    "DOCS_NX_GPU_MODE",
    "DOCS_PREREQ_MODE",
    "DOCS_REPO",
    "DOCS_REPO_ROOT",
    "DOCS_RUN_MODE",
    "DOCS_SCHEMA",
    "DOCS_SKIP_PREREQS",
    "DOCS_VALIDATE",
    "DOCS_VALIDATION_MODE",
    "GRAPH_DEPENDENCY_POLICY",
    "GRAPH_NAMES",
    "GRAPH_PLAN",
    "GRAPH_SELECTION_POLICY",
    "HELP_BY_GROUP",
    "HELP_OPERATION_ID",
    "HELP_QUERY",
    "HISTORY_COMMITS",
    "HISTORY_DB_DIR",
    "HISTORY_ENTITY_KIND",
    "HISTORY_MAX_ENTITIES",
    "HISTORY_OUTPUT_DB",
    "HISTORY_REPO",
    "HISTORY_REPO_ROOT",
    "HISTORY_SELECTION_STRATEGY",
    "JOBS_JOB_ID",
    "JOBS_LIMIT",
    "JOBS_MAX_AGE_DAYS",
    "JOBS_STATUS_FILTER",
    "JSON_FLAG",
    "OUTPUT_FORMAT",
    "PLUGINS_NAME",
    "PLUGINS_OUTPUT_DIR",
    "PLUGINS_PATH",
    "PROJECT_ROOT",
    "SERVE_HOST",
    "SERVE_PORT",
    "SERVE_RELOAD",
    "SHARED_FLAGS",
    "STORAGE_DB_PATH",
    "STORAGE_INCLUDE_VIEWS",
    "STORAGE_INPUT_DIR",
    "STORAGE_OUTPUT_DIR",
    "STORAGE_VALIDATION_MODE",
    "VERBOSE",
]
