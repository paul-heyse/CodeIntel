"""Native pipeline runner with retry support.

This module provides native orchestration for the CodeIntel pipeline,
using the StepRegistry and tenacity-based RetryPolicy infrastructure.

Functions
---------
- run_pipeline_with_retries: Execute pipeline steps with per-step retries
- run_history_timeseries: Execute history timeseries analytics
- run_export_docs: Create views and export artifacts
- build_pipeline_context: Construct context from ExportArgs
- close_gateways: Close cached gateways
- gateway_cache_stats: Return cache statistics
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from codeintel.analytics.history import compute_history_timeseries_gateways
from codeintel.config import ConfigBuilder, GraphRunScope, ScanProfiles, SnapshotRef
from codeintel.config.models import ToolsConfig
from codeintel.config.parser_types import FunctionParserKind
from codeintel.config.primitives import BuildPaths, GraphBackendConfig
from codeintel.core.execution.retry import PLUGIN_RETRY_POLICY, RetryPolicy, with_retry
from codeintel.graphs.engine.backend import maybe_enable_nx_gpu
from codeintel.ingestion.engine.infrastructure import ToolRunner
from codeintel.ingestion.engine.service import ToolService
from codeintel.ingestion.infrastructure.scanning import (
    ScanProfile,
    default_code_profile,
    default_config_profile,
    profile_from_env,
)

# Runtime imports that don't cause circular dependencies
from codeintel.pipeline.execution.context import PipelineContext
from codeintel.pipeline.export.export_jsonl import ExportCallOptions
from codeintel.pipeline.export.runner import ExportOptions, ExportRunner, run_validated_exports
from codeintel.serving.backend.datasets import validate_dataset_registry
from codeintel.storage.gateway import (
    StorageConfig,
    StorageGateway,
    build_snapshot_gateway_resolver,
    open_gateway,
)
from codeintel.storage.views import create_all_views

if TYPE_CHECKING:
    from codeintel.pipeline.steps.registry import StepRegistry

log = logging.getLogger(__name__)

_GATEWAY_CACHE: dict[
    tuple[str, str, bool, bool, bool, bool, bool],
    StorageGateway,
] = {}
_GATEWAY_STATS: dict[str, int] = {"opens": 0, "hits": 0}
DEFAULT_BUILD_SUBDIR = Path("build")


# -----------------------------------------------------------------------------
# Export Arguments
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportArgs:
    """Configuration for pipeline execution.

    Parameters
    ----------
    repo_root
        Path to the repository root directory.
    repo
        Repository identifier (e.g., "org/repo").
    commit
        Commit SHA to process.
    db_path
        Path to the DuckDB database file.
    build_dir
        Build directory for intermediate artifacts.
    serve_db_path
        Optional path to the serving database.
    log_db_path
        Optional path to the log database.
    tools
        Optional tools configuration.
    code_profile
        Optional code scan profile.
    config_profile
        Optional config scan profile.
    function_fail_on_missing_spans
        Fail when function spans are missing.
    function_parser
        Parser selector for function analytics.
    validate_exports
        Whether to validate exports.
    export_schemas
        Schemas to export.
    export_datasets
        Datasets to export.
    export_validation_profile
        Validation profile (strict or lenient).
    force_full_export
        Force full export even when incremental markers match.
    history_commits
        Commits to include in history timeseries.
    history_db_dir
        Directory containing per-commit DuckDB snapshots.
    graph_backend
        Graph backend configuration.
    graph_scope
        Graph run scope filtering.
    """

    repo_root: Path
    repo: str
    commit: str
    db_path: Path
    build_dir: Path
    serve_db_path: Path | None = None
    log_db_path: Path | None = None
    tools: ToolsConfig | None = None
    code_profile: ScanProfile | None = None
    config_profile: ScanProfile | None = None
    function_fail_on_missing_spans: bool = False
    function_parser: FunctionParserKind | None = None
    validate_exports: bool = False
    export_schemas: list[str] | None = None
    export_datasets: tuple[str, ...] | None = None
    export_validation_profile: Literal["strict", "lenient"] | None = None
    force_full_export: bool = False
    history_commits: tuple[str, ...] | None = None
    history_db_dir: Path | None = None
    graph_backend: GraphBackendConfig | None = None
    graph_scope: GraphRunScope | None = None

    def snapshot_config(self) -> SnapshotRef:
        """Build a snapshot configuration from the provided arguments.

        Returns
        -------
        SnapshotRef
            Normalized snapshot descriptor.
        """
        return SnapshotRef(repo_root=self.repo_root, repo=self.repo, commit=self.commit)

    def resolved_tools(self) -> ToolsConfig:
        """Return tools configuration with environment defaults applied.

        Returns
        -------
        ToolsConfig
            Tools configuration with environment overrides applied.
        """
        return _tools_from_env(self.tools)

    def resolved_profiles(self) -> ScanProfiles:
        """Return code/config scan profiles with env overrides applied.

        Returns
        -------
        ScanProfiles
            Resolved code and config scan profiles.
        """
        code_profile = self.code_profile or profile_from_env(default_code_profile(self.repo_root))
        config_profile = self.config_profile or profile_from_env(
            default_config_profile(self.repo_root)
        )
        return ScanProfiles(code=code_profile, config=config_profile)

    def resolved_graph_backend(self) -> GraphBackendConfig:
        """Return the graph backend configuration.

        Returns
        -------
        GraphBackendConfig
            Graph backend settings with defaults applied.
        """
        return self.graph_backend or GraphBackendConfig()

    def storage_config(self) -> StorageConfig:
        """Return an ingest-capable storage configuration.

        Returns
        -------
        StorageConfig
            Gateway configuration for ingest mode.
        """
        return StorageConfig.for_ingest(self.db_path, history_db_path=self.history_db_dir)

    def build_paths(self, *, db_path: Path | None = None) -> BuildPaths:
        """Derive build paths for the current snapshot/execution pair.

        Parameters
        ----------
        db_path
            Optional override for database path.

        Returns
        -------
        BuildPaths
            Normalized build paths anchored to repo_root/build.
        """
        return BuildPaths.from_layout(
            repo_root=self.repo_root,
            build_dir=self.build_dir,
            db_path=db_path or self.db_path,
            document_output_dir=self.repo_root / "Document Output",
            log_db_path=self.log_db_path,
        )


# -----------------------------------------------------------------------------
# Gateway Caching
# -----------------------------------------------------------------------------


def _gateway_cache_key(config: StorageConfig) -> tuple[str, str, bool, bool, bool, bool, bool]:
    """Generate a cache key for a storage configuration.

    Parameters
    ----------
    config
        Storage configuration to generate key for.

    Returns
    -------
    tuple[str, str, bool, bool, bool, bool, bool]
        Cache key tuple.
    """
    history = str(config.history_db_path.resolve()) if config.history_db_path is not None else ""
    return (
        str(config.db_path.resolve()),
        history,
        config.read_only,
        config.apply_schema,
        config.ensure_views,
        config.validate_schema,
        config.attach_history,
    )


def _get_gateway(config: StorageConfig) -> StorageGateway:
    """Return a cached StorageGateway for the pipeline run.

    Parameters
    ----------
    config
        Storage configuration for the gateway.

    Returns
    -------
    StorageGateway
        Cached gateway bound to the provided configuration.
    """
    key = _gateway_cache_key(config)
    cached = _GATEWAY_CACHE.get(key)
    if cached is not None:
        _GATEWAY_STATS["hits"] += 1
        return cached
    gateway = open_gateway(config)
    _GATEWAY_STATS["opens"] += 1
    _GATEWAY_CACHE[key] = gateway
    return gateway


def close_gateways() -> None:
    """Close and clear any cached gateways."""
    for gateway in _GATEWAY_CACHE.values():
        gateway.close()
    _GATEWAY_CACHE.clear()
    _GATEWAY_STATS["opens"] = 0
    _GATEWAY_STATS["hits"] = 0


def gateway_cache_stats() -> dict[str, int]:
    """Return cache statistics for gateway reuse.

    Returns
    -------
    dict[str, int]
        Dictionary containing opens, hits, and current cache size.
    """
    return {
        "opens": _GATEWAY_STATS["opens"],
        "hits": _GATEWAY_STATS["hits"],
        "size": len(_GATEWAY_CACHE),
    }


# -----------------------------------------------------------------------------
# Tools Configuration
# -----------------------------------------------------------------------------


def _tools_from_env(base: ToolsConfig | None = None) -> ToolsConfig:
    """Build a ToolsConfig applying environment overrides when present.

    Parameters
    ----------
    base
        Optional base configuration to extend.

    Returns
    -------
    ToolsConfig
        Tools configuration with environment overrides applied.
    """
    data = base.model_dump() if base is not None else {}
    env_map = {
        "CODEINTEL_SCIP_PYTHON_BIN": "scip_python_bin",
        "CODEINTEL_SCIP_BIN": "scip_bin",
        "CODEINTEL_PYRIGHT_BIN": "pyright_bin",
        "CODEINTEL_PYREFLY_BIN": "pyrefly_bin",
        "CODEINTEL_RUFF_BIN": "ruff_bin",
        "CODEINTEL_COVERAGE_BIN": "coverage_bin",
        "CODEINTEL_PYTEST_BIN": "pytest_bin",
        "CODEINTEL_GIT_BIN": "git_bin",
        "CODEINTEL_COVERAGE_FILE": "coverage_file",
        "CODEINTEL_PYTEST_REPORT": "pytest_report_path",
    }
    for env_var, field in env_map.items():
        value = os.getenv(env_var)
        if value:
            data[field] = value
    return ToolsConfig.model_validate(data)


# -----------------------------------------------------------------------------
# Pipeline Context Building
# -----------------------------------------------------------------------------


def build_pipeline_context(
    args: ExportArgs, *, graph_backend: GraphBackendConfig | None = None
) -> PipelineContext:
    """Construct a PipelineContext from export arguments.

    Parameters
    ----------
    args
        Export arguments containing repo, commit, and path configuration.
    graph_backend
        Optional graph backend configuration override.

    Returns
    -------
    PipelineContext
        Context ready for pipeline execution.
    """
    resolved_backend = graph_backend or args.resolved_graph_backend()
    snapshot = args.snapshot_config()
    tools_cfg = args.resolved_tools()
    profiles = args.resolved_profiles()
    storage_config = args.storage_config()
    paths = args.build_paths(db_path=storage_config.db_path)
    gateway = _get_gateway(storage_config)
    tool_runner = ToolRunner(
        tools_config=tools_cfg,
        cache_dir=paths.tool_cache,
    )
    tool_service = ToolService(runner=tool_runner, tools_config=tools_cfg)
    extra: dict[str, object] = {}
    if args.history_commits:
        extra["history_commits"] = args.history_commits
    return PipelineContext(
        snapshot=snapshot,
        paths=paths,
        gateway=gateway,
        tool_runner=tool_runner,
        tool_service=tool_service,
        tools=tools_cfg,
        code_profile_cfg=profiles.code,
        config_profile_cfg=profiles.config,
        graph_backend_cfg=resolved_backend,
        function_fail_on_missing_spans=args.function_fail_on_missing_spans,
        function_parser=args.function_parser,
        extra=extra,
        export_datasets=args.export_datasets,
        export_validation_profile=args.export_validation_profile,
        force_full_export=args.force_full_export,
        graph_scope=args.graph_scope,
    )


# -----------------------------------------------------------------------------
# Pipeline Execution with Retries
# -----------------------------------------------------------------------------


def run_pipeline_with_retries(
    ctx: PipelineContext,
    registry: StepRegistry,
    *,
    selected_steps: Sequence[str] | None = None,
    retry_policy: RetryPolicy | None = None,
) -> None:
    """Execute pipeline steps with per-step retry support.

    Parameters
    ----------
    ctx
        PipelineContext containing configs and runtime services.
    registry
        StepRegistry containing all pipeline steps.
    selected_steps
        Optional subset of steps to execute; dependencies are included automatically.
    retry_policy
        Optional retry policy; defaults to PLUGIN_RETRY_POLICY.
    """
    policy = retry_policy or PLUGIN_RETRY_POLICY
    step_names = tuple(selected_steps) if selected_steps else registry.list_all_names()

    # Expand with dependencies
    expanded = registry.expand_with_deps(step_names)

    # Preserve sequence order for expanded steps
    sequence = registry.list_all_names()
    ordered_names = [name for name in sequence if name in expanded]

    # Topological sort
    ordered = registry.topological_order(tuple(ordered_names))

    # Execute steps with retry
    for name in ordered:
        step = registry[name]
        log.info("Executing step: %s", name)
        with_retry(policy, step.run, ctx)


def run_full_pipeline(
    args: ExportArgs,
    targets: Sequence[str] | None = None,
    *,
    registry: StepRegistry | None = None,
    retry_policy: RetryPolicy | None = None,
) -> None:
    """Run the full CodeIntel pipeline.

    This is the main entry point for pipeline execution, handling context
    setup, GPU enablement, and cleanup.

    Parameters
    ----------
    args
        Export arguments containing repo, commit, and path configuration.
    targets
        Optional subset of steps to execute; dependencies are included automatically.
    registry
        Optional step registry; defaults to the global REGISTRY from steps module.
    retry_policy
        Optional retry policy for step execution.
    """
    # Lazy import to avoid circular dependency - steps.py imports from runner.py
    if registry is None:
        from codeintel.pipeline import steps as steps_module

        registry = steps_module.REGISTRY

    graph_backend = args.resolved_graph_backend()
    maybe_enable_nx_gpu(graph_backend)

    ctx = build_pipeline_context(args, graph_backend=graph_backend)
    selected = tuple(targets) if targets is not None else None
    try:
        log.info("Starting pipeline for %s@%s", ctx.repo, ctx.commit)
        run_pipeline_with_retries(ctx, registry, selected_steps=selected, retry_policy=retry_policy)
        log.info("Pipeline complete for %s@%s", ctx.repo, ctx.commit)
    finally:
        close_gateways()


# -----------------------------------------------------------------------------
# History Timeseries
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class HistoryTimeseriesParams:
    """Parameters for history timeseries execution.

    Parameters
    ----------
    repo_root
        Path to the repository root.
    repo
        Repository identifier.
    commits
        Tuple of commit SHAs to process.
    history_db_dir
        Directory containing per-commit DuckDB snapshots.
    db_path
        Path to output database.
    runner
        Optional tool runner for git operations.
    """

    repo_root: Path
    repo: str
    commits: tuple[str, ...]
    history_db_dir: Path
    db_path: Path
    runner: ToolRunner | None = None


def run_history_timeseries(params: HistoryTimeseriesParams) -> None:
    """Execute history timeseries analytics across provided commits.

    Parameters
    ----------
    params
        History timeseries execution parameters.
    """
    snapshot = SnapshotRef(repo_root=params.repo_root, repo=params.repo, commit=params.commits[0])
    paths = BuildPaths.from_layout(
        repo_root=params.repo_root,
        build_dir=params.repo_root / DEFAULT_BUILD_SUBDIR,
        db_path=params.db_path,
    )
    builder = ConfigBuilder.from_primitives(snapshot=snapshot, paths=paths)
    cfg = builder.history_timeseries(commits=params.commits)
    # Note: Don't use history_db_path here - the snapshot_resolver handles
    # loading individual snapshot DBs from the directory. attach_history is
    # for attaching a single history DB file, not a directory of snapshots.
    gateway = _get_gateway(StorageConfig.for_ingest(params.db_path))
    snapshot_resolver = build_snapshot_gateway_resolver(
        db_dir=params.history_db_dir,
        repo=params.repo,
        primary_gateway=gateway,
    )
    compute_history_timeseries_gateways(
        gateway,
        cfg,
        snapshot_resolver,
        runner=params.runner,
    )


# -----------------------------------------------------------------------------
# Export Docs
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportHooks:
    """Override hooks for export docs execution.

    Parameters
    ----------
    validator
        Function to validate dataset registry.
    export_runner
        Function to run exports.
    gateway_factory
        Function to create storage gateway.
    create_views
        Function to create database views.
    """

    validator: Callable[[StorageGateway], None] = validate_dataset_registry
    export_runner: ExportRunner = run_validated_exports
    gateway_factory: Callable[[Path], StorageGateway] = lambda db_path: _get_gateway(
        StorageConfig.for_ingest(db_path)
    )
    create_views: Callable[[Any], None] = create_all_views


def run_export_docs(
    *,
    db_path: Path,
    document_output_dir: Path,
    options: ExportOptions | None = None,
    hooks: ExportHooks | None = None,
) -> None:
    """Create views and export Parquet/JSONL artifacts.

    Parameters
    ----------
    db_path
        Path to the DuckDB database.
    document_output_dir
        Directory for exported artifacts.
    options
        Export options configuration.
    hooks
        Override hooks for customizing export behavior.
    """
    resolved_hooks = hooks or ExportHooks()
    export_options = options or ExportOptions(
        export=ExportCallOptions(
            validate_exports=False,
            schemas=None,
            datasets=None,
            validation_profile=None,
            force_full_export=False,
        )
    )
    export_options = replace(export_options, validator=resolved_hooks.validator)
    gateway = resolved_hooks.gateway_factory(db_path)
    resolved_hooks.create_views(gateway.con)
    resolved_hooks.export_runner(
        gateway=gateway,
        output_dir=document_output_dir,
        options=export_options,
    )


__all__ = [
    "DEFAULT_BUILD_SUBDIR",
    "ExportArgs",
    "ExportHooks",
    "HistoryTimeseriesParams",
    "build_pipeline_context",
    "close_gateways",
    "gateway_cache_stats",
    "run_export_docs",
    "run_full_pipeline",
    "run_history_timeseries",
    "run_pipeline_with_retries",
]
