"""Prefect 3 flow wrapping the CodeIntel pipeline."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prefect import flow, get_run_logger, task
from prefect.logging.configuration import setup_logging
from prefect.logging.handlers import PrefectConsoleHandler

from codeintel.analytics.history import compute_history_timeseries_gateways
from codeintel.cli.nx_backend import maybe_enable_nx_gpu
from codeintel.config import ConfigBuilder, ScanProfiles, SnapshotRef
from codeintel.config.models import ToolsConfig
from codeintel.config.parser_types import FunctionParserKind
from codeintel.config.primitives import BuildPaths, GraphBackendConfig
from codeintel.ingestion.source_scanner import (
    ScanProfile,
    default_code_profile,
    default_config_profile,
    profile_from_env,
)
from codeintel.ingestion.tool_runner import ToolRunner
from codeintel.ingestion.tool_service import ToolService
from codeintel.pipeline.export.export_jsonl import ExportCallOptions
from codeintel.pipeline.export.runner import ExportOptions, ExportRunner, run_validated_exports
from codeintel.pipeline.orchestration.steps import PipelineContext, run_pipeline
from codeintel.serving.http.datasets import validate_dataset_registry
from codeintel.storage.gateway import (
    StorageConfig,
    StorageGateway,
    build_snapshot_gateway_resolver,
    open_gateway,
)
from codeintel.storage.views import create_all_views

log = logging.getLogger(__name__)
_GATEWAY_CACHE: dict[
    tuple[str, str, bool, bool, bool, bool, bool],
    StorageGateway,
] = {}
_GATEWAY_STATS: dict[str, int] = {"opens": 0, "hits": 0}
_PREFECT_LOGGING_SETTINGS_PATH = Path(__file__).with_name("prefect_logging.yml")
os.environ.setdefault("PREFECT_LOGGING_SETTINGS_PATH", str(_PREFECT_LOGGING_SETTINGS_PATH))
DEFAULT_BUILD_SUBDIR = Path("build")


@dataclass(frozen=True)
class ExportArgs:
    """Inputs for the export_docs Prefect flow."""

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
    history_commits: tuple[str, ...] | None = None
    history_db_dir: Path | None = None
    graph_backend: GraphBackendConfig | None = None

    def snapshot_config(self) -> SnapshotRef:
        """
        Build a snapshot configuration from the provided arguments.

        Returns
        -------
        SnapshotRef
            Normalized snapshot descriptor for the flow.
        """
        return SnapshotRef(repo_root=self.repo_root, repo=self.repo, commit=self.commit)

    def resolved_tools(self) -> ToolsConfig:
        """
        Return tools configuration with environment defaults applied.

        Returns
        -------
        ToolsConfig
            Tools configuration with environment overrides applied.
        """
        return _tools_from_env(self.tools)

    def resolved_profiles(self) -> ScanProfiles:
        """
        Return code/config scan profiles with env overrides applied.

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
        """
        Return the graph backend configuration.

        Returns
        -------
        GraphBackendConfig
            Graph backend settings with defaults applied.
        """
        return self.graph_backend or GraphBackendConfig()

    def storage_config(self) -> StorageConfig:
        """
        Return an ingest-capable storage configuration.

        Returns
        -------
        StorageConfig
            Gateway configuration for ingest mode.
        """
        return StorageConfig.for_ingest(self.db_path, history_db_path=self.history_db_dir)

    def build_paths(self, *, db_path: Path | None = None) -> BuildPaths:
        """
        Derive build paths for the current snapshot/execution pair.

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


def _build_pipeline_context(
    args: ExportArgs, *, graph_backend: GraphBackendConfig
) -> PipelineContext:
    """
    Construct a PipelineContext from export arguments using consolidated configs.

    Returns
    -------
    PipelineContext
        Context ready for pipeline execution.
    """
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
        graph_backend_cfg=graph_backend,
        function_fail_on_missing_spans=args.function_fail_on_missing_spans,
        function_parser=args.function_parser,
        extra=extra,
        export_datasets=args.export_datasets,
    )


def _gateway_cache_key(config: StorageConfig) -> tuple[str, str, bool, bool, bool, bool, bool]:
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
    """
    Return a cached StorageGateway for the flow run.

    Returns
    -------
    StorageGateway
        Cached gateway bound to the provided db_path.
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


def _close_gateways() -> None:
    """Close and clear any cached gateways."""
    for gateway in _GATEWAY_CACHE.values():
        gateway.close()
    _GATEWAY_CACHE.clear()
    _GATEWAY_STATS["opens"] = 0
    _GATEWAY_STATS["hits"] = 0


def gateway_cache_stats() -> dict[str, int]:
    """
    Return cache statistics for flow gateway reuse.

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


_PREFECT_LOGGING_CONFIGURED = False


def _configure_prefect_logging() -> None:
    """Configure Prefect logging to use a simple stderr handler instead of Rich console."""
    global _PREFECT_LOGGING_CONFIGURED  # noqa: PLW0603
    if _PREFECT_LOGGING_CONFIGURED:
        return
    os.environ["PREFECT_LOGGING_SETTINGS_PATH"] = str(_PREFECT_LOGGING_SETTINGS_PATH)
    setup_logging(incremental=False)
    stderr_handler = logging.StreamHandler(sys.__stderr__)
    stderr_handler.setLevel(logging.INFO)
    stderr_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    for logger_name in ("prefect", "prefect.server", "prefect.server.api.server"):
        logger = logging.getLogger(logger_name)
        logger.handlers = [h for h in logger.handlers if not isinstance(h, PrefectConsoleHandler)]
        logger.handlers = []
        logger.addHandler(stderr_handler)
        logger.propagate = False
    root_logger = logging.getLogger()
    root_logger.handlers = [
        h for h in root_logger.handlers if not isinstance(h, PrefectConsoleHandler)
    ]
    root_logger.addHandler(stderr_handler)
    for logger in logging.Logger.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            logger.handlers = [
                h for h in logger.handlers if not isinstance(h, PrefectConsoleHandler)
            ]
            if stderr_handler not in logger.handlers:
                logger.addHandler(stderr_handler)
    _PREFECT_LOGGING_CONFIGURED = True


def _tools_from_env(base: ToolsConfig | None = None) -> ToolsConfig:
    """
    Build a ToolsConfig applying environment overrides when present.

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


@dataclass(frozen=True)
class HistoryTimeseriesTaskParams:
    """Parameters for the history_timeseries Prefect task."""

    repo_root: Path
    repo: str
    commits: tuple[str, ...]
    history_db_dir: Path
    db_path: Path
    runner: ToolRunner | None = None


@task(name="history_timeseries", retries=1, retry_delay_seconds=2)
def t_history_timeseries(params: HistoryTimeseriesTaskParams) -> None:
    """Execute history_timeseries analytics across provided commits."""
    snapshot = SnapshotRef(repo_root=params.repo_root, repo=params.repo, commit=params.commits[0])
    paths = BuildPaths.from_layout(
        repo_root=params.repo_root,
        build_dir=params.repo_root / DEFAULT_BUILD_SUBDIR,
        db_path=params.db_path,
    )
    builder = ConfigBuilder.from_primitives(snapshot=snapshot, paths=paths)
    cfg = builder.history_timeseries(commits=params.commits)
    gateway = _get_gateway(
        StorageConfig.for_ingest(params.db_path, history_db_path=params.history_db_dir)
    )
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


@dataclass(frozen=True)
class ExportTaskHooks:
    """Override hooks for export_docs task wiring."""

    validator: Callable[[StorageGateway], None] = validate_dataset_registry
    export_runner: ExportRunner = run_validated_exports
    gateway_factory: Callable[[Path], StorageGateway] = lambda db_path: _get_gateway(
        StorageConfig.for_ingest(db_path)
    )
    create_views: Callable[[Any], None] = create_all_views


def t_export_docs(
    *,
    db_path: Path,
    document_output_dir: Path,
    options: ExportOptions | None = None,
    hooks: ExportTaskHooks | None = None,
) -> None:
    """
    Create views and export Parquet/JSONL artifacts.

    Raises
    ------
    Exception
        Propagates any export runner errors to the caller.
    """
    resolved_hooks = hooks or ExportTaskHooks()
    export_options = options or ExportOptions(
        export=ExportCallOptions(validate_exports=False, schemas=None, datasets=None)
    )
    gateway = resolved_hooks.gateway_factory(db_path)
    resolved_hooks.create_views(gateway.con)
    resolved_hooks.validator(gateway)
    resolved_hooks.export_runner(
        gateway=gateway,
        output_dir=document_output_dir,
        options=export_options,
    )


t_export_docs.fn = t_export_docs  # type: ignore[attr-defined]
t_history_timeseries.fn = t_history_timeseries  # type: ignore[attr-defined]


@flow(name="export_docs_flow")
def export_docs_flow(
    args: ExportArgs,
    targets: Iterable[str] | None = None,
) -> None:
    """Run the CodeIntel pipeline within a Prefect flow."""
    _configure_prefect_logging()
    run_logger = get_run_logger()
    graph_backend = args.resolved_graph_backend()
    maybe_enable_nx_gpu(graph_backend)

    ctx = _build_pipeline_context(args, graph_backend=graph_backend)
    selected = tuple(targets) if targets is not None else None
    try:
        run_logger.info("Starting pipeline for %s@%s", ctx.repo, ctx.commit)
        run_pipeline(ctx, selected_steps=selected)
        run_logger.info("Pipeline complete for %s@%s", ctx.repo, ctx.commit)
    finally:
        _close_gateways()


__all__ = [
    "ExportArgs",
    "ExportTaskHooks",
    "HistoryTimeseriesTaskParams",
    "export_docs_flow",
    "gateway_cache_stats",
    "t_export_docs",
    "t_history_timeseries",
]
