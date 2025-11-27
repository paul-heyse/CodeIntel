"""CLI entrypoint for CodeIntel pipeline and document exports."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable, Iterable
from dataclasses import replace
from pathlib import Path
from typing import Literal, Protocol

from codeintel.analytics.graph_runtime import build_graph_runtime
from codeintel.analytics.history import compute_history_timeseries_gateways
from codeintel.config import ConfigBuilder
from codeintel.config.models import CliPathsInput, CodeIntelConfig, RepoConfig
from codeintel.config.parser_types import FunctionParserKind
from codeintel.config.primitives import GraphBackendConfig, SnapshotRef
from codeintel.graphs.nx_backend import maybe_enable_nx_gpu
from codeintel.ingestion.source_scanner import (
    default_code_profile,
    default_config_profile,
    profile_from_env,
)
from codeintel.ingestion.tool_runner import ToolRunner
from codeintel.pipeline.export.export_jsonl import ExportCallOptions
from codeintel.pipeline.export.runner import (
    ExportOptions,
    ExportRunner,
    run_validated_exports,
)
from codeintel.pipeline.orchestration.prefect_flow import ExportArgs, export_docs_flow
from codeintel.pipeline.orchestration.steps import REGISTRY, StepPhase
from codeintel.serving.http.datasets import validate_dataset_registry
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.services.errors import ExportError, log_problem, problem
from codeintel.storage.gateway import (
    DuckDBError,
    StorageConfig,
    StorageGateway,
    build_snapshot_gateway_resolver,
    open_gateway,
)
from codeintel.storage.metadata_bootstrap import (
    _assert_macro_coverage,
    dataset_rows_only_entries,
    ingest_macro_coverage,
    validate_dataset_schema_registry,
    validate_macro_registry,
    validate_normalized_macro_schemas,
)

LOG = logging.getLogger("codeintel.cli")


class GatewayFactory(Protocol):
    """Factory for creating gateways with optional read-only mode."""

    def __call__(self, cfg: CodeIntelConfig, *, read_only: bool) -> StorageGateway:
        """Create a gateway."""
        ...


CommandHandler = Callable[..., int]


# ---------------------------------------------------------------------------
# Argument parsing / logging setup
# ---------------------------------------------------------------------------


def _setup_logging(verbosity: int) -> None:
    """
    Configure logging based on -v/--verbose count.

    0 -> WARNING, 1 -> INFO, 2+ -> DEBUG.
    """
    if verbosity <= 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _add_common_repo_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--repo-root",
        type=Path,
        default=Path(),
        help="Path to the repository root (default: current directory)",
    )
    p.add_argument(
        "--repo",
        required=True,
        help="Repository slug (used in GOIDs and exports, e.g. 'my-org/my-repo')",
    )
    p.add_argument(
        "--commit",
        required=True,
        help="Commit SHA for this analysis run (embedded into GOIDs)",
    )
    p.add_argument(
        "--db-path",
        type=Path,
        default=Path("build/db/codeintel_prefect.duckdb"),
        help="Path to the DuckDB database (default: build/db/codeintel_prefect.duckdb)",
    )
    p.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build"),
        help="Build directory (default: build/)",
    )
    p.add_argument(
        "--document-output-dir",
        type=Path,
        default=None,
        help="Override Document Output/ directory (default: <repo-root>/Document Output)",
    )


def _add_graph_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--nx-gpu",
        action="store_true",
        help="Prefer GPU backend for NetworkX (nx-cugraph) when available.",
    )
    parser.add_argument(
        "--nx-backend",
        choices=["auto", "cpu", "nx-cugraph"],
        default="auto",
        help="NetworkX backend selection (default: auto).",
    )
    parser.add_argument(
        "--nx-gpu-strict",
        action="store_true",
        help="Fail instead of falling back to CPU if GPU backend cannot be enabled.",
    )


def _add_subsystem_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register subsystem-related subcommands."""
    p_subsys = subparsers.add_parser("subsystem", help="Subsystem exploration helpers")
    subsys_sub = p_subsys.add_subparsers(dest="subcommand", required=True)

    p_subsys_list = subsys_sub.add_parser(
        "list",
        help="List inferred subsystems with role/risk metadata.",
    )
    _add_common_repo_args(p_subsys_list)
    p_subsys_list.add_argument("--role", help="Filter subsystems by role tag")
    p_subsys_list.add_argument("--q", help="Search substring on name/description")
    p_subsys_list.add_argument("--limit", type=int, default=None, help="Limit subsystem count")
    p_subsys_list.set_defaults(func=_cmd_subsystem_list)

    p_subsys_show = subsys_sub.add_parser(
        "show",
        help="Show subsystem detail and modules.",
    )
    _add_common_repo_args(p_subsys_show)
    p_subsys_show.add_argument("--subsystem-id", required=True, help="Subsystem identifier")
    p_subsys_show.set_defaults(func=_cmd_subsystem_show)

    p_subsys_modules = subsys_sub.add_parser(
        "module-memberships",
        help="List subsystem memberships for a module.",
    )
    _add_common_repo_args(p_subsys_modules)
    p_subsys_modules.add_argument("--module", required=True, help="Module name (e.g., pkg.mod)")
    p_subsys_modules.set_defaults(func=_cmd_subsystem_modules)


def _add_pipeline_run_subparser(
    pipeline_sub: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p_run = pipeline_sub.add_parser(
        "run",
        help="Run the full pipeline via Prefect (targets are ignored; runs export_docs_flow).",
    )
    _add_common_repo_args(p_run)
    p_run.add_argument(
        "--target",
        action="append",
        default=None,
        help=(
            "Name of a pipeline step to run (can be specified multiple times). "
            "Defaults to 'export_docs' if omitted."
        ),
    )
    p_run.add_argument(
        "--skip-scip",
        action="store_true",
        help="Skip SCIP ingestion (sets CODEINTEL_SKIP_SCIP=true).",
    )
    p_run.add_argument(
        "--function-fail-on-missing-spans",
        action="store_true",
        help="Fail pipeline when function spans are missing or files cannot be parsed.",
    )
    p_run.add_argument(
        "--function-parser",
        choices=[kind.value for kind in FunctionParserKind],
        help="Optional parser selector for function analytics (e.g., 'python').",
    )
    p_run.add_argument(
        "--history-commit",
        dest="history_commits",
        action="append",
        help="Commit SHA to include in history_timeseries (can be repeated).",
    )
    p_run.add_argument(
        "--history-db-dir",
        type=Path,
        default=Path("build/db"),
        help="Directory containing per-commit DuckDB snapshots for history_timeseries.",
    )
    p_run.add_argument(
        "--export-dataset",
        dest="export_datasets",
        action="append",
        help="Dataset name to export during docs export step (can be repeated).",
    )
    _add_graph_backend_args(p_run)
    p_run.set_defaults(func=_cmd_pipeline_run)


def _add_pipeline_list_steps_subparser(
    pipeline_sub: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p_list_steps = pipeline_sub.add_parser(
        "list-steps",
        help="List all available pipeline steps with descriptions.",
    )
    p_list_steps.add_argument(
        "--phase",
        choices=["ingestion", "graphs", "analytics", "export"],
        help="Filter steps by phase.",
    )
    p_list_steps.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        help="Output as JSON for machine consumption.",
    )
    p_list_steps.set_defaults(func=_cmd_pipeline_list_steps)


def _add_pipeline_deps_subparser(
    pipeline_sub: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p_deps = pipeline_sub.add_parser(
        "deps",
        help="Show dependency tree for a pipeline step.",
    )
    p_deps.add_argument(
        "step_name",
        help="Name of the step to show dependencies for.",
    )
    p_deps.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        help="Output as JSON for machine consumption.",
    )
    p_deps.set_defaults(func=_cmd_pipeline_deps)


def _register_pipeline_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    pipeline_parser = subparsers.add_parser("pipeline", help="Run enrichment pipeline")
    pipeline_sub = pipeline_parser.add_subparsers(dest="subcommand", required=True)
    _add_pipeline_run_subparser(pipeline_sub)
    _add_pipeline_list_steps_subparser(pipeline_sub)
    _add_pipeline_deps_subparser(pipeline_sub)


def _add_docs_export_subparser(
    docs_sub: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p_export = docs_sub.add_parser(
        "export",
        help="Export Parquet + JSONL datasets from DuckDB into Document Output/",
    )
    _add_common_repo_args(p_export)
    p_export.add_argument(
        "--validate",
        dest="validate_exports",
        action="store_true",
        help="Validate exported datasets against JSON Schema definitions.",
    )
    p_export.add_argument(
        "--schema",
        dest="schemas",
        action="append",
        help="Schema name to validate (can be repeated). Defaults to the standard export set.",
    )
    p_export.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        help="Dataset name to export (can be repeated). Defaults to all mapped datasets.",
    )
    p_export.add_argument(
        "--require-normalized-macros",
        action="store_true",
        help="Fail if any requested dataset lacks a normalized macro.",
    )
    _add_graph_backend_args(p_export)
    p_export.set_defaults(func=cmd_docs_export)


def _register_docs_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p_docs = subparsers.add_parser("docs", help="Document Output helpers")
    docs_sub = p_docs.add_subparsers(dest="subcommand", required=True)
    _add_docs_export_subparser(docs_sub)


def _register_storage_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p_storage = subparsers.add_parser("storage", help="Storage validation helpers")
    storage_sub = p_storage.add_subparsers(dest="subcommand", required=True)
    p_validate_macros = storage_sub.add_parser(
        "validate-macros",
        help="Validate macro registry hashes and normalized macro schemas.",
    )
    p_validate_macros.add_argument(
        "--db-path",
        type=Path,
        default=Path("build/db/codeintel_prefect.duckdb"),
        help="Path to the DuckDB database to validate.",
    )
    p_validate_macros.add_argument(
        "--require-ingest-macros",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if any ingest macros are missing (default: True).",
    )
    p_validate_macros.set_defaults(func=_cmd_storage_validate_macros)


def _register_history_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p_history = subparsers.add_parser(
        "history-timeseries",
        help="Aggregate analytics.history_timeseries across commits.",
    )
    p_history.add_argument(
        "--repo-root",
        type=Path,
        default=Path(),
        help="Path to the repository root (default: current directory)",
    )
    p_history.add_argument(
        "--repo",
        required=True,
        help="Repository slug (e.g., 'my-org/my-repo')",
    )
    p_history.add_argument(
        "--commits",
        nargs="+",
        required=True,
        help="Commits to include in the timeseries (latest first).",
    )
    p_history.add_argument(
        "--db-dir",
        type=Path,
        default=Path("build/db"),
        help="Directory with per-commit DuckDB snapshots (codeintel-<commit>.duckdb).",
    )
    p_history.add_argument(
        "--output-db",
        type=Path,
        default=Path("build/db/history.duckdb"),
        help="Destination DuckDB for history_timeseries (will be created if missing).",
    )
    p_history.add_argument(
        "--entity-kind",
        choices=["function", "module", "both"],
        default="function",
        help="Entity kind to include in the history aggregation.",
    )
    p_history.add_argument(
        "--max-entities",
        type=int,
        default=500,
        help="Maximum entities to track (top-N by selection strategy).",
    )
    p_history.add_argument(
        "--selection-strategy",
        default="risk_score",
        help="Selection strategy for picking entities (default: risk_score).",
    )
    p_history.set_defaults(func=_cmd_history_timeseries)


def _register_ide_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p_ide = subparsers.add_parser("ide", help="IDE helper commands")
    ide_sub = p_ide.add_subparsers(dest="subcommand", required=True)

    p_ide_hints = ide_sub.add_parser(
        "hints",
        help="Emit IDE hints (module + subsystem context) for a relative file path.",
    )
    _add_common_repo_args(p_ide_hints)
    p_ide_hints.add_argument(
        "--rel-path",
        required=True,
        help="File path relative to repo root (e.g., pkg/module.py)",
    )
    p_ide_hints.set_defaults(func=_cmd_ide_hints)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codeintel",
        description="CodeIntel metadata pipeline and export CLI",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (can be repeated)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    _register_pipeline_commands(subparsers)
    _register_docs_commands(subparsers)
    _register_storage_commands(subparsers)
    _register_history_commands(subparsers)
    _register_ide_commands(subparsers)
    _add_subsystem_subparser(subparsers)

    return parser


def make_parser() -> argparse.ArgumentParser:
    """
    Public helper to construct the CLI parser (for tests/tools).

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with all subcommands registered.
    """
    return _make_parser()


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def _build_config_from_args(args: argparse.Namespace) -> CodeIntelConfig:
    graph_backend = _build_graph_backend_config(args)
    paths_cfg = CliPathsInput(
        repo_root=args.repo_root,
        build_dir=args.build_dir,
        db_path=args.db_path,
        document_output_dir=args.document_output_dir,
    )
    repo_cfg = RepoConfig(repo=args.repo, commit=args.commit)
    return CodeIntelConfig.from_cli_args(
        repo_cfg=repo_cfg,
        paths_cfg=paths_cfg,
        graph_backend=graph_backend,
    )


def _build_graph_backend_config(args: argparse.Namespace) -> GraphBackendConfig:
    backend_value = str(getattr(args, "nx_backend", "auto"))
    backend: Literal["auto", "cpu", "nx-cugraph"] = "auto"
    if backend_value == "cpu":
        backend = "cpu"
    elif backend_value == "nx-cugraph":
        backend = "nx-cugraph"
    return GraphBackendConfig(
        use_gpu=bool(getattr(args, "nx_gpu", False)),
        backend=backend,
        strict=bool(getattr(args, "nx_gpu_strict", False)),
    )


def _open_gateway(cfg: CodeIntelConfig, *, read_only: bool) -> StorageGateway:
    """
    Open a StorageGateway and ensure schemas/views exist for write mode.

    Parameters
    ----------
    cfg:
        CodeIntel configuration holding paths.
    read_only:
        Whether to open the database read-only.

    Returns
    -------
    StorageGateway
        Gateway bound to the configured DuckDB database.
    """
    cfg.paths.db_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = (
        StorageConfig.for_readonly(cfg.paths.db_path)
        if read_only
        else StorageConfig.for_ingest(cfg.paths.db_path)
    )
    gateway_cfg = replace(
        base_cfg,
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
    )
    return open_gateway(gateway_cfg)


def _build_runtime(cfg: CodeIntelConfig, gateway: StorageGateway):
    """Construct a GraphRuntime for CLI commands."""
    snapshot = SnapshotRef(
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        repo_root=cfg.repo.repo_root,
    )
    return build_graph_runtime(
        gateway,
        snapshot,
        cfg.graph_backend,
    )


def _cmd_pipeline_run(args: argparse.Namespace) -> int:
    cfg = _build_config_from_args(args)
    maybe_enable_nx_gpu(cfg.graph_backend)
    repo_root = cfg.paths.repo_root
    db_path = cfg.paths.db_path
    build_dir = cfg.paths.build_dir
    output_dir = cfg.paths.document_output_dir
    if output_dir is not None:
        os.environ.setdefault("CODEINTEL_OUTPUT_DIR", str(output_dir))

    if args.skip_scip:
        os.environ["CODEINTEL_SKIP_SCIP"] = "true"

    targets = list(args.target) if args.target else None
    LOG.info(
        "Running Prefect export_docs_flow for repo=%s commit=%s targets=%s",
        cfg.repo.repo,
        cfg.repo.commit,
        targets,
    )

    export_docs_flow(
        args=ExportArgs(
            repo_root=repo_root,
            repo=cfg.repo.repo,
            commit=cfg.repo.commit,
            db_path=db_path,
            build_dir=build_dir,
            tools=cfg.tools,
            code_profile=profile_from_env(default_code_profile(repo_root)),
            config_profile=profile_from_env(default_config_profile(repo_root)),
            function_fail_on_missing_spans=args.function_fail_on_missing_spans,
            function_parser=FunctionParserKind(args.function_parser)
            if args.function_parser
            else None,
            history_commits=tuple(args.history_commits) if args.history_commits else None,
            history_db_dir=args.history_db_dir,
            graph_backend=cfg.graph_backend,
            export_datasets=tuple(args.export_datasets) if args.export_datasets else None,
        ),
        targets=targets,
    )
    return 0


def _cmd_pipeline_list_steps(args: argparse.Namespace) -> int:
    """
    List all available pipeline steps with descriptions.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    phase_filter = getattr(args, "phase", None)
    output_json = getattr(args, "output_json", False)

    if phase_filter:
        phase = StepPhase(phase_filter)
        steps = REGISTRY.list_by_phase(phase)
    else:
        steps = REGISTRY.list_all()

    if output_json:
        data = [
            {
                "name": meta.name,
                "description": meta.description,
                "phase": meta.phase.value,
                "deps": list(meta.deps),
            }
            for meta in steps
        ]
        sys.stdout.write(json.dumps(data, indent=2))
        sys.stdout.write("\n")
    else:
        for meta in steps:
            deps_str = ", ".join(meta.deps) if meta.deps else "(none)"
            sys.stdout.write(f"{meta.name} [{meta.phase.value}]\n")
            sys.stdout.write(f"  {meta.description}\n")
            sys.stdout.write(f"  deps: {deps_str}\n")
            sys.stdout.write("\n")
    return 0


def _cmd_pipeline_deps(args: argparse.Namespace) -> int:
    """
    Show dependency tree for a pipeline step.

    Returns
    -------
    int
        Exit code (0 on success, 1 if step not found).
    """
    step_name = args.step_name
    output_json = getattr(args, "output_json", False)

    if step_name not in REGISTRY:
        LOG.error("Unknown step: %s", step_name)
        return 1

    # Get all transitive dependencies
    expanded = REGISTRY.expand_with_deps([step_name])
    expanded.discard(step_name)  # Remove the step itself from deps

    # Get direct deps
    direct_deps = tuple(REGISTRY.get_deps(step_name))

    if output_json:
        data = {
            "step": step_name,
            "direct_deps": list(direct_deps),
            "transitive_deps": sorted(expanded),
        }
        sys.stdout.write(json.dumps(data, indent=2))
        sys.stdout.write("\n")
    else:
        step = REGISTRY[step_name]
        sys.stdout.write(f"Step: {step_name}\n")
        sys.stdout.write(f"Description: {step.description}\n")
        sys.stdout.write(f"Phase: {step.phase.value}\n")
        sys.stdout.write("\n")
        sys.stdout.write(f"Direct dependencies ({len(direct_deps)}):\n")
        if direct_deps:
            for dep in direct_deps:
                sys.stdout.write(f"  - {dep}\n")
        else:
            sys.stdout.write("  (none)\n")
        sys.stdout.write("\n")
        sys.stdout.write(f"All transitive dependencies ({len(expanded)}):\n")
        if expanded:
            for dep in sorted(expanded):
                sys.stdout.write(f"  - {dep}\n")
        else:
            sys.stdout.write("  (none)\n")
    return 0


def _cmd_ide_hints(args: argparse.Namespace) -> int:
    cfg = _build_config_from_args(args)
    gateway = _open_gateway(cfg, read_only=True)
    runtime = _build_runtime(cfg, gateway)
    engine = runtime.engine
    backend = DuckDBBackend(
        gateway=gateway,
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        query_engine=engine,
    )
    response = backend.get_file_hints(rel_path=args.rel_path)
    if not response.found or not response.hints:
        LOG.error("No IDE hints found for %s", args.rel_path)
        return 1

    payload = {
        "rel_path": args.rel_path,
        "hints": [hint.model_dump() for hint in response.hints],
        "meta": response.meta.model_dump(),
    }
    sys.stdout.write(json.dumps(payload))
    sys.stdout.write("\n")
    return 0


def _cmd_subsystem_list(args: argparse.Namespace) -> int:
    cfg = _build_config_from_args(args)
    gateway = _open_gateway(cfg, read_only=True)
    runtime = _build_runtime(cfg, gateway)
    engine = runtime.engine
    backend = DuckDBBackend(
        gateway=gateway,
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        query_engine=engine,
    )
    response = backend.list_subsystems(
        limit=args.limit,
        role=args.role,
        q=args.q,
    )
    payload = {
        "subsystems": [row.model_dump() for row in response.subsystems],
        "meta": response.meta.model_dump(),
    }
    sys.stdout.write(json.dumps(payload))
    sys.stdout.write("\n")
    return 0


def _cmd_subsystem_show(args: argparse.Namespace) -> int:
    cfg = _build_config_from_args(args)
    gateway = _open_gateway(cfg, read_only=True)
    runtime = _build_runtime(cfg, gateway)
    engine = runtime.engine
    backend = DuckDBBackend(
        gateway=gateway,
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        query_engine=engine,
    )
    response = backend.get_subsystem_modules(subsystem_id=args.subsystem_id)
    if not response.found or response.subsystem is None:
        LOG.error("Subsystem not found: %s", args.subsystem_id)
        return 1
    payload = {
        "subsystem": response.subsystem.model_dump(),
        "modules": [row.model_dump() for row in response.modules],
        "meta": response.meta.model_dump(),
    }
    sys.stdout.write(json.dumps(payload))
    sys.stdout.write("\n")
    return 0


def _cmd_subsystem_modules(args: argparse.Namespace) -> int:
    cfg = _build_config_from_args(args)
    gateway = _open_gateway(cfg, read_only=True)
    runtime = _build_runtime(cfg, gateway)
    engine = runtime.engine
    backend = DuckDBBackend(
        gateway=gateway,
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        query_engine=engine,
    )
    response = backend.get_module_subsystems(module=args.module)
    payload = {
        "found": response.found,
        "memberships": [row.model_dump() for row in response.memberships],
        "meta": response.meta.model_dump(),
    }
    sys.stdout.write(json.dumps(payload))
    sys.stdout.write("\n")
    return 0


def _cmd_storage_validate_macros(args: argparse.Namespace) -> int:
    """
    Validate macro registry hashes and normalized macro schemas for a database.

    Returns
    -------
    int
        Exit code indicating success (0) or failure (non-zero).
    """
    cfg = StorageConfig.for_ingest(args.db_path)
    gateway = open_gateway(cfg)
    missing_ingest: list[str] = []
    error: RuntimeError | None = None
    try:
        _assert_macro_coverage()
        validate_macro_registry(gateway.con)
        validate_dataset_schema_registry(gateway.con)
        validate_normalized_macro_schemas(gateway.con)
        missing_ingest, present_ingest = ingest_macro_coverage(gateway.con)
        if missing_ingest:
            LOG.warning("Missing ingest macros: %s", ", ".join(missing_ingest))
        LOG.debug("Present ingest macros: %s", ", ".join(present_ingest))
        if args.require_ingest_macros and missing_ingest:
            message = ", ".join(missing_ingest)
            error = RuntimeError(f"Ingest macros missing: {message}")
    except RuntimeError as exc:
        error = exc
    if error is not None:
        LOG.error("Macro validation failed", exc_info=error)
        gateway.close()
        return 1
    dataset_rows_list = dataset_rows_only_entries()
    if dataset_rows_list:
        LOG.info(
            "dataset_rows-only datasets (no normalized macro): %s",
            ", ".join(dataset_rows_list),
        )
    gateway.close()
    LOG.info("Macro validation passed.")
    return 0


def cmd_docs_export(
    args: argparse.Namespace,
    *,
    validator: Callable[[StorageGateway], None] = validate_dataset_registry,
    export_runner: ExportRunner = run_validated_exports,
    gateway_factory: GatewayFactory | None = None,
) -> int:
    """
    Export only, assuming the pipeline has already populated the DuckDB db.

    This is what `scripts/generate_documents.sh` should call:

        codeintel docs export --repo <slug> --commit <sha>

    Returns
    -------
    int
        Exit code (0 on success).

    Raises
    ------
    RuntimeError
        If document_output_dir could not be resolved.
    """
    cfg = _build_config_from_args(args)
    maybe_enable_nx_gpu(cfg.graph_backend)
    gateway = (gateway_factory or _open_gateway)(cfg, read_only=True)

    out_dir = cfg.paths.document_output_dir
    if out_dir is None:
        message = "document_output_dir was not resolved"
        raise RuntimeError(message)
    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Exporting Parquet + JSONL datasets into %s", out_dir)
    schemas = list(args.schemas) if getattr(args, "schemas", None) else None
    datasets = list(args.datasets) if getattr(args, "datasets", None) else None
    try:
        export_runner(
            gateway=gateway,
            output_dir=out_dir,
            options=ExportOptions(
                export=ExportCallOptions(
                    validate_exports=bool(getattr(args, "validate_exports", False)),
                    schemas=schemas,
                    datasets=datasets,
                    require_normalized_macros=bool(
                        getattr(args, "require_normalized_macros", False)
                    ),
                ),
                validator=validator,
            ),
        )
    except ExportError as exc:
        log_problem(LOG, exc.problem_detail)
        return 1

    LOG.info("Export complete.")
    return 0


def _cmd_history_timeseries(args: argparse.Namespace) -> int:
    """
    Compute analytics.history_timeseries across multiple commits.

    Returns
    -------
    int
        Exit code indicating success (0) or failure.
    """
    _setup_logging(args.verbose)
    runner = ToolRunner(cache_dir=args.repo_root / "build" / ".tool_cache")
    builder = ConfigBuilder.from_snapshot(
        repo=args.repo,
        commit=args.commits[0] if args.commits else "",
        repo_root=args.repo_root,
    )
    cfg = builder.history_timeseries(
        commits=tuple(args.commits),
        entity_kind=args.entity_kind,
        max_entities=args.max_entities,
        selection_strategy=args.selection_strategy,
    )

    storage_cfg = StorageConfig.for_ingest(args.output_db)
    gateway = open_gateway(storage_cfg)
    snapshot_resolver = build_snapshot_gateway_resolver(
        db_dir=args.db_dir,
        repo=args.repo,
        primary_gateway=gateway,
    )
    try:
        compute_history_timeseries_gateways(
            gateway,
            cfg,
            snapshot_resolver,
            runner=runner,
        )
    except FileNotFoundError:
        LOG.exception("Missing snapshot database for history_timeseries")
        return 1
    except DuckDBError:  # pragma: no cover - surfaced to caller
        LOG.exception("Failed to compute history_timeseries")
        return 1
    LOG.info(
        "history_timeseries written to %s for %d commits",
        args.output_db,
        len(args.commits),
    )
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Iterable[str] | None = None) -> int:
    """
    CLI entrypoint for the CodeIntel pipeline and export commands.

    Parameters
    ----------
    argv:
        Optional argument list (defaults to sys.argv).

    Returns
    -------
    int
        Exit code (0 on success).
    """
    parser = _make_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    _setup_logging(args.verbose)

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    try:
        func: CommandHandler = args.func
        return int(func(args))
    except Exception as exc:  # noqa: BLE001 pragma: no cover - error path
        pd = problem(
            code="cli.failure",
            title="CLI command failed",
            detail=str(exc),
            extras={"command": args.command},
        )
        log_problem(LOG, pd)
        return 1


if __name__ == "__main__":
    sys.exit(main())
