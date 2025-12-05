"""Document export commands for the CodeIntel CLI.

This module provides Typer commands for exporting datasets from DuckDB
to Parquet and JSONL formats.

Commands
--------
- **export**: Export Parquet + JSONL datasets from DuckDB
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Protocol

import typer

from codeintel.cli.commands._common import (
    BuildDirOpt,
    CommitOpt,
    DbPathOpt,
    DocumentOutputDirOpt,
    NxBackendOpt,
    NxGpuOpt,
    NxGpuStrictOpt,
    ProjectRootOpt,
    RepoOpt,
    RepoRootOpt,
    VerboseOpt,
    build_graph_backend_config,
    build_graph_feature_flags_from_env,
    open_gateway_from_config,
    setup_logging,
)
from codeintel.config.models import CliConfigOptions, CliPathsInput, CodeIntelConfig, RepoConfig
from codeintel.export.export_jsonl import ExportCallOptions
from codeintel.export.runner import (
    ExportOptions,
    ExportRunner,
    run_validated_exports,
)
from codeintel.graphs.engine.backend import maybe_enable_nx_gpu
from codeintel.serving.backend.datasets import validate_dataset_registry
from codeintel.serving.services.errors import ExportError, log_problem
from codeintel.storage.gateway import StorageGateway

LOG = logging.getLogger(__name__)


class GatewayFactory(Protocol):
    """Factory for creating gateways with optional read-only mode."""

    def __call__(self, cfg: CodeIntelConfig, *, read_only: bool) -> StorageGateway:
        """Create a gateway."""
        ...


docs_app = typer.Typer(
    name="docs",
    help="Document Output export commands.",
    no_args_is_help=True,
)


# -----------------------------------------------------------------------------
# Option Type Aliases
# -----------------------------------------------------------------------------

ValidateOpt = Annotated[
    bool,
    typer.Option(
        "--validate",
        is_flag=True,
        help="Validate exported datasets against JSON Schema definitions.",
    ),
]

SchemasOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--schema",
        help="Schema name to validate (can be repeated). Defaults to standard export set.",
    ),
]

DatasetsOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--dataset",
        help="Dataset name to export (can be repeated). Defaults to all mapped datasets.",
    ),
]

RequireNormalizedMacrosOpt = Annotated[
    bool,
    typer.Option(
        "--require-normalized-macros",
        is_flag=True,
        help="Fail if any requested dataset lacks a normalized macro.",
    ),
]

SkipPrereqsOpt = Annotated[
    bool,
    typer.Option(
        "--skip-prereqs",
        is_flag=True,
        help="Skip prerequisite computation (assume analytics already complete).",
    ),
]


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------


def _resolve_export_config(
    project_root: Path | None,
    repo: str | None,
    commit: str | None,
    db_path: Path | None,
    build_dir: Path | None,
    repo_root: Path | None,
    document_output_dir: Path | None,
    nx_gpu: bool,
    nx_backend: str,
    nx_gpu_strict: bool,
) -> CodeIntelConfig:
    """Resolve export configuration from options.

    Parameters
    ----------
    project_root
        Project root path.
    repo
        Repository slug.
    commit
        Commit SHA.
    db_path
        Database path.
    build_dir
        Build directory.
    repo_root
        Repository root.
    document_output_dir
        Document output directory.
    nx_gpu
        Whether to use GPU.
    nx_backend
        NetworkX backend.
    nx_gpu_strict
        Whether strict GPU mode.

    Returns
    -------
    CodeIntelConfig
        Resolved configuration.
    """
    from codeintel.cli.project import ProjectNotFoundError, find_project_root, load_project_config

    try:
        project_root_path = find_project_root(project_root)
        project_config = load_project_config(project_root_path)

        from codeintel.cli.project import detect_commit

        resolved = {
            "repo": repo or project_config.repo,
            "commit": commit or detect_commit(project_root_path),
            "db_path": db_path or (project_root_path / project_config.storage.db_path),
            "repo_root": repo_root or project_root_path,
            "build_dir": build_dir or (project_root_path / ".codeintel"),
        }
    except ProjectNotFoundError:
        if repo is None or commit is None:
            typer.secho(
                "Error: No codeintel.yaml found. Provide --repo and --commit explicitly.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1) from None
        resolved = {
            "repo": repo,
            "commit": commit,
            "db_path": db_path or Path("build/db/codeintel.duckdb"),
            "repo_root": repo_root or Path.cwd(),
            "build_dir": build_dir or Path("build"),
        }

    graph_backend = build_graph_backend_config(nx_gpu, nx_backend, nx_gpu_strict)
    graph_features = build_graph_feature_flags_from_env()

    paths_cfg = CliPathsInput(
        repo_root=resolved["repo_root"],
        build_dir=resolved["build_dir"],
        db_path=resolved["db_path"],
        document_output_dir=document_output_dir,
    )
    repo_cfg = RepoConfig(repo=resolved["repo"], commit=resolved["commit"])
    return CodeIntelConfig.from_cli_args(
        repo_cfg=repo_cfg,
        paths_cfg=paths_cfg,
        options=CliConfigOptions(graph_backend=graph_backend, graph_features=graph_features),
    )


def run_docs_export_via_build_system(
    cfg: CodeIntelConfig,
    *,
    validate_exports: bool = True,
    schemas: list[str] | None = None,
    datasets: list[str] | None = None,
    require_normalized_macros: bool = False,
) -> None:
    """Execute docs export using the build system for dependency-aware execution.

    This function uses the build system to ensure all prerequisites are met
    before running the export. It will run any missing analytics/graph targets
    that the export depends on.

    Parameters
    ----------
    cfg
        Resolved configuration.
    validate_exports
        Whether to validate exports against JSON Schema.
    schemas
        Specific schemas to validate (None for all).
    datasets
        Specific datasets to export (None for all).
    require_normalized_macros
        Whether to require normalized macros.

    Raises
    ------
    typer.Exit
        If the build fails.
    """
    from codeintel.build import get_target_graph
    from codeintel.build.executor import BuildExecutor
    from codeintel.build.plan import PlanGenerator
    from codeintel.build.resolver import BuildResolver
    from codeintel.build.state import StateValidator
    from codeintel.config.primitives import BuildPaths, SnapshotRef
    from codeintel.export.export_jsonl import ExportCallOptions

    maybe_enable_nx_gpu(cfg.graph_backend)
    gateway = open_gateway_from_config(cfg, read_only=False)

    out_dir = cfg.paths.document_output_dir
    if out_dir is None:
        typer.secho(
            "Error: document_output_dir was not resolved",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    snapshot = SnapshotRef(
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        repo_root=cfg.paths.repo_root,
    )
    paths = BuildPaths.from_layout(
        repo_root=cfg.paths.repo_root,
        build_dir=cfg.paths.build_dir,
        db_path=cfg.paths.db_path,
        document_output_dir=cfg.paths.document_output_dir,
    )

    graph = get_target_graph()
    validator = StateValidator(graph=graph, gateway=gateway, snapshot=snapshot)
    state = validator.validate()

    # Resolve what needs to run for export targets
    resolver = BuildResolver(graph=graph, state=state)
    resolution = resolver.resolve(
        goals=["export_jsonl", "export_parquet"],
        force_recompute=None,
    )

    if not resolution.to_compute:
        LOG.info("All export targets are up to date.")
        typer.secho("Exports are up to date.", fg=typer.colors.GREEN)
        return

    # Generate and execute the build plan
    generator = PlanGenerator(graph=graph)
    plan = generator.generate(resolution)

    LOG.info(
        "Build system: %d targets to compute, %d to skip",
        len(resolution.to_compute),
        len(resolution.to_skip),
    )

    executor = BuildExecutor(
        gateway=gateway,
        snapshot=snapshot,
        paths=paths,
        tools=cfg.tools,
        graph=graph,
    )

    # Set export options from CLI parameters
    executor.export_options = ExportCallOptions(
        validate_exports=validate_exports,
        schemas=schemas,
        datasets=datasets,
        require_normalized_macros=require_normalized_macros,
    )
    result = executor.execute(plan)

    if result.status == "failed":
        typer.secho(
            f"Export failed: {result.error_summary}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    LOG.info("Export complete via build system.")
    typer.secho("Export complete.", fg=typer.colors.GREEN)


def run_docs_export(
    cfg: CodeIntelConfig,
    validate_exports: bool,
    schemas: list[str] | None,
    datasets: list[str] | None,
    require_normalized_macros: bool,
    validator: Callable[[StorageGateway], None],
    export_runner: ExportRunner,
) -> None:
    """Execute the docs export with provided configuration and callbacks (legacy).

    Parameters
    ----------
    cfg
        Resolved configuration.
    validate_exports
        Whether to validate exports.
    schemas
        Schemas to validate.
    datasets
        Datasets to export.
    require_normalized_macros
        Whether to require normalized macros.
    validator
        Dataset validation callback.
    export_runner
        Export runner callback.
    """
    maybe_enable_nx_gpu(cfg.graph_backend)
    gateway = open_gateway_from_config(cfg, read_only=True)
    out_dir = cfg.paths.document_output_dir
    if out_dir is None:
        typer.secho(
            "Error: document_output_dir was not resolved",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Exporting Parquet + JSONL datasets into %s", out_dir)
    schemas_list = list(schemas) if schemas else None
    datasets_list = list(datasets) if datasets else None

    try:
        export_runner(
            gateway=gateway,
            output_dir=out_dir,
            options=ExportOptions(
                export=ExportCallOptions(
                    validate_exports=validate_exports,
                    schemas=schemas_list,
                    datasets=datasets_list,
                    require_normalized_macros=require_normalized_macros,
                ),
                validator=validator,
            ),
        )
    except ExportError as exc:
        log_problem(LOG, exc.detail)
        raise typer.Exit(code=1) from exc

    LOG.info("Export complete.")
    typer.secho("Export complete.", fg=typer.colors.GREEN)


@docs_app.command("export")
def docs_export(
    project_root: ProjectRootOpt = None,
    repo: RepoOpt = None,
    commit: CommitOpt = None,
    db_path: DbPathOpt = None,
    build_dir: BuildDirOpt = None,
    repo_root: RepoRootOpt = None,
    document_output_dir: DocumentOutputDirOpt = None,
    validate: ValidateOpt = False,
    schemas: SchemasOpt = None,
    datasets: DatasetsOpt = None,
    require_normalized_macros: RequireNormalizedMacrosOpt = False,
    skip_prereqs: SkipPrereqsOpt = False,
    nx_gpu: NxGpuOpt = False,
    nx_backend: NxBackendOpt = "auto",
    nx_gpu_strict: NxGpuStrictOpt = False,
    verbose: VerboseOpt = 0,
) -> None:
    """Export Parquet + JSONL datasets from DuckDB into Document Output/.

    By default, uses the build system for dependency-aware export, which
    ensures all prerequisites (analytics, profiles) are computed first.

    Use --skip-prereqs to skip prerequisite computation if analytics are
    already complete.

    Examples
    --------
    .. code-block:: bash

        # Export all datasets (runs prerequisites if needed)
        codeintel docs export

        # Export with explicit repo/commit
        codeintel docs export --repo my-org/repo --commit abc123

        # Export specific datasets with validation
        codeintel docs export --dataset functions --dataset modules --validate

        # Skip prerequisites (assume analytics complete)
        codeintel docs export --skip-prereqs
    """
    setup_logging(verbose)

    cfg = _resolve_export_config(
        project_root,
        repo,
        commit,
        db_path,
        build_dir,
        repo_root,
        document_output_dir,
        nx_gpu,
        nx_backend,
        nx_gpu_strict,
    )

    if skip_prereqs:
        # Direct export without build system (legacy behavior)
        run_docs_export(
            cfg=cfg,
            validate_exports=validate,
            schemas=list(schemas) if schemas else None,
            datasets=list(datasets) if datasets else None,
            require_normalized_macros=require_normalized_macros,
            validator=validate_dataset_registry,
            export_runner=run_validated_exports,
        )
    else:
        # Use build system with all options
        run_docs_export_via_build_system(
            cfg,
            validate_exports=validate,
            schemas=list(schemas) if schemas else None,
            datasets=list(datasets) if datasets else None,
            require_normalized_macros=require_normalized_macros,
        )


__all__ = [
    "docs_app",
    "run_docs_export",
    "run_docs_export_via_build_system",
]
