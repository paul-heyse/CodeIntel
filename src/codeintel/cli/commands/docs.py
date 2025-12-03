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
from codeintel.graphs.nx_backend import maybe_enable_nx_gpu
from codeintel.pipeline.export.export_jsonl import ExportCallOptions
from codeintel.pipeline.export.runner import (
    ExportOptions,
    ExportRunner,
    run_validated_exports,
)
from codeintel.serving.http.datasets import validate_dataset_registry
from codeintel.serving.services.errors import ExportError, log_problem
from codeintel.storage.gateway import StorageGateway

LOG = logging.getLogger(__name__)


class GatewayFactory(Protocol):
    """Factory for creating gateways with optional read-only mode (test compatibility)."""

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


def _run_docs_export(
    cfg: CodeIntelConfig,
    validate_exports: bool,
    schemas: list[str] | None,
    datasets: list[str] | None,
    require_normalized_macros: bool,
    validator: Callable[[StorageGateway], None],
    export_runner: ExportRunner,
) -> None:
    """Execute the docs export with provided configuration and callbacks.

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
    nx_gpu: NxGpuOpt = False,
    nx_backend: NxBackendOpt = "auto",
    nx_gpu_strict: NxGpuStrictOpt = False,
    verbose: VerboseOpt = 0,
) -> None:
    """Export Parquet + JSONL datasets from DuckDB into Document Output/.

    Assumes the pipeline has already populated the DuckDB database.

    Examples
    --------
    .. code-block:: bash

        # Export all datasets
        codeintel docs export --repo my-org/repo --commit abc123

        # Export specific datasets with validation
        codeintel docs export --dataset functions --dataset modules --validate

        # Using project file
        codeintel docs export
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

    _run_docs_export(
        cfg=cfg,
        validate_exports=validate,
        schemas=list(schemas) if schemas else None,
        datasets=list(datasets) if datasets else None,
        require_normalized_macros=require_normalized_macros,
        validator=validate_dataset_registry,
        export_runner=run_validated_exports,
    )


def cmd_docs_export(
    args: object,
    validator: Callable[[StorageGateway], None] = validate_dataset_registry,
    export_runner: ExportRunner = run_validated_exports,
    gateway_factory: GatewayFactory | None = None,
) -> int:
    """Execute docs export from argparse namespace (test compatibility).

    Parameters
    ----------
    args
        Argparse namespace with export options.
    validator
        Dataset validation callback.
    export_runner
        Export runner callback.
    gateway_factory
        Optional gateway factory override for testing.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    from codeintel.cli.commands._common import (
        build_graph_backend_config,
        build_graph_feature_flags_from_env,
    )
    from codeintel.storage.config import StorageConfig
    from codeintel.storage.gateway import open_gateway

    db_path = Path(getattr(args, "db_path", ""))
    paths_cfg = CliPathsInput(
        repo_root=Path(getattr(args, "repo_root", "")),
        build_dir=Path(getattr(args, "build_dir", "")),
        db_path=db_path,
        document_output_dir=Path(getattr(args, "document_output_dir", "")),
    )
    repo_cfg = RepoConfig(
        repo=str(getattr(args, "repo", "")),
        commit=str(getattr(args, "commit", "")),
    )
    cfg = CodeIntelConfig.from_cli_args(
        repo_cfg=repo_cfg,
        paths_cfg=paths_cfg,
        options=CliConfigOptions(
            graph_backend=build_graph_backend_config(
                bool(getattr(args, "nx_gpu", False)),
                str(getattr(args, "nx_backend", "auto")),
                bool(getattr(args, "nx_gpu_strict", False)),
            ),
            graph_features=build_graph_feature_flags_from_env(),
        ),
    )

    maybe_enable_nx_gpu(cfg.graph_backend)

    if gateway_factory is not None:
        gateway = gateway_factory(cfg, read_only=True)
    else:
        gateway = open_gateway(StorageConfig.for_readonly(db_path))

    out_dir = cfg.paths.document_output_dir
    if out_dir is None:
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("Exporting Parquet + JSONL datasets into %s", out_dir)

    schemas_raw = getattr(args, "schemas", None)
    datasets_raw = getattr(args, "datasets", None)
    try:
        export_runner(
            gateway=gateway,
            output_dir=out_dir,
            options=ExportOptions(
                export=ExportCallOptions(
                    validate_exports=bool(getattr(args, "validate_exports", False)),
                    schemas=list(schemas_raw) if schemas_raw else None,
                    datasets=list(datasets_raw) if datasets_raw else None,
                    require_normalized_macros=bool(
                        getattr(args, "require_normalized_macros", False)
                    ),
                ),
                validator=validator,
            ),
        )
    except ExportError as exc:
        log_problem(LOG, exc.detail)
        return 1

    LOG.info("Export complete.")
    return 0


__all__ = [
    "GatewayFactory",
    "cmd_docs_export",
    "docs_app",
    "run_docs_export",
]


# Alias for test compatibility
run_docs_export = _run_docs_export
