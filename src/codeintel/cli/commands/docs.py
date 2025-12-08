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
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from codeintel.build import get_target_graph
from codeintel.build.executor import BuildExecutor
from codeintel.build.plan import PlanGenerator
from codeintel.build.resolver import BuildResolver
from codeintel.build.state import StateValidator
from codeintel.cli.commands._common import (
    BuildDirOpt,
    CommitOpt,
    DbPathOpt,
    DocumentOutputDirOpt,
    NxBackendOpt,
    OutputFormat,
    ProjectRootOpt,
    RepoOpt,
    RepoRootOpt,
    VerboseOpt,
    build_graph_backend_config,
    build_graph_feature_flags_from_env,
    open_gateway_from_config,
    setup_logging,
)
from codeintel.cli.project import (
    ProjectNotFoundError,
    detect_commit,
    find_project_root,
    load_project_config,
)
from codeintel.config.models import CliConfigOptions, CliPathsInput, CodeIntelConfig, RepoConfig
from codeintel.config.primitives import BuildLayoutOptions, BuildPaths, SnapshotRef
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


class ExportValidationMode(Enum):
    """Validation strategy for docs exports."""

    REQUIRED = "required"
    SKIP = "skip"


class MacroRequirement(Enum):
    """Requirement policy for normalized macros."""

    REQUIRE_NORMALIZED = "require_normalized"
    ALLOW_PARTIAL = "allow_partial"


class DryRunMode(Enum):
    """Execution mode for docs exports."""

    EXECUTE = "execute"
    DRY_RUN = "dry_run"


class NxGpuMode(Enum):
    """GPU backend mode for NetworkX."""

    DISABLED = "disabled"
    ENABLED = "enabled"
    STRICT = "strict"


class PrereqMode(Enum):
    """Prerequisite execution strategy."""

    RUN = "run"
    SKIP = "skip"


@dataclass(frozen=True)
class DocsExportOptions:
    """Bundled options for docs export workflows."""

    validation: ExportValidationMode = ExportValidationMode.REQUIRED
    macro_requirement: MacroRequirement = MacroRequirement.REQUIRE_NORMALIZED
    datasets: list[str] | None = None
    schemas: list[str] | None = None
    output_format: OutputFormat = OutputFormat.TEXT
    run_mode: DryRunMode = DryRunMode.EXECUTE
    prereq_mode: PrereqMode = PrereqMode.RUN


@dataclass(frozen=True)
class ProjectOptions:
    """Project/runtime resolution inputs."""

    project_root: Path | None
    repo: str | None
    commit: str | None
    db_path: Path | None
    build_dir: Path | None
    repo_root: Path | None
    document_output_dir: Path | None


@dataclass(frozen=True)
class BackendOptions:
    """Graph backend selection."""

    nx_backend: str
    nx_gpu_mode: NxGpuMode


@dataclass(frozen=True)
class RepoSelection:
    """Repository selection inputs."""

    project_root: Path | None
    repo: str | None
    commit: str | None
    repo_root: Path | None


@dataclass(frozen=True)
class StorageSelection:
    """Storage and build path inputs."""

    db_path: Path | None
    build_dir: Path | None
    document_output_dir: Path | None


@dataclass(frozen=True)
class DocsValidationOptions:
    """Validation toggles for docs exports."""

    validation: ExportValidationMode
    macro_requirement: MacroRequirement


@dataclass(frozen=True)
class DocsSelectionOptions:
    """Dataset/schema selection for docs exports."""

    schemas: list[str] | None
    datasets: list[str] | None


@dataclass(frozen=True)
class DocsExecutionOptions:
    """Execution and output options for docs exports."""

    output_format: OutputFormat
    run_mode: DryRunMode
    prereq_mode: PrereqMode


docs_app = typer.Typer(
    name="docs",
    help="Document Output export commands.",
    no_args_is_help=True,
)


# -----------------------------------------------------------------------------
# Option Type Aliases
# -----------------------------------------------------------------------------

ValidationOpt = Annotated[
    ExportValidationMode,
    typer.Option(
        ExportValidationMode.SKIP,
        "--validation",
        "--validate",
        flag_value=ExportValidationMode.REQUIRED,
        help="Validation strategy for exports.",
        case_sensitive=False,
    ),
]

MacroRequirementOpt = Annotated[
    MacroRequirement,
    typer.Option(
        MacroRequirement.ALLOW_PARTIAL,
        "--macro-requirement",
        "--require-normalized-macros",
        flag_value=MacroRequirement.REQUIRE_NORMALIZED,
        help="Requirement policy for normalized macros.",
        case_sensitive=False,
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

PrereqModeOpt = Annotated[
    PrereqMode,
    typer.Option(
        PrereqMode.RUN,
        "--skip-prereqs",
        flag_value=PrereqMode.SKIP,
        help="Skip prerequisite computation (assume analytics already complete).",
        case_sensitive=False,
    ),
]

NxGpuModeOpt = Annotated[
    NxGpuMode,
    typer.Option(
        NxGpuMode.DISABLED,
        "--nx-gpu-mode",
        help="GPU backend mode: disabled, enabled, or strict.",
        case_sensitive=False,
    ),
]

OutputFormatOpt = typer.Option(
    OutputFormat.TEXT,
    "--output-format",
    help="Output format for command results.",
    case_sensitive=False,
    show_choices=True,
)

DryRunModeOpt = Annotated[
    DryRunMode,
    typer.Option(
        DryRunMode.EXECUTE,
        "--dry-run",
        flag_value=DryRunMode.DRY_RUN,
        help="Plan without executing exports.",
        case_sensitive=False,
    ),
]


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------


def _project_options(
    repo_selection: RepoSelection,
    storage_selection: StorageSelection,
) -> ProjectOptions:
    return ProjectOptions(
        project_root=repo_selection.project_root,
        repo=repo_selection.repo,
        commit=repo_selection.commit,
        db_path=storage_selection.db_path,
        build_dir=storage_selection.build_dir,
        repo_root=repo_selection.repo_root,
        document_output_dir=storage_selection.document_output_dir,
    )


def _backend_options(
    nx_backend: str,
    nx_gpu_mode: NxGpuMode,
) -> BackendOptions:
    return BackendOptions(nx_backend=nx_backend, nx_gpu_mode=nx_gpu_mode)


def _docs_validation_options(
    validation: ExportValidationMode,
    macro_requirement: MacroRequirement,
) -> DocsValidationOptions:
    return DocsValidationOptions(
        validation=validation,
        macro_requirement=macro_requirement,
    )


def _docs_selection_options(
    schemas: list[str] | None,
    datasets: list[str] | None,
) -> DocsSelectionOptions:
    return DocsSelectionOptions(
        schemas=schemas,
        datasets=datasets,
    )


def _docs_execution_options(
    output_format: OutputFormat,
    run_mode: DryRunMode,
    prereq_mode: PrereqMode,
) -> DocsExecutionOptions:
    return DocsExecutionOptions(
        output_format=output_format,
        run_mode=run_mode,
        prereq_mode=prereq_mode,
    )


def _docs_export_options(
    validation: DocsValidationOptions,
    selection: DocsSelectionOptions,
    execution: DocsExecutionOptions,
) -> DocsExportOptions:
    return DocsExportOptions(
        validation=validation.validation,
        macro_requirement=validation.macro_requirement,
        datasets=selection.datasets,
        schemas=selection.schemas,
        output_format=execution.output_format,
        run_mode=execution.run_mode,
        prereq_mode=execution.prereq_mode,
    )


# -----------------------------------------------------------------------------
# Dependency Builders
# -----------------------------------------------------------------------------


def _repo_selection_dep(
    project_root: Path | None = ProjectRootOpt,
    repo: RepoOpt = None,
    commit: CommitOpt = None,
    repo_root: RepoRootOpt = None,
) -> RepoSelection:
    return RepoSelection(
        project_root=project_root,
        repo=repo,
        commit=commit,
        repo_root=repo_root,
    )


def _storage_selection_dep(
    db_path: DbPathOpt = None,
    build_dir: BuildDirOpt = None,
    document_output_dir: DocumentOutputDirOpt = None,
) -> StorageSelection:
    return StorageSelection(
        db_path=db_path,
        build_dir=build_dir,
        document_output_dir=document_output_dir,
    )


def _project_options_dep(
    repo_selection: Annotated[RepoSelection, typer.Depends(_repo_selection_dep)],
    storage_selection: Annotated[StorageSelection, typer.Depends(_storage_selection_dep)],
) -> ProjectOptions:
    return _project_options(repo_selection, storage_selection)


def _backend_options_dep(
    nx_backend: NxBackendOpt = "auto",
    nx_gpu_mode: NxGpuModeOpt = NxGpuMode.DISABLED,
) -> BackendOptions:
    return _backend_options(nx_backend, nx_gpu_mode)


def _docs_validation_options_dep(
    validation: ValidationOpt = ExportValidationMode.SKIP,
    macro_requirement: MacroRequirementOpt = MacroRequirement.ALLOW_PARTIAL,
) -> DocsValidationOptions:
    return _docs_validation_options(validation, macro_requirement)


def _docs_selection_options_dep(
    schemas: SchemasOpt = None,
    datasets: DatasetsOpt = None,
) -> DocsSelectionOptions:
    return _docs_selection_options(schemas, datasets)


def _docs_execution_options_dep(
    output_format: OutputFormat = OutputFormatOpt,
    run_mode: DryRunModeOpt = DryRunMode.EXECUTE,
    prereq_mode: PrereqModeOpt = PrereqMode.RUN,
) -> DocsExecutionOptions:
    return _docs_execution_options(output_format, run_mode, prereq_mode)


def _docs_export_options_dep(
    validation: Annotated[DocsValidationOptions, typer.Depends(_docs_validation_options_dep)],
    selection: Annotated[DocsSelectionOptions, typer.Depends(_docs_selection_options_dep)],
    execution: Annotated[DocsExecutionOptions, typer.Depends(_docs_execution_options_dep)],
) -> DocsExportOptions:
    return _docs_export_options(validation, selection, execution)


def _resolve_export_config(
    project: ProjectOptions,
    backend: BackendOptions,
) -> CodeIntelConfig:
    """Resolve export configuration from options.

    Parameters
    ----------
    project
        Project resolution options.
    backend
        Graph backend options.

    Returns
    -------
    CodeIntelConfig
        Resolved configuration.

    Raises
    ------
    typer.Exit
        When required repository information is missing.
    """
    try:
        project_root_path = find_project_root(project.project_root)
        project_config = load_project_config(project_root_path)

        resolved = {
            "repo": project.repo or project_config.repo,
            "commit": project.commit or detect_commit(project_root_path),
            "db_path": project.db_path or (project_root_path / project_config.storage.db_path),
            "repo_root": project.repo_root or project_root_path,
            "build_dir": project.build_dir or (project_root_path / ".codeintel"),
        }
    except ProjectNotFoundError:
        if project.repo is None or project.commit is None:
            typer.secho(
                "Error: No codeintel.yaml found. Provide --repo and --commit explicitly.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1) from None
        resolved = {
            "repo": project.repo,
            "commit": project.commit,
            "db_path": project.db_path or Path("build/db/codeintel.duckdb"),
            "repo_root": project.repo_root or Path.cwd(),
            "build_dir": project.build_dir or Path("build"),
        }

    graph_backend = build_graph_backend_config(
        backend.nx_gpu_mode in {NxGpuMode.ENABLED, NxGpuMode.STRICT},
        backend.nx_backend,
        backend.nx_gpu_mode is NxGpuMode.STRICT,
    )
    graph_features = build_graph_feature_flags_from_env()

    paths_cfg = CliPathsInput(
        repo_root=resolved["repo_root"],
        build_dir=resolved["build_dir"],
        db_path=resolved["db_path"],
        document_output_dir=project.document_output_dir,
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
    options: DocsExportOptions,
) -> None:
    """Execute docs export using the build system for dependency-aware execution.

    This function uses the build system to ensure all prerequisites are met
    before running the export. It will run any missing analytics/graph targets
    that the export depends on.

    Parameters
    ----------
    cfg
        Resolved configuration.
    options
        Export options bundle.

    Raises
    ------
    typer.Exit
        When the build plan fails or execution errors occur.

    Raises
    ------
    typer.Exit
        If the build fails.
    """
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
        overrides=BuildLayoutOptions(
            build_dir=cfg.paths.build_dir,
            db_path=cfg.paths.db_path,
            document_output_dir=cfg.paths.document_output_dir,
        ),
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
        validate_exports=options.validation is ExportValidationMode.REQUIRED,
        schemas=options.schemas,
        datasets=options.datasets,
        require_normalized_macros=options.macro_requirement is MacroRequirement.REQUIRE_NORMALIZED,
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
    if options.output_format is OutputFormat.JSON:
        typer.echo(
            {
                "status": "ok",
                "validation": options.validation.value,
                "macro_requirement": options.macro_requirement.value,
                "datasets": options.datasets,
                "schemas": options.schemas,
                "mode": "build_system",
            }
        )
    else:
        typer.secho("Export complete.", fg=typer.colors.GREEN)


def run_docs_export(
    cfg: CodeIntelConfig,
    options: DocsExportOptions,
    validator: Callable[[StorageGateway], None],
    export_runner: ExportRunner,
) -> None:
    """Execute the docs export with provided configuration and callbacks (legacy).

    Parameters
    ----------
    cfg
        Resolved configuration.
    options
        Export options bundle.
    validator
        Dataset validation callback.
    export_runner
        Export runner callback.

    Raises
    ------
    typer.Exit
        When export validation fails or execution errors occur.
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
    schemas_list = list(options.schemas) if options.schemas else None
    datasets_list = list(options.datasets) if options.datasets else None

    if options.run_mode is DryRunMode.DRY_RUN:
        payload = {
            "output_dir": str(out_dir),
            "schemas": schemas_list,
            "datasets": datasets_list,
            "validation": options.validation.value,
            "macro_requirement": options.macro_requirement.value,
            "mode": "dry_run",
        }
        if options.output_format is OutputFormat.JSON:
            typer.echo(payload)
        else:
            typer.secho("Dry run: exports planned, no files written.", fg=typer.colors.YELLOW)
        return

    try:
        export_runner(
            gateway=gateway,
            output_dir=out_dir,
            options=ExportOptions(
                export=ExportCallOptions(
                    validate_exports=options.validation is ExportValidationMode.REQUIRED,
                    schemas=schemas_list,
                    datasets=datasets_list,
                    require_normalized_macros=options.macro_requirement
                    is MacroRequirement.REQUIRE_NORMALIZED,
                ),
                validator=validator,
            ),
        )
    except ExportError as exc:
        log_problem(LOG, exc.detail)
        raise typer.Exit(code=1) from exc

    LOG.info("Export complete.")
    if options.output_format is OutputFormat.JSON:
        typer.echo(
            {
                "status": "ok",
                "validation": options.validation.value,
                "macro_requirement": options.macro_requirement.value,
                "datasets": datasets_list,
                "schemas": schemas_list,
                "mode": "direct",
            }
        )
    else:
        typer.secho("Export complete.", fg=typer.colors.GREEN)


@docs_app.command("export")
def docs_export(
    project: Annotated[ProjectOptions, typer.Depends(_project_options_dep)],
    backend: Annotated[BackendOptions, typer.Depends(_backend_options_dep)],
    export_options: Annotated[DocsExportOptions, typer.Depends(_docs_export_options_dep)],
    verbose: int = VerboseOpt,
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

    project_opts = project
    backend_opts = backend
    export_opts = export_options

    cfg = _resolve_export_config(project_opts, backend_opts)

    if export_opts.run_mode is DryRunMode.DRY_RUN:
        payload = {
            "mode": "dry_run",
            "prereq_mode": export_opts.prereq_mode.value,
            "validation": export_opts.validation.value,
            "macro_requirement": export_opts.macro_requirement.value,
            "datasets": export_opts.datasets,
            "schemas": export_opts.schemas,
            "backend": backend_opts.nx_backend,
            "gpu_mode": backend_opts.nx_gpu_mode.value,
        }
        if export_opts.output_format is OutputFormat.JSON:
            typer.echo(payload)
        else:
            typer.secho("Dry run: exports planned, no actions taken.", fg=typer.colors.YELLOW)
        return

    if export_opts.prereq_mode is PrereqMode.SKIP:
        # Direct export without build system (legacy behavior)
        run_docs_export(
            cfg=cfg,
            options=export_opts,
            validator=validate_dataset_registry,
            export_runner=run_validated_exports,
        )
    else:
        # Use build system with all options
        run_docs_export_via_build_system(
            cfg,
            options=export_opts,
        )


__all__ = [
    "docs_app",
    "run_docs_export",
    "run_docs_export_via_build_system",
]
