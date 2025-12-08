"""Document export commands for the CodeIntel CLI.

This module provides Typer commands for exporting datasets from DuckDB
to Parquet and JSONL formats.

Commands
--------
- **export**: Export Parquet + JSONL datasets from DuckDB
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, cast

import typer

from codeintel.build import get_target_graph
from codeintel.build.executor import BuildExecutor
from codeintel.build.plan import PlanGenerator
from codeintel.build.resolver import BuildResolver
from codeintel.build.state import StateValidator
from codeintel.cli.commands._common import (
    BackendFlags,
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
from codeintel.cli.commands._option_shim import OptionSpec, wrap_command
from codeintel.cli.errors import CLI_EXIT_VALIDATION, DocsValidationError
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

DEFAULT_VALIDATE = False
DEFAULT_REQUIRE_NORMALIZED_MACROS = False


# -----------------------------------------------------------------------------
# Option Type Aliases
# -----------------------------------------------------------------------------

ValidationOpt = Annotated[
    bool,
    typer.Option(
        DEFAULT_VALIDATE,
        "--validation",
        "--validate",
        help="Require validation for exports (exit code 1 on failures).",
        is_flag=True,
        show_default=True,
    ),
]

MacroRequirementOpt = Annotated[
    bool,
    typer.Option(
        DEFAULT_REQUIRE_NORMALIZED_MACROS,
        "--macro-requirement",
        "--require-normalized-macros",
        help="Require normalized macros during export.",
        is_flag=True,
        show_default=True,
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

NxGpuModeOpt = Annotated[
    NxGpuMode,
    typer.Option(
        ...,
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
        ...,
        "--dry-run",
        flag_value=DryRunMode.DRY_RUN,
        help="Plan without executing exports.",
        case_sensitive=False,
    ),
]

PrereqSkipFlagOpt = Annotated[
    bool,
    typer.Option(
        "--skip-prereqs",
        help="Skip prerequisite computation (assume analytics already complete).",
        is_flag=True,
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
    prereq_mode: PrereqMode | None,
) -> DocsExecutionOptions:
    prereq = prereq_mode or PrereqMode.RUN
    return DocsExecutionOptions(
        output_format=output_format,
        run_mode=run_mode,
        prereq_mode=prereq,
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
        BackendFlags(
            use_gpu=backend.nx_gpu_mode in {NxGpuMode.ENABLED, NxGpuMode.STRICT},
            backend=backend.nx_backend,
            strict=backend.nx_gpu_mode is NxGpuMode.STRICT,
        )
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
        check_collisions=True,
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
        When required paths are missing.
    DocsValidationError
        When dataset validation or export validation fails.
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
    except DocsValidationError:
        raise
    except ValueError as exc:
        raise DocsValidationError(str(exc)) from exc
    except ExportError as exc:
        log_problem(LOG, exc.detail)
        message = str(exc.detail.detail or exc.detail.title or "Export validation failed")
        raise DocsValidationError(message) from exc

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


def docs_export_handler(
    project: ProjectOptions,
    backend: BackendOptions,
    export_options: DocsExportOptions,
    verbose: int,
) -> None:
    """Export Parquet + JSONL datasets from DuckDB into Document Output/.

    By default, uses the build system for dependency-aware export, which
    ensures all prerequisites (analytics, profiles) are computed first.

    Use --skip-prereqs to skip prerequisite computation if analytics are
    already complete.

    Exit codes
    ----------
    0 on success, 1 on validation failure, 2 on usage/argument errors.

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

    Raises
    ------
    typer.Exit
        When validation fails or a required path cannot be resolved.
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

    try:
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
    except DocsValidationError as exc:
        typer.secho(f"Validation failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=CLI_EXIT_VALIDATION) from exc


def _bundle_docs_export(cli_kwargs: Mapping[str, object]) -> Mapping[str, object]:
    project = _project_options(
        RepoSelection(
            project_root=cast("Path | None", cli_kwargs.get("project_root")),
            repo=cast("str | None", cli_kwargs.get("repo")),
            commit=cast("str | None", cli_kwargs.get("commit")),
            repo_root=cast("Path | None", cli_kwargs.get("repo_root")),
        ),
        StorageSelection(
            db_path=cast("Path | None", cli_kwargs.get("db_path")),
            build_dir=cast("Path | None", cli_kwargs.get("build_dir")),
            document_output_dir=cast("Path | None", cli_kwargs.get("document_output_dir")),
        ),
    )
    backend = _backend_options(
        nx_backend=cast("str", cli_kwargs.get("nx_backend", "auto")),
        nx_gpu_mode=cast("NxGpuMode", cli_kwargs.get("nx_gpu_mode", NxGpuMode.DISABLED)),
    )
    validation = _docs_validation_options(
        validation=(
            cast("ExportValidationMode", cli_kwargs["validation"])
            if isinstance(cli_kwargs.get("validation"), ExportValidationMode)
            else (
                ExportValidationMode.REQUIRED
                if bool(cli_kwargs.get("validation"))
                else ExportValidationMode.SKIP
            )
        ),
        macro_requirement=(
            cast("MacroRequirement", cli_kwargs["macro_requirement"])
            if isinstance(cli_kwargs.get("macro_requirement"), MacroRequirement)
            else (
                MacroRequirement.REQUIRE_NORMALIZED
                if bool(cli_kwargs.get("macro_requirement"))
                else MacroRequirement.ALLOW_PARTIAL
            )
        ),
    )
    selection = _docs_selection_options(
        schemas=cast("list[str] | None", cli_kwargs.get("schemas")),
        datasets=cast("list[str] | None", cli_kwargs.get("datasets")),
    )
    execution = _docs_execution_options(
        output_format=cast("OutputFormat", cli_kwargs.get("output_format", OutputFormat.TEXT)),
        run_mode=cast("DryRunMode", cli_kwargs.get("run_mode", DryRunMode.EXECUTE)),
        prereq_mode=(
            PrereqMode.SKIP if bool(cli_kwargs.get("prereq_mode", False)) else PrereqMode.RUN
        ),
    )
    export_options = _docs_export_options(validation, selection, execution)
    return {
        "project": project,
        "backend": backend,
        "export_options": export_options,
        "verbose": int(cast("int | str | None", cli_kwargs.get("verbose", 0)) or 0),
    }


_DOCS_EXPORT_SPECS = [
    OptionSpec("project_root", Path | None, ProjectRootOpt),
    OptionSpec("repo", RepoOpt, None),
    OptionSpec("commit", CommitOpt, None),
    OptionSpec("repo_root", RepoRootOpt, None),
    OptionSpec("db_path", DbPathOpt, None),
    OptionSpec("build_dir", BuildDirOpt, None),
    OptionSpec("document_output_dir", DocumentOutputDirOpt, None),
    OptionSpec("nx_backend", NxBackendOpt, "auto"),
    OptionSpec("nx_gpu_mode", NxGpuModeOpt, NxGpuMode.DISABLED),
    OptionSpec(
        "validation",
        bool,
        typer.Option(
            DEFAULT_VALIDATE,
            "--validation",
            "--validate",
            help="Require validation for exports (exit code 1 on failures).",
            is_flag=True,
            show_default=True,
        ),
    ),
    OptionSpec(
        "macro_requirement",
        bool,
        typer.Option(
            DEFAULT_REQUIRE_NORMALIZED_MACROS,
            "--macro-requirement",
            "--require-normalized-macros",
            help="Require normalized macros during export.",
            is_flag=True,
            show_default=True,
        ),
    ),
    OptionSpec("schemas", SchemasOpt, None),
    OptionSpec("datasets", DatasetsOpt, None),
    OptionSpec("output_format", OutputFormat, OutputFormatOpt),
    OptionSpec("run_mode", DryRunModeOpt, DryRunMode.EXECUTE),
    OptionSpec(name="prereq_mode", annotation=PrereqSkipFlagOpt, default=False),
    OptionSpec("verbose", int, VerboseOpt),
]


docs_export = docs_app.command("export")(
    wrap_command(
        docs_export_handler,
        _DOCS_EXPORT_SPECS,
        bundle=_bundle_docs_export,
        name="docs_export",
    )
)


__all__ = [
    "docs_app",
    "run_docs_export",
    "run_docs_export_via_build_system",
]
