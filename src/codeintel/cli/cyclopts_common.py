"""Shared Cyclopts primitives and runtime helpers for the CodeIntel CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.commands._common import (
    BackendFlags,
    OutputFormat,
    RuntimeCliOptions,
    build_config_from_options,
)
from codeintel.cli.project import (
    ProjectConfig,
    ProjectNotFoundError,
    ProjectRuntime,
    StorageProjectConfig,
    build_project_runtime,
)
from codeintel.config.models import CliPathsInput
from codeintel.config.primitives import SnapshotRef
from codeintel.config.serving_models import ServingConfig
from codeintel.storage.gateway import StorageConfig, open_gateway


def make_root_app() -> App:
    """Construct the root Cyclopts application with shared defaults.

    Returns
    -------
    App
        Root Cyclopts application configured with default parameters.
    """
    return App(
        name="codeintel",
        help="CodeIntel unified CLI for build, analytics, and serving operations.",
        default_parameter=Parameter(
            show_default=True,
        ),
    )


Verbose = Annotated[
    int,
    Parameter(
        name=["--verbose", "-v"],
        help="Increase verbosity (0=warnings, 1=info, 2=debug).",
        count=True,
    ),
]

ProjectRoot = Annotated[
    Path | None,
    Parameter(
        name=["--root", "-r"],
        help="Explicit project root directory.",
    ),
]

OutputFmt = Annotated[
    OutputFormat,
    Parameter(
        name="--output-format",
        help="Output format.",
        show_choices=True,
    ),
]

JsonFlag = Annotated[
    bool,
    Parameter(
        name="--json",
        help="Alias for --output-format json.",
        negative=(),
    ),
]


class RuntimeCliError(Exception):
    """Raised when CLI runtime resolution fails."""


@dataclass
class RuntimeCLI:
    """Shared runtime selection flags for Cyclopts commands."""

    project_root: ProjectRoot = None
    repo: Annotated[
        str | None,
        Parameter(
            name="--repo",
            help="Repository slug (e.g., 'org/repo'). Uses project config if omitted.",
        ),
    ] = None
    commit: Annotated[
        str | None,
        Parameter(
            name="--commit",
            help="Commit SHA. Uses project config if omitted.",
        ),
    ] = None
    db_path: Annotated[
        Path | None,
        Parameter(
            name="--db-path",
            help="Path to DuckDB database. Uses project config if omitted.",
        ),
    ] = None
    build_dir: Annotated[
        Path | None,
        Parameter(
            name="--build-dir",
            help="Build directory (default: build/).",
        ),
    ] = None
    repo_root: Annotated[
        Path | None,
        Parameter(
            name="--repo-root",
            help="Path to repository root (default: current directory).",
        ),
    ] = None
    document_output_dir: Annotated[
        Path | None,
        Parameter(
            name="--document-output-dir",
            help="Override Document Output/ directory.",
        ),
    ] = None
    verbose: Verbose = 0


RuntimeParam = Annotated[RuntimeCLI, Parameter(name="*")]


@dataclass
class ProjectCLI:
    """Bundle runtime selection under a project alias."""

    runtime: RuntimeParam = field(default_factory=RuntimeCLI)


@dataclass
class OutputFormatCLI:
    """Shared output format toggles for commands supporting JSON output."""

    output_format: OutputFmt = OutputFormat.TEXT
    json: JsonFlag = False


def resolve_output_format(
    *,
    json_flag: bool,
    explicit: OutputFormat | None,
    default: OutputFormat = OutputFormat.TEXT,
) -> OutputFormat:
    """Resolve output format with consistent precedence.

    Returns
    -------
    OutputFormat
        Effective output format after applying overrides.
    """
    if json_flag:
        return OutputFormat.JSON
    if explicit is not None:
        return explicit
    return default


def runtime_cli_to_options(
    cli: RuntimeCLI, *, backend: BackendFlags | None = None
) -> RuntimeCliOptions:
    """Convert a RuntimeCLI dataclass to RuntimeCliOptions.

    Returns
    -------
    RuntimeCliOptions
        Options object suitable for runtime construction.
    """
    return RuntimeCliOptions(
        project_root=cli.project_root,
        repo=cli.repo,
        commit=cli.commit,
        db_path=cli.db_path,
        build_dir=cli.build_dir,
        repo_root=cli.repo_root,
        document_output_dir=cli.document_output_dir,
        backend=backend or BackendFlags(),
    )


def build_runtime_from_cli(
    options: RuntimeCliOptions | RuntimeCLI,
    *,
    allow_fallback: bool = True,
) -> ProjectRuntime:
    """Build a :class:`ProjectRuntime` from CLI options without Typer exits.

    Returns
    -------
    ProjectRuntime
        Constructed runtime context.

    Raises
    ------
    RuntimeCliError
        If a project cannot be resolved from the provided options.
    """
    if isinstance(options, RuntimeCLI):
        options = runtime_cli_to_options(options)

    try:
        return build_project_runtime(options.project_root)
    except ProjectNotFoundError:
        if not allow_fallback:
            message = "No codeintel.yaml found and fallback disabled."
            raise RuntimeCliError(message) from None

    if options.repo is None or options.commit is None:
        msg = (
            "No codeintel.yaml found. Provide --repo and --commit explicitly, "
            "or create a project file."
        )
        raise RuntimeCliError(msg)

    resolved_repo_root = options.repo_root or Path.cwd()
    resolved_db_path = options.db_path or Path("build/db/codeintel.duckdb")
    resolved_build_dir = options.build_dir or Path("build")

    paths_cfg = CliPathsInput(
        repo_root=resolved_repo_root,
        build_dir=resolved_build_dir,
        db_path=resolved_db_path,
        document_output_dir=options.document_output_dir,
    )

    cfg = build_config_from_options(
        repo=options.repo,
        commit=options.commit,
        paths_cfg=paths_cfg,
        backend=options.backend,
    )

    snapshot = SnapshotRef(
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        repo_root=cfg.paths.repo_root,
    )
    paths = cfg.build_paths
    paths.db_path.parent.mkdir(parents=True, exist_ok=True)

    storage_cfg = StorageConfig.for_ingest(db_path=paths.db_path)
    gateway = open_gateway(storage_cfg)

    project = ProjectConfig(
        repo=cfg.repo.repo,
        storage=StorageProjectConfig(db_path=paths.db_path),
    )

    serving = ServingConfig(
        mode="local_db",
        repo_root=cfg.paths.repo_root,
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        db_path=paths.db_path,
        read_only=True,
    )

    return ProjectRuntime(
        root=resolved_repo_root,
        project=project,
        cfg=cfg,
        snapshot=snapshot,
        paths=paths,
        gateway=gateway,
        tools=cfg.tools,
        serving=serving,
    )


@dataclass(frozen=True)
class RuntimeWithFormat:
    """Bundle runtime options with output formatting toggles."""

    runtime: RuntimeCliOptions
    output_format: OutputFormat


__all__ = [
    "BackendFlags",
    "JsonFlag",
    "OutputFmt",
    "OutputFormat",
    "OutputFormatCLI",
    "ProjectRoot",
    "RuntimeCLI",
    "RuntimeParam",
    "RuntimeCliError",
    "RuntimeCliOptions",
    "ProjectCLI",
    "RuntimeWithFormat",
    "Verbose",
    "build_runtime_from_cli",
    "make_root_app",
    "resolve_output_format",
    "runtime_cli_to_options",
]
