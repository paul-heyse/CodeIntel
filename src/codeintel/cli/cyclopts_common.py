"""Shared Cyclopts primitives and runtime helpers for the CodeIntel CLI.

Configuration precedence
------------------------
CLI flags override environment variables (``CODEINTEL_*``), which override the
optional TOML config file (``codeintel.toml`` or ``CODEINTEL_CONFIG_PATH``),
which finally fall back to defaults in function signatures.

Execution model
---------------
The root :class:`cyclopts.App` is configured with ``result_action`` set to
``["call_if_callable", "return_value"]`` so commands can be embedded in tests
or other orchestrators without forcing ``sys.exit``. Commands implemented as
dataclasses with ``__call__`` will run naturally under this policy.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.cli_types import BackendFlags, OutputFormat
from codeintel.cli.command_context import command_context
from codeintel.cli.common_handlers import RuntimeCliOptions, build_config_from_options
from codeintel.cli.config import ConfigService
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

    Use ConfigService for unified configuration loading with proper precedence.

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
        config=ConfigService.get_cyclopts_config_chain(),
        result_action=["call_if_callable", "return_value"],
        print_error=True,
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

# Reusable path aliases (no runtime validator to avoid heavy dependency).
ExistingPath = Annotated[
    Path,
    Parameter(
        help="Path that should exist.",
    ),
]

ExistingDir = Annotated[
    Path,
    Parameter(
        help="Directory path that should exist.",
    ),
]

OutputPath = Annotated[
    Path,
    Parameter(
        help="Output file path (parent directory should exist).",
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


RuntimeParam = RuntimeCLI


RUNTIME_PARAM_METADATA: dict[str, Parameter] = {"parameter": Parameter(name="*")}
"""Metadata for RuntimeCLI fields to enable Cyclopts nested parameter flattening."""


def runtime_field() -> RuntimeCLI:
    """Reusable runtime field with shared Cyclopts parameter metadata.

    This function returns a dataclass ``field()`` configured for nested
    runtime CLI flags. Use this for dynamically created dataclasses.
    For static dataclasses, use ``field(default_factory=RuntimeCLI, metadata=RUNTIME_PARAM_METADATA)``.

    Returns
    -------
    RuntimeCLI
        Dataclass field (typed as RuntimeCLI for type checker compatibility).
    """
    return field(default_factory=RuntimeCLI, metadata=RUNTIME_PARAM_METADATA)


@dataclass
class ProjectCLI:
    """Bundle runtime selection under a project alias."""

    runtime: RuntimeParam = field(default_factory=RuntimeCLI)


StorageCLI = ProjectCLI


@dataclass
class OutputFormatCLI:
    """Shared output format toggles for commands supporting JSON output."""

    output_format: OutputFmt = OutputFormat.TEXT
    json: JsonFlag = False


OutputParam = OutputFormatCLI


OUTPUT_PARAM_METADATA: dict[str, Parameter] = {"parameter": Parameter(name="*")}
"""Metadata for OutputFormatCLI fields to enable Cyclopts nested parameter flattening."""


def output_field() -> OutputFormatCLI:
    """Reusable output-format field with shared Cyclopts parameter metadata.

    This function returns a dataclass ``field()`` configured for nested
    output format flags. Use this for dynamically created dataclasses.
    For static dataclasses, use ``field(default_factory=OutputFormatCLI, metadata=OUTPUT_PARAM_METADATA)``.

    Returns
    -------
    OutputFormatCLI
        Dataclass field (typed as OutputFormatCLI for type checker compatibility).
    """
    return field(default_factory=OutputFormatCLI, metadata=OUTPUT_PARAM_METADATA)


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
    cli: RuntimeCLI | None, *, backend: BackendFlags | None = None
) -> RuntimeCliOptions:
    """Convert a RuntimeCLI dataclass to RuntimeCliOptions.

    .. deprecated:: 2.0
        Use ``RuntimeParams.from_cyclopts()`` instead.
        This function will be removed in version 3.0.

    Returns
    -------
    RuntimeCliOptions
        Options object suitable for runtime construction.
    """
    warnings.warn(
        "runtime_cli_to_options is deprecated. Use RuntimeParams.from_cyclopts() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    resolved_cli = cli or RuntimeCLI()
    return RuntimeCliOptions(
        project_root=resolved_cli.project_root,
        repo=resolved_cli.repo,
        commit=resolved_cli.commit,
        db_path=resolved_cli.db_path,
        build_dir=resolved_cli.build_dir,
        repo_root=resolved_cli.repo_root,
        document_output_dir=resolved_cli.document_output_dir,
        backend=backend or BackendFlags(),
    )


def _runtime_cli_to_options_internal(
    cli: RuntimeCLI | None, *, backend: BackendFlags | None = None
) -> RuntimeCliOptions:
    """Convert a RuntimeCLI dataclass to RuntimeCliOptions (internal, no warning).

    Parameters
    ----------
    cli
        RuntimeCLI instance or None.
    backend
        Backend flags or None.

    Returns
    -------
    RuntimeCliOptions
        Options object suitable for runtime construction.
    """
    resolved_cli = cli or RuntimeCLI()
    return RuntimeCliOptions(
        project_root=resolved_cli.project_root,
        repo=resolved_cli.repo,
        commit=resolved_cli.commit,
        db_path=resolved_cli.db_path,
        build_dir=resolved_cli.build_dir,
        repo_root=resolved_cli.repo_root,
        document_output_dir=resolved_cli.document_output_dir,
        backend=backend or BackendFlags(),
    )


def build_runtime_from_cli(
    options: RuntimeCliOptions | RuntimeCLI | None,
    *,
    allow_fallback: bool = True,
) -> ProjectRuntime:
    """Build a :class:`ProjectRuntime` from CLI options without Typer exits.

    .. deprecated:: 2.0
        Use ``RuntimeResolver.resolve(RuntimeParams)`` instead.
        This function will be removed in version 3.0.

    Returns
    -------
    ProjectRuntime
        Constructed runtime context.

    Raises
    ------
    RuntimeCliError
        If a project cannot be resolved from the provided options.
    """
    warnings.warn(
        "build_runtime_from_cli is deprecated. Use RuntimeResolver.resolve(RuntimeParams) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if options is None:
        options = RuntimeCLI()
    if isinstance(options, RuntimeCLI):
        options = _runtime_cli_to_options_internal(options)

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


def get_verbose(cli: RuntimeCLI) -> int:
    """Extract verbosity count from RuntimeCLI.

    Returns
    -------
    int
        Verbosity level specified by the user.
    """
    return cli.verbose


def get_output_format(
    cli: OutputFormatCLI, *, default: OutputFormat = OutputFormat.TEXT
) -> OutputFormat:
    """Resolve the output format from OutputFormatCLI.

    Returns
    -------
    OutputFormat
        Effective output format after applying CLI toggles.
    """
    return resolve_output_format(
        json_flag=cli.json,
        explicit=cli.output_format,
        default=default,
    )


def make_handler_context(
    runtime_cli: RuntimeCLI,
    output_cli: OutputFormatCLI,
    *,
    default_output: OutputFormat,
) -> tuple[RuntimeCliOptions, int, OutputFormat]:
    """Return runtime options, verbosity, and output format for handlers.

    Returns
    -------
    tuple[RuntimeCliOptions, int, OutputFormat]
        Runtime options, verbosity count, and output format.
    """
    runtime_opts = _runtime_cli_to_options_internal(runtime_cli)
    verbose = get_verbose(runtime_cli)
    output_format = get_output_format(output_cli, default=default_output)
    return runtime_opts, verbose, output_format


# command_context is imported from codeintel.cli.command_context
# and re-exported here for backwards compatibility and convenience


__all__ = [
    "OUTPUT_PARAM_METADATA",
    "RUNTIME_PARAM_METADATA",
    "BackendFlags",
    "ExistingDir",
    "ExistingPath",
    "JsonFlag",
    "OutputFmt",
    "OutputFormat",
    "OutputFormatCLI",
    "OutputParam",
    "OutputPath",
    "ProjectCLI",
    "ProjectRoot",
    "RuntimeCLI",
    "RuntimeCliError",
    "RuntimeCliOptions",
    "RuntimeParam",
    "RuntimeWithFormat",
    "StorageCLI",
    "Verbose",
    "build_runtime_from_cli",
    "command_context",
    "get_output_format",
    "get_verbose",
    "make_handler_context",
    "make_root_app",
    "output_field",
    "resolve_output_format",
    "runtime_cli_to_options",
    "runtime_field",
]
