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

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, cast

from cyclopts import App, Parameter
from cyclopts import config as cyclopts_config

from codeintel.cli.common_handlers import (
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

CONFIG_ENV_PREFIX = "CODEINTEL_"
CONFIG_PATH_ENV_VAR = "CODEINTEL_CONFIG_PATH"
DEFAULT_CONFIG_PATH = Path("codeintel.toml")

_ENV_CONFIG = cyclopts_config.Env(CONFIG_ENV_PREFIX)


def _resolve_config_path() -> Path:
    """Return the configured TOML path (env override or default).

    Returns
    -------
    Path
        Path to the config file, defaulting to ``codeintel.toml``.
    """
    env_path = os.environ.get(CONFIG_PATH_ENV_VAR)
    return Path(env_path) if env_path else DEFAULT_CONFIG_PATH


def _optional_toml_config(apps: object, commands: tuple[str, ...], arguments: object) -> object:
    """Apply TOML config if present; otherwise return the arguments unchanged.

    Returns
    -------
    object
        Possibly updated arguments after applying TOML overrides.
    """
    path = _resolve_config_path()
    if not path.exists():
        return arguments
    toml_loader = cast("Any", cyclopts_config.Toml(str(path)))
    app_arg = cast("App", apps)
    args_arg = cast("Any", arguments)
    return toml_loader(app_arg, commands, args_arg)


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
        config=[_optional_toml_config, _ENV_CONFIG],
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


def runtime_field() -> RuntimeCLI:
    """Reusable runtime field with shared Cyclopts parameter metadata.

    This function returns a dataclass ``field()`` configured for nested
    runtime CLI flags. The return type is ``RuntimeCLI`` rather than
    ``Field[RuntimeCLI]`` to match standard dataclass typing conventions
    (the dataclass decorator replaces fields with actual values at runtime).

    Returns
    -------
    RuntimeCLI
        Dataclass field (typed as RuntimeCLI for type checker compatibility).
    """
    return field(default_factory=RuntimeCLI, metadata={"parameter": Parameter(name="*")})


RUNTIME_PARAM_FIELD = runtime_field()


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


def output_field() -> OutputFormatCLI:
    """Reusable output-format field with shared Cyclopts parameter metadata.

    This function returns a dataclass ``field()`` configured for nested
    output format flags. The return type is ``OutputFormatCLI`` rather than
    ``Field[OutputFormatCLI]`` to match standard dataclass typing conventions
    (the dataclass decorator replaces fields with actual values at runtime).

    Returns
    -------
    OutputFormatCLI
        Dataclass field (typed as OutputFormatCLI for type checker compatibility).
    """
    return field(default_factory=OutputFormatCLI, metadata={"parameter": Parameter(name="*")})


OUTPUT_PARAM_FIELD = output_field()


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

    Returns
    -------
    ProjectRuntime
        Constructed runtime context.

    Raises
    ------
    RuntimeCliError
        If a project cannot be resolved from the provided options.
    """
    if options is None:
        options = RuntimeCLI()
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
    runtime_opts = runtime_cli_to_options(runtime_cli)
    verbose = get_verbose(runtime_cli)
    output_format = get_output_format(output_cli, default=default_output)
    return runtime_opts, verbose, output_format


__all__ = [
    "OUTPUT_PARAM_FIELD",
    "RUNTIME_PARAM_FIELD",
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
    "get_output_format",
    "get_verbose",
    "make_handler_context",
    "make_root_app",
    "output_field",
    "resolve_output_format",
    "runtime_cli_to_options",
]
