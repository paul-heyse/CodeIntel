"""Shared Cyclopts primitives and runtime helpers for the CodeIntel CLI.

Configuration precedence
------------------------
CLI flags override the optional TOML config file (``codeintel.toml`` or
``CODEINTEL_CONFIG_PATH``), which finally falls back to defaults in function
signatures.

Execution model
---------------
The root :class:`cyclopts.App` is configured with ``result_action`` set to
``["call_if_callable", "return_value"]`` so commands can be embedded in tests
or other orchestrators without forcing ``sys.exit``. Commands implemented as
dataclasses with ``__call__`` will run naturally under this policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.config import ConfigService
from codeintel.cli.rendering.types import OutputFormat
from codeintel.cli.resolution.params import RuntimeParams


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


def _runtime_param_metadata() -> dict[str, Parameter]:
    return {"parameter": Parameter(name="*")}


def runtime_field() -> RuntimeCLI:
    """Reusable runtime field with shared Cyclopts parameter metadata.

    This function returns a dataclass ``field()`` configured for nested
    runtime CLI flags. Use this for dynamically created dataclasses.
    For static dataclasses, use ``field(default_factory=RuntimeCLI, metadata=_runtime_param_metadata())``.

    Returns
    -------
    RuntimeCLI
        Dataclass field (typed as RuntimeCLI for type checker compatibility).
    """
    return field(default_factory=RuntimeCLI, metadata=_runtime_param_metadata())


@dataclass
class OutputFormatCLI:
    """Shared output format toggles for commands supporting JSON output."""

    output_format: OutputFmt = OutputFormat.TEXT
    json: JsonFlag = False


def _output_param_metadata() -> dict[str, Parameter]:
    return {"parameter": Parameter(name="*")}


def output_field() -> OutputFormatCLI:
    """Reusable output-format field with shared Cyclopts parameter metadata.

    This function returns a dataclass ``field()`` configured for nested
    output format flags. Use this for dynamically created dataclasses.
    For static dataclasses, use ``field(default_factory=OutputFormatCLI, metadata=_output_param_metadata())``.

    Returns
    -------
    OutputFormatCLI
        Dataclass field (typed as OutputFormatCLI for type checker compatibility).
    """
    return field(default_factory=OutputFormatCLI, metadata=_output_param_metadata())


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


def get_verbose(cli: RuntimeCLI) -> int:
    """Extract verbosity count from RuntimeCLI.

    Returns
    -------
    int
        Verbosity level specified by the user.
    """
    return cli.verbose


def get_runtime_params(cli: RuntimeCLI) -> RuntimeParams:
    """Convert RuntimeCLI to canonical RuntimeParams.

    Returns
    -------
    RuntimeParams
        Canonical runtime parameters derived from CLI flags.
    """
    return RuntimeParams.from_cyclopts(cli)


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


__all__ = [
    "ExistingDir",
    "ExistingPath",
    "JsonFlag",
    "OutputFmt",
    "OutputFormat",
    "OutputFormatCLI",
    "OutputPath",
    "ProjectRoot",
    "RuntimeCLI",
    "RuntimeCliError",
    "Verbose",
    "get_output_format",
    "get_runtime_params",
    "get_verbose",
    "make_root_app",
    "output_field",
    "resolve_output_format",
    "runtime_field",
]
