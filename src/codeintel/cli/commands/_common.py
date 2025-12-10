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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.config import ConfigService
from codeintel.cli.rendering.types import OutputFormat


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


# =============================================================================
# CommonOptions - Unified Option Bundle
# =============================================================================


@dataclass
class CommonOptions:
    """Single option bundle combining runtime, output, and execution options.

    This dataclass combines all common CLI options into a single bundle that
    can be embedded in Cyclopts command classes and converted to a params dict.

    Parameters
    ----------
    project_root
        Explicit project root directory. If None, auto-discovery is used.
    repo
        Repository slug (e.g., "org/repo"). Required if no project file.
    commit
        Commit SHA. Required if no project file.
    db_path
        Path to DuckDB database. Defaults to build/db/codeintel.duckdb.
    build_dir
        Build directory. Defaults to build/.
    repo_root
        Repository root. Defaults to current directory.
    document_output_dir
        Override for document output directory.
    output_format
        Output format (text or json).
    json
        Shorthand for --output-format json.
    verbose
        Verbosity level (0=warning, 1=info, 2+=debug).
    dry_run
        If True, show what would be done without doing it.
    use_gpu
        Enable GPU acceleration for graph operations.
    """

    # Runtime selection
    project_root: Annotated[
        Path | None,
        Parameter(
            name=["--root", "-r"],
            help="Explicit project root directory.",
        ),
    ] = None

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
            help="Override document output directory.",
        ),
    ] = None

    # Output control
    output_format: Annotated[
        OutputFormat,
        Parameter(
            name="--output-format",
            help="Output format.",
            show_choices=True,
        ),
    ] = OutputFormat.TEXT

    json: Annotated[
        bool,
        Parameter(
            name="--json",
            help="Alias for --output-format json.",
            negative=(),
        ),
    ] = False

    # Execution control
    verbose: Annotated[
        int,
        Parameter(
            name=["--verbose", "-v"],
            help="Increase verbosity (0=warnings, 1=info, 2=debug).",
            count=True,
        ),
    ] = 0

    dry_run: Annotated[
        bool,
        Parameter(
            name=["--dry-run", "-n"],
            help="Show what would be done without doing it.",
            negative=(),
        ),
    ] = False

    # Backend control
    use_gpu: Annotated[
        bool,
        Parameter(
            name="--gpu",
            help="Enable GPU acceleration for graph operations.",
            negative=(),
        ),
    ] = False

    def to_params(self) -> dict[str, object]:
        """Convert to parameter dictionary for ExecutionContext.

        Returns
        -------
        dict[str, object]
            All options as a flat dictionary.
        """
        return {
            "project_root": self.project_root,
            "repo": self.repo,
            "commit": self.commit,
            "db_path": self.db_path,
            "build_dir": self.build_dir,
            "repo_root": self.repo_root,
            "document_output_dir": self.document_output_dir,
            "output_format": self.resolve_output_format(),
            "verbose": self.verbose,
            "dry_run": self.dry_run,
            "use_gpu": self.use_gpu,
        }

    def resolve_output_format(self) -> OutputFormat:
        """Resolve output format with json flag precedence.

        Returns
        -------
        OutputFormat
            JSON if json flag is True, otherwise output_format.
        """
        return OutputFormat.JSON if self.json else self.output_format


# Metadata for Cyclopts parameter flattening
COMMON_OPTIONS_METADATA: dict[str, Parameter] = {"parameter": Parameter(name="*")}


__all__ = [
    "COMMON_OPTIONS_METADATA",
    "OUTPUT_PARAM_METADATA",
    "RUNTIME_PARAM_METADATA",
    "CommonOptions",
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
    "RuntimeParam",
    "StorageCLI",
    "Verbose",
    "get_output_format",
    "get_verbose",
    "make_root_app",
    "output_field",
    "resolve_output_format",
    "runtime_field",
]
