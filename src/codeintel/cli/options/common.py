"""Unified option bundle for CLI commands.

This module provides CommonOptions, a single dataclass that combines all
common CLI options:
- Runtime options (project_root, repo, commit, db_path, etc.)
- Output options (output_format, json flag)
- Execution options (verbose, dry_run)
- Backend options (use_gpu)

CommonOptions replaces the scattered:
- RuntimeCLI in cyclopts_common.py
- OutputFormatCLI in cyclopts_common.py
- BackendFlags in cli_types.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from cyclopts import Parameter

from codeintel.cli.cli_types import OutputFormat


@dataclass
class CommonOptions:
    """Single option bundle for all CLI commands.

    This dataclass is designed to be embedded in Cyclopts command classes
    using ``Annotated[CommonOptions, Parameter(name="*")]`` with
    ``field(default_factory=CommonOptions)``.

    Cyclopts will flatten all fields as top-level CLI flags.

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

    Examples
    --------
    In a Cyclopts command:

    >>> @dataclass
    ... class MyCommand:
    ...     target: str = "default"
    ...     options: Annotated[CommonOptions, Parameter(name="*")] = None
    ...
    ...     def __post_init__(self) -> None:
    ...         if self.options is None:
    ...             self.options = CommonOptions()
    ...
    ...     def __call__(self) -> None:
    ...         params = self.options.to_params()
    ...         params["target"] = self.target
    ...         # Use params with execute_command()
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

    def to_params(self) -> dict[str, Any]:
        """Convert to parameter dictionary for ExecutionContext.

        Returns
        -------
        dict[str, Any]
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
    "CommonOptions",
]
