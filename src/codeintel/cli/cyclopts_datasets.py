"""Cyclopts wiring for dataset management commands.

This module wires Cyclopts command classes to unified handlers via command_context.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import App, Parameter

from codeintel.cli.cli_types import OutputFormat
from codeintel.cli.command_context import command_context
from codeintel.cli.cyclopts_common import OutputFormatCLI, RuntimeCLI
from codeintel.cli.handlers.datasets import (
    datasets_diff_handler,
    datasets_lint_handler,
    datasets_list_handler,
    datasets_snapshot_handler,
)

datasets_ext_app = App(
    name="datasets",
    help="Extended dataset management commands.",
)


DocsFilterMode = Literal["include", "only", "exclude"]
ReadOnlyFilterMode = Literal["include", "only", "exclude"]


class SamplingStrictness(Enum):
    """Strictness policy when sampling rows."""

    STRICT = "strict"
    LENIENT = "lenient"


class OverwritePolicy(Enum):
    """Behavior when scaffold outputs already exist."""

    OVERWRITE = "overwrite"
    SKIP = "skip"
    ERROR = "error"


class BootstrapSnippet(Enum):
    """Whether to emit a bootstrap snippet during scaffold."""

    EMIT = "emit"
    SKIP = "skip"


@datasets_ext_app.command(name="lint")
@dataclass
class LintCommand:
    """Validate dataset contract health."""

    schema_dir: Annotated[
        Path,
        Parameter(
            name="--schema-dir",
            help="Directory containing export JSON Schemas.",
        ),
    ] = Path("src/codeintel/config/schemas/export")
    sampling: Annotated[
        str,
        Parameter(
            name="--sampling",
            help="Sampling mode: enabled or disabled.",
        ),
    ] = "disabled"
    project_root: Annotated[
        Path | None,
        Parameter(
            name="--root",
            help="Project root directory.",
        ),
    ] = None
    output_format: Annotated[
        OutputFormat,
        Parameter(
            name="--output-format",
            help="Output format (text or json).",
        ),
    ] = OutputFormat.TEXT
    verbose: Annotated[
        int,
        Parameter(
            name=["-v", "--verbose"],
            help="Increase verbosity level.",
            count=True,
        ),
    ] = 0

    def __call__(self) -> None:
        """Execute the datasets lint command."""
        runtime_cli = RuntimeCLI(
            project_root=self.project_root,
            verbose=self.verbose,
        )
        output_cli = OutputFormatCLI(output_format=self.output_format)

        params: dict[str, object] = {
            "schema_dir": str(self.schema_dir),
            "sampling": self.sampling,
        }

        with command_context(
            "datasets.lint",
            runtime_cli,
            output_cli,
            params=params,
        ) as (ctx, renderer):
            result = datasets_lint_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


@datasets_ext_app.command(name="list")
@dataclass
class ListDatasetsCommand:
    """List datasets with capabilities and optional filters."""

    docs_view: Annotated[
        DocsFilterMode,
        Parameter(
            name="--docs-view",
            help='Docs view filter: "include", "exclude", or "only".',
        ),
    ] = "include"
    read_only: Annotated[
        ReadOnlyFilterMode,
        Parameter(
            name="--read-only",
            help='Read-only filter: "include", "exclude", or "only".',
        ),
    ] = "include"
    max_description: Annotated[
        int,
        Parameter(
            name="--max-description",
            help="Maximum description length before truncation.",
        ),
    ] = 80
    project_root: Annotated[
        Path | None,
        Parameter(
            name="--root",
            help="Project root directory.",
        ),
    ] = None
    output_format: Annotated[
        OutputFormat,
        Parameter(
            name="--output-format",
            help="Output format (text or json).",
        ),
    ] = OutputFormat.TEXT
    verbose: Annotated[
        int,
        Parameter(
            name=["-v", "--verbose"],
            help="Increase verbosity level.",
            count=True,
        ),
    ] = 0

    def __call__(self) -> None:
        """Execute the datasets list command."""
        runtime_cli = RuntimeCLI(
            project_root=self.project_root,
            verbose=self.verbose,
        )
        output_cli = OutputFormatCLI(output_format=self.output_format)

        params: dict[str, object] = {
            "docs_view": self.docs_view,
            "read_only": self.read_only,
            "max_description": self.max_description,
        }

        with command_context(
            "datasets.list",
            runtime_cli,
            output_cli,
            params=params,
        ) as (ctx, renderer):
            result = datasets_list_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


@datasets_ext_app.command(name="snapshot")
@dataclass
class SnapshotCommand:
    """Write current dataset specs to a JSON snapshot file."""

    output: Annotated[
        Path,
        Parameter(
            name="--output",
            help="Output file path for JSON dataset specs.",
        ),
    ] = Path("build/dataset_specs.json")
    project_root: Annotated[
        Path | None,
        Parameter(
            name="--root",
            help="Project root directory.",
        ),
    ] = None
    output_format: Annotated[
        OutputFormat,
        Parameter(
            name="--output-format",
            help="Output format (text or json).",
        ),
    ] = OutputFormat.TEXT
    verbose: Annotated[
        int,
        Parameter(
            name=["-v", "--verbose"],
            help="Increase verbosity level.",
            count=True,
        ),
    ] = 0

    def __call__(self) -> None:
        """Execute the datasets snapshot command."""
        runtime_cli = RuntimeCLI(
            project_root=self.project_root,
            verbose=self.verbose,
        )
        output_cli = OutputFormatCLI(output_format=self.output_format)

        params: dict[str, object] = {
            "output": str(self.output),
        }

        with command_context(
            "datasets.snapshot",
            runtime_cli,
            output_cli,
            params=params,
        ) as (ctx, renderer):
            result = datasets_snapshot_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


@datasets_ext_app.command(name="diff")
@dataclass
class DiffCommand:
    """Diff current dataset specs against a baseline."""

    baseline: Annotated[
        Path | None,
        Parameter(
            name="--baseline",
            help="Path to JSON baseline from `codeintel datasets snapshot`.",
        ),
    ] = None
    output: Annotated[
        Path | None,
        Parameter(
            name="--output",
            help="Optional output file path for writing current specs.",
        ),
    ] = None
    against_ref: Annotated[
        str | None,
        Parameter(
            name="--against-ref",
            help="Git ref to diff against (e.g. HEAD~, main).",
        ),
    ] = None
    baseline_path: Annotated[
        Path,
        Parameter(
            name="--baseline-path",
            help="Path of the snapshot file inside the git ref.",
        ),
    ] = Path("build/dataset_specs.json")
    project_root: Annotated[
        Path | None,
        Parameter(
            name="--root",
            help="Project root directory.",
        ),
    ] = None
    output_format: Annotated[
        OutputFormat,
        Parameter(
            name="--output-format",
            help="Output format (text or json).",
        ),
    ] = OutputFormat.TEXT
    verbose: Annotated[
        int,
        Parameter(
            name=["-v", "--verbose"],
            help="Increase verbosity level.",
            count=True,
        ),
    ] = 0

    def __call__(self) -> None:
        """Execute the datasets diff command."""
        runtime_cli = RuntimeCLI(
            project_root=self.project_root,
            verbose=self.verbose,
        )
        output_cli = OutputFormatCLI(output_format=self.output_format)

        # Use baseline if provided, otherwise construct from baseline_path
        baseline_file = self.baseline or self.baseline_path
        params: dict[str, object] = {
            "baseline_path": str(baseline_file),
            "output": str(self.output) if self.output else None,
            "against_ref": self.against_ref,
        }

        with command_context(
            "datasets.diff",
            runtime_cli,
            output_cli,
            params=params,
        ) as (ctx, renderer):
            result = datasets_diff_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


# Note: The following commands (conformance, generate-schemas, catalog, scaffold,
# validate-files) require more complex handlers that are not yet fully migrated.
# They are temporarily removed from this module.
# To restore them, add handlers to codeintel.cli.handlers.datasets.

__all__ = ["datasets_ext_app"]
