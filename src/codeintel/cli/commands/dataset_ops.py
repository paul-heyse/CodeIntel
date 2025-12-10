"""Cyclopts wiring for dataset command group."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.commands._common import OutputFormatCLI, RuntimeCLI
from codeintel.cli.commands.context import command_context
from codeintel.cli.handlers.ops import (
    dataset_describe_handler,
    dataset_list_handler,
    dataset_verify_handler,
)
from codeintel.cli.rendering.types import OutputFormat

dataset_app = App(
    name="dataset",
    help="Dataset inspection commands.",
)


@dataset_app.command(name="list")
@dataclass
class DatasetListCommand:
    """List datasets from the registry."""

    root: Annotated[
        Path | None,
        Parameter(
            name=["--root", "-r"],
            help="Project root directory.",
        ),
    ] = None
    output_format: Annotated[
        OutputFormat,
        Parameter(
            name="--output-format",
            help="Output format.",
            show_choices=True,
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
        """Execute the dataset list command."""
        runtime_cli = RuntimeCLI(
            project_root=self.root,
            verbose=self.verbose,
        )
        output_cli = OutputFormatCLI(output_format=self.output_format)

        params: dict[str, object] = {}

        with command_context(
            "dataset.list",
            runtime_cli,
            output_cli,
            params=params,
            require_runtime=False,  # No project needed for listing datasets
        ) as (ctx, renderer):
            result = dataset_list_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


@dataset_app.command(name="describe")
@dataclass
class DatasetDescribeCommand:
    """Show contract details for a dataset."""

    table_key: Annotated[
        str,
        Parameter(
            name=None,
            help="Dataset table key (e.g., 'core.goids').",
        ),
    ]
    output_format: Annotated[
        OutputFormat,
        Parameter(
            name="--output-format",
            help="Output format.",
            show_choices=True,
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
        """Execute the dataset describe command."""
        runtime_cli = RuntimeCLI(verbose=self.verbose)
        output_cli = OutputFormatCLI(output_format=self.output_format)

        params: dict[str, object] = {
            "table_key": self.table_key,
        }

        with command_context(
            "dataset.describe",
            runtime_cli,
            output_cli,
            params=params,
            require_runtime=False,  # No project needed for describing contracts
        ) as (ctx, renderer):
            result = dataset_describe_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


@dataset_app.command(name="verify")
@dataclass
class DatasetVerifyCommand:
    """Verify dataset contracts against actual data."""

    table_key: Annotated[
        str | None,
        Parameter(
            name=None,
            help="Dataset table key to verify (verifies all if not specified).",
        ),
    ] = None
    root: Annotated[
        Path | None,
        Parameter(
            name=["--root", "-r"],
            help="Project root directory.",
        ),
    ] = None
    verbose: Annotated[
        int,
        Parameter(
            name=["-v", "--verbose"],
            help="Increase verbosity level.",
            count=True,
        ),
    ] = 0

    def __call__(self) -> None:
        """Execute the dataset verify command."""
        runtime_cli = RuntimeCLI(
            project_root=self.root,
            verbose=self.verbose,
        )
        output_cli = OutputFormatCLI()

        params: dict[str, object] = {
            "table_key": self.table_key,
        }

        with command_context(
            "dataset.verify",
            runtime_cli,
            output_cli,
            params=params,
        ) as (ctx, renderer):
            result = dataset_verify_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


__all__ = [
    "DatasetDescribeCommand",
    "DatasetListCommand",
    "DatasetVerifyCommand",
    "dataset_app",
]
