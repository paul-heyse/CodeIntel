"""Cyclopts wiring for dataset command group."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.commands.decorators import CommandConfig, cli_command
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

# Config for dataset commands - no runtime needed for listing/describing
_DATASET_NO_RUNTIME_CONFIG = CommandConfig(require_runtime=False, require_gateway=False)
# Config for dataset verify - requires runtime
_DATASET_RUNTIME_CONFIG = CommandConfig(require_runtime=True, require_gateway=True)


@cli_command("dataset.list", handler=dataset_list_handler, config=_DATASET_NO_RUNTIME_CONFIG)
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


@cli_command(
    "dataset.describe", handler=dataset_describe_handler, config=_DATASET_NO_RUNTIME_CONFIG
)
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


@cli_command("dataset.verify", handler=dataset_verify_handler, config=_DATASET_RUNTIME_CONFIG)
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


__all__ = [
    "DatasetDescribeCommand",
    "DatasetListCommand",
    "DatasetVerifyCommand",
    "dataset_app",
]
