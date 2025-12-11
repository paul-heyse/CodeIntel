"""CLI commands package.

Provide a unified public API for CLI command modules. This package re-exports
command apps and utilities from their canonical locations while providing a
clean import path for consumers.

The commands package serves as the public interface for:
- Command app definitions (build_app, docs_app, etc.)
- Command utilities (RuntimeCLI, OutputFormatCLI, etc.)
- Main app entry points

Example
-------
>>> from codeintel.cli.commands import app, build_app
>>> app.command(build_app, name="build")
"""

from __future__ import annotations

from codeintel.cli.commands import _help as help_utils
from codeintel.cli.commands._common import (
    OutputFormatCLI,
    RuntimeCLI,
    get_output_format,
    get_verbose,
    make_root_app,
    resolve_output_format,
    runtime_field,
)
from codeintel.cli.commands.app import app, main
from codeintel.cli.commands.build import build_app
from codeintel.cli.commands.completions import completions_app
from codeintel.cli.commands.config import config_app
from codeintel.cli.commands.dataset_ops import dataset_app
from codeintel.cli.commands.datasets import datasets_ext_app
from codeintel.cli.commands.docs import docs_app
from codeintel.cli.commands.graphs import graphs_app
from codeintel.cli.commands.health import health_app
from codeintel.cli.commands.help_commands import help_commands_app
from codeintel.cli.commands.history import history_app
from codeintel.cli.commands.ide import ide_app
from codeintel.cli.commands.jobs import jobs_app
from codeintel.cli.commands.ops import op_app
from codeintel.cli.commands.plugins import plugins_app
from codeintel.cli.commands.serve import serve_app
from codeintel.cli.commands.storage import storage_app
from codeintel.cli.commands.subsystem import subsystem_app

__all__ = [
    "OutputFormatCLI",
    "RuntimeCLI",
    "app",
    "build_app",
    "completions_app",
    "config_app",
    "dataset_app",
    "datasets_ext_app",
    "docs_app",
    "get_output_format",
    "get_verbose",
    "graphs_app",
    "health_app",
    "help_commands_app",
    "help_utils",
    "history_app",
    "ide_app",
    "jobs_app",
    "main",
    "make_root_app",
    "op_app",
    "plugins_app",
    "resolve_output_format",
    "runtime_field",
    "serve_app",
    "storage_app",
    "subsystem_app",
]
