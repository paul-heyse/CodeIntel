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

# Main app and entry point
from codeintel.cli.cyclopts_app import app, main

# Command apps - Core
from codeintel.cli.cyclopts_build import build_app

# Common utilities for command definitions
from codeintel.cli.cyclopts_common import (
    OutputFormatCLI,
    RuntimeCLI,
    make_root_app,
)

# Command apps - Utilities
from codeintel.cli.cyclopts_completions import completions_app
from codeintel.cli.cyclopts_config import config_app

# Command apps - Domain
from codeintel.cli.cyclopts_datasets import datasets_ext_app
from codeintel.cli.cyclopts_docs import docs_app
from codeintel.cli.cyclopts_graphs import graphs_app
from codeintel.cli.cyclopts_health import health_app
from codeintel.cli.cyclopts_help_commands import help_commands_app
from codeintel.cli.cyclopts_history import history_app
from codeintel.cli.cyclopts_ide import ide_app
from codeintel.cli.cyclopts_jobs import jobs_app
from codeintel.cli.cyclopts_ops import dataset_app, op_app, serve_app
from codeintel.cli.cyclopts_plugins import plugins_app
from codeintel.cli.cyclopts_storage import storage_app
from codeintel.cli.cyclopts_subsystem import subsystem_app

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
    "graphs_app",
    "health_app",
    "help_commands_app",
    "history_app",
    "ide_app",
    "jobs_app",
    "main",
    "make_root_app",
    "op_app",
    "plugins_app",
    "serve_app",
    "storage_app",
    "subsystem_app",
]
