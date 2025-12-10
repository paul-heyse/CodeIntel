"""CodeIntel unified CLI entry point (Cyclopts-based).

This module provides:

1. CLI apps for each domain (build, datasets, docs, graphs, etc.)
2. Re-exports from the handlers package for programmatic use
3. The main entry point for the ``codeintel`` command

Examples
--------
>>> from codeintel.cli import app, main
>>> from codeintel.cli.handlers import ide_hints_handler, EnhancedHandlerContext
"""

from __future__ import annotations

from codeintel.cli.command_context import command_context
from codeintel.cli.cyclopts_app import app
from codeintel.cli.cyclopts_build import build_app
from codeintel.cli.cyclopts_datasets import datasets_ext_app
from codeintel.cli.cyclopts_docs import docs_app
from codeintel.cli.cyclopts_graphs import graphs_app
from codeintel.cli.cyclopts_history import history_app
from codeintel.cli.cyclopts_ide import ide_app
from codeintel.cli.cyclopts_ops import dataset_app, op_app, serve_app
from codeintel.cli.cyclopts_storage import storage_app
from codeintel.cli.cyclopts_subsystem import subsystem_app


def main() -> None:
    """Entry point for the codeintel CLI."""
    app()


__all__ = [
    "app",
    "build_app",
    "command_context",
    "dataset_app",
    "datasets_ext_app",
    "docs_app",
    "graphs_app",
    "history_app",
    "ide_app",
    "main",
    "op_app",
    "serve_app",
    "storage_app",
    "subsystem_app",
]
