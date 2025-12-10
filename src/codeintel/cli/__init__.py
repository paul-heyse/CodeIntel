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

# Import from the commands package (canonical location)
from codeintel.cli.commands import (
    app,
    build_app,
    command_context,
    dataset_app,
    datasets_ext_app,
    docs_app,
    graphs_app,
    history_app,
    ide_app,
    main,
    op_app,
    serve_app,
    storage_app,
    subsystem_app,
)

# Re-export ops module as cyclopts_ops for backward compatibility
from codeintel.cli.commands import ops as cyclopts_ops

__all__ = [
    "app",
    "build_app",
    "command_context",
    "cyclopts_ops",
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
