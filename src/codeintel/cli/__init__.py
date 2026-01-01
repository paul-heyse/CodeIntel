"""CodeIntel unified CLI entry point (Cyclopts-based).

This module provides:

1. CLI apps for each domain (build, datasets, docs, graphs, etc.)
2. Re-exports from the handlers package for programmatic use
3. The main entry point for the ``codeintel`` command

Examples
--------
>>> from codeintel.cli import app, main
"""

from __future__ import annotations

from codeintel.cli.commands import (
    app,
    build_app,
    dataset_app,
    datasets_ext_app,
    docs_app,
    graphs_app,
    main,
    plugins_app,
    registry_app,
    semantic_app,
    serve_app,
    storage_app,
    targets_app,
)

__all__ = [
    "app",
    "build_app",
    "dataset_app",
    "datasets_ext_app",
    "docs_app",
    "graphs_app",
    "main",
    "plugins_app",
    "registry_app",
    "semantic_app",
    "serve_app",
    "storage_app",
    "targets_app",
]
