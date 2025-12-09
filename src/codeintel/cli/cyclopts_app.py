"""Root Cyclopts application for the CodeIntel CLI."""

from __future__ import annotations

from cyclopts import App

from codeintel.cli.cyclopts_build import build_app
from codeintel.cli.cyclopts_common import make_root_app
from codeintel.cli.cyclopts_datasets import datasets_ext_app
from codeintel.cli.cyclopts_docs import docs_app
from codeintel.cli.cyclopts_graphs import graphs_app
from codeintel.cli.cyclopts_history import history_app
from codeintel.cli.cyclopts_ops import dataset_app, op_app, serve_app
from codeintel.cli.cyclopts_storage import storage_app

app: App = make_root_app()

# Core
app.command(build_app, name="build")
app.command(op_app, name="op")
app.command(dataset_app, name="dataset")
app.command(serve_app, name="serve")

# Domain
app.command(graphs_app, name="graph")
app.command(docs_app, name="docs")
app.command(storage_app, name="storage")
app.command(history_app, name="history")
app.command(datasets_ext_app, name="datasets")


def main() -> None:
    """Entry point used by console_scripts."""
    app()


__all__ = [
    "app",
    "build_app",
    "dataset_app",
    "datasets_ext_app",
    "docs_app",
    "graphs_app",
    "history_app",
    "op_app",
    "serve_app",
    "storage_app",
]
