"""Root Cyclopts application for the CodeIntel CLI."""

from __future__ import annotations

import os
import sys

from cyclopts import App

from codeintel.cli.cli_errors import OutputFormat, handle_cli_error
from codeintel.cli.cyclopts_build import build_app
from codeintel.cli.cyclopts_common import make_root_app
from codeintel.cli.cyclopts_config import config_app
from codeintel.cli.cyclopts_datasets import datasets_ext_app
from codeintel.cli.cyclopts_docs import docs_app
from codeintel.cli.cyclopts_graphs import graphs_app
from codeintel.cli.cyclopts_help import build_patched_app
from codeintel.cli.cyclopts_history import history_app
from codeintel.cli.cyclopts_ide import ide_app
from codeintel.cli.cyclopts_ops import dataset_app, op_app, serve_app, set_root_app
from codeintel.cli.cyclopts_storage import storage_app
from codeintel.cli.cyclopts_subsystem import subsystem_app

app: App = build_patched_app(make_root_app)
set_root_app(app)

# Shell completion support
app.register_install_completion_command(
    name="--install-completion",
    help="Install shell completion for bash/zsh/fish.",
)

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
app.command(ide_app, name="ide")
app.command(subsystem_app, name="subsystem")

# Utilities
app.command(config_app, name="config")


def _detect_output_format() -> OutputFormat:
    """Detect output format from environment or CLI args.

    Check for JSON output request via environment variable or CLI flags
    before parsing begins. This enables structured error output even
    when parsing fails.

    Returns
    -------
    OutputFormat
        Detected output format preference.
    """
    # Check environment variable first
    env_format = os.environ.get("CODEINTEL_OUTPUT_FORMAT", "").lower()
    if env_format == "json":
        return OutputFormat.JSON

    # Check for --json flag in argv
    if "--json" in sys.argv:
        return OutputFormat.JSON

    # Check for --output-format json
    for i, arg in enumerate(sys.argv[:-1]):
        if arg == "--output-format" and sys.argv[i + 1].lower() == "json":
            return OutputFormat.JSON

    return OutputFormat.TEXT


def main() -> None:
    """Entry point used by console_scripts.

    Raises
    ------
    SystemExit
        Propagated with normalized CLI exit codes on failure.
    """
    output_format = _detect_output_format()
    try:
        app()
    except BaseException as exc:
        exit_code = handle_cli_error(exc, sys.stderr, output_format=output_format)
        raise SystemExit(exit_code) from exc


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
