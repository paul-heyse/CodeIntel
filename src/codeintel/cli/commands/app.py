"""Root Cyclopts application for the CodeIntel CLI."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from codeintel.cli.commands._common import make_root_app
from codeintel.cli.commands._help import build_patched_app
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
from codeintel.cli.commands.ops import op_app, set_root_app
from codeintel.cli.commands.plugins import plugins_app
from codeintel.cli.commands.serve import serve_app
from codeintel.cli.commands.storage import storage_app
from codeintel.cli.commands.subsystem import subsystem_app
from codeintel.cli.errors import OutputFormat, handle_cli_error

if TYPE_CHECKING:
    from collections.abc import Callable

    from cyclopts import App


_init_plugins: Callable[..., object] | None
try:
    from codeintel.cli.plugins import initialize_plugins as _init_plugins
except ImportError:
    _init_plugins = None

app: App = build_patched_app(make_root_app)
set_root_app(app)


app.register_install_completion_command(
    name="--install-completion",
    help="Install shell completion for bash/zsh/fish.",
)


app.command(build_app, name="build")
app.command(op_app, name="op")
app.command(dataset_app, name="dataset")
app.command(serve_app, name="serve")


app.command(graphs_app, name="graph")
app.command(docs_app, name="docs")
app.command(storage_app, name="storage")
app.command(history_app, name="history")
app.command(datasets_ext_app, name="datasets")
app.command(ide_app, name="ide")
app.command(subsystem_app, name="subsystem")


app.command(config_app, name="config")
app.command(health_app, name="health")
app.command(jobs_app, name="jobs")
app.command(plugins_app, name="plugins")
app.command(completions_app, name="completions")
app.command(help_commands_app, name="help-ops")


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
    env_format = os.environ.get("CODEINTEL_OUTPUT_FORMAT", "").lower()
    if env_format == "json":
        return OutputFormat.JSON

    if "--json" in sys.argv:
        return OutputFormat.JSON

    for i, arg in enumerate(sys.argv[:-1]):
        if arg == "--output-format" and sys.argv[i + 1].lower() == "json":
            return OutputFormat.JSON

    return OutputFormat.TEXT


def _initialize_cli() -> None:
    """Initialize CLI infrastructure.

    Register operations and load plugins before running commands.
    """
    if _init_plugins is not None:
        _init_plugins()


def main() -> None:
    """Entry point used by console_scripts.

    Raises
    ------
    SystemExit
        Propagated with normalized CLI exit codes on failure.
    """
    output_format = _detect_output_format()

    _initialize_cli()

    try:
        app()
    except BaseException as exc:
        exit_code = handle_cli_error(exc, sys.stderr, output_format=output_format)
        raise SystemExit(exit_code) from exc


__all__ = [
    "app",
    "build_app",
    "completions_app",
    "dataset_app",
    "datasets_ext_app",
    "docs_app",
    "graphs_app",
    "health_app",
    "help_commands_app",
    "history_app",
    "jobs_app",
    "op_app",
    "plugins_app",
    "serve_app",
    "storage_app",
]
