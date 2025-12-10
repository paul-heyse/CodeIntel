"""Plugin management commands.

Provide commands to discover, list, and inspect CLI plugins.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.commands._common import OutputFormatCLI, RuntimeCLI
from codeintel.cli.commands.context import command_context
from codeintel.cli.handlers.plugins import (
    plugins_discover_handler,
    plugins_info_handler,
    plugins_list_handler,
    plugins_new_handler,
    plugins_paths_handler,
    plugins_test_handler,
    plugins_validate_handler,
)
from codeintel.cli.rendering.types import OutputFormat

plugins_app = App(name="plugins", help="Manage CLI plugins")


@plugins_app.command(name="list")
@dataclass
class PluginsListCommand:
    """List installed plugins.

    Display a table of all loaded plugins with their version,
    operation count, and description.
    """

    output_format: Annotated[
        OutputFormat,
        Parameter(name="--format", help="Output format"),
    ] = OutputFormat.TEXT

    def __call__(self) -> None:
        """Execute the plugins list command."""
        runtime_cli = RuntimeCLI()
        output_cli = OutputFormatCLI(output_format=self.output_format)

        with command_context(
            "plugins.list",
            runtime_cli,
            output_cli,
            params={},
            require_runtime=False,
        ) as (ctx, renderer):
            result = plugins_list_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


@plugins_app.command(name="discover")
@dataclass
class PluginsDiscoverCommand:
    """Discover available plugins.

    Search plugin directories for available plugins and show
    their loading status.
    """

    output_format: Annotated[
        OutputFormat,
        Parameter(name="--format", help="Output format"),
    ] = OutputFormat.TEXT

    def __call__(self) -> None:
        """Execute the plugins discover command."""
        runtime_cli = RuntimeCLI()
        output_cli = OutputFormatCLI(output_format=self.output_format)

        with command_context(
            "plugins.discover",
            runtime_cli,
            output_cli,
            params={},
            require_runtime=False,
        ) as (ctx, renderer):
            result = plugins_discover_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


@plugins_app.command(name="info")
@dataclass
class PluginsInfoCommand:
    """Show details about a plugin.

    Display detailed information about a specific loaded plugin
    including its operations and file path.
    """

    name: Annotated[str, Parameter(help="Plugin name")]
    output_format: Annotated[
        OutputFormat,
        Parameter(name="--format", help="Output format"),
    ] = OutputFormat.TEXT

    def __call__(self) -> None:
        """Execute the plugins info command."""
        runtime_cli = RuntimeCLI()
        output_cli = OutputFormatCLI(output_format=self.output_format)

        params: dict[str, object] = {"name": self.name}

        with command_context(
            "plugins.info",
            runtime_cli,
            output_cli,
            params=params,
            require_runtime=False,
        ) as (ctx, renderer):
            result = plugins_info_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


@plugins_app.command(name="paths")
@dataclass
class PluginsPathsCommand:
    """Show plugin search paths.

    Display all directories where plugins are searched,
    with indicators for which paths exist.
    """

    output_format: Annotated[
        OutputFormat,
        Parameter(name="--format", help="Output format"),
    ] = OutputFormat.TEXT

    def __call__(self) -> None:
        """Execute the plugins paths command."""
        runtime_cli = RuntimeCLI()
        output_cli = OutputFormatCLI(output_format=self.output_format)

        with command_context(
            "plugins.paths",
            runtime_cli,
            output_cli,
            params={},
            require_runtime=False,
        ) as (ctx, renderer):
            result = plugins_paths_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


@plugins_app.command(name="new")
@dataclass
class PluginsNewCommand:
    """Create new plugin from template.

    Generate a plugin scaffold with manifest, entry point,
    and test files.
    """

    name: Annotated[str, Parameter(help="Plugin name")]
    output: Annotated[
        Path | None,
        Parameter(name="--output", help="Output directory"),
    ] = None
    output_format: Annotated[
        OutputFormat,
        Parameter(name="--format", help="Output format"),
    ] = OutputFormat.TEXT

    def __call__(self) -> None:
        """Execute the plugins new command."""
        runtime_cli = RuntimeCLI()
        output_cli = OutputFormatCLI(output_format=self.output_format)

        params: dict[str, object] = {
            "name": self.name,
            "output": str(self.output) if self.output else None,
        }

        with command_context(
            "plugins.new",
            runtime_cli,
            output_cli,
            params=params,
            require_runtime=False,
        ) as (ctx, renderer):
            result = plugins_new_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


@plugins_app.command(name="test")
@dataclass
class PluginsTestCommand:
    """Test a plugin.

    Run the test harness to validate plugin manifest, loading,
    and operation registration.
    """

    path: Annotated[Path, Parameter(help="Plugin directory")]
    output_format: Annotated[
        OutputFormat,
        Parameter(name="--format", help="Output format"),
    ] = OutputFormat.TEXT

    def __call__(self) -> None:
        """Execute the plugins test command."""
        runtime_cli = RuntimeCLI()
        output_cli = OutputFormatCLI(output_format=self.output_format)

        params: dict[str, object] = {"path": str(self.path)}

        with command_context(
            "plugins.test",
            runtime_cli,
            output_cli,
            params=params,
            require_runtime=False,
        ) as (ctx, renderer):
            result = plugins_test_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


@plugins_app.command(name="validate")
@dataclass
class PluginsValidateCommand:
    """Validate plugin manifest.

    Check that the plugin manifest is valid and compatible
    with the current CLI version.
    """

    path: Annotated[Path, Parameter(help="Plugin directory")]
    output_format: Annotated[
        OutputFormat,
        Parameter(name="--format", help="Output format"),
    ] = OutputFormat.TEXT

    def __call__(self) -> None:
        """Execute the plugins validate command."""
        runtime_cli = RuntimeCLI()
        output_cli = OutputFormatCLI(output_format=self.output_format)

        params: dict[str, object] = {"path": str(self.path)}

        with command_context(
            "plugins.validate",
            runtime_cli,
            output_cli,
            params=params,
            require_runtime=False,
        ) as (ctx, renderer):
            result = plugins_validate_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


__all__ = [
    "plugins_app",
]
