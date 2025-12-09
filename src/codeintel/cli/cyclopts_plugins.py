"""Plugin management commands.

Provide commands to discover, list, and inspect CLI plugins.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Annotated, Literal

from cyclopts import App, Parameter
from rich.console import Console
from rich.table import Table

from codeintel.cli.plugins import get_plugin_manager

plugins_app = App(name="plugins", help="Manage CLI plugins")


@plugins_app.command(name="list")
@dataclass
class PluginsListCommand:
    """List installed plugins.

    Display a table of all loaded plugins with their version,
    operation count, and description.
    """

    output_format: Annotated[
        Literal["text", "json"],
        Parameter(name="--format", help="Output format"),
    ] = "text"

    def __call__(self) -> None:
        """Execute the plugins list command."""
        manager = get_plugin_manager()
        plugins = manager.list_plugins()

        if self.output_format == "json":
            sys.stdout.write(json.dumps([p.to_dict() for p in plugins], indent=2))
            sys.stdout.write("\n")
            return

        if not plugins:
            sys.stdout.write("No plugins installed\n")
            return

        console = Console()
        table = Table(title="Installed Plugins")
        table.add_column("Name", style="cyan")
        table.add_column("Version")
        table.add_column("Operations", justify="right")
        table.add_column("Description")

        for plugin in plugins:
            table.add_row(
                plugin.name,
                plugin.version,
                str(plugin.operations),
                plugin.description,
            )

        console.print(table)


@plugins_app.command(name="discover")
@dataclass
class PluginsDiscoverCommand:
    """Discover available plugins.

    Search plugin directories for available plugins and show
    their loading status.
    """

    def __call__(self) -> None:
        """Execute the plugins discover command."""
        manager = get_plugin_manager()
        paths = manager.discover()

        if not paths:
            sys.stdout.write("No plugins found\n")
            sys.stdout.write("\nPlugin directories searched:\n")
            for plugin_dir in manager.plugin_dirs:
                sys.stdout.write(f"  • {plugin_dir}\n")
            return

        sys.stdout.write("Available plugins:\n")
        loaded_names = {p.name for p in manager.loaded_plugins.values()}

        for path in paths:
            # Check if plugin is loaded by looking for its stem in loaded plugins
            is_loaded = any(path.stem in name or name in path.stem for name in loaded_names)
            status = "✓ loaded" if is_loaded else "○ available"
            sys.stdout.write(f"  {status} {path.name}\n")


@plugins_app.command(name="info")
@dataclass
class PluginsInfoCommand:
    """Show details about a plugin.

    Display detailed information about a specific loaded plugin
    including its operations and file path.
    """

    name: Annotated[str, Parameter(help="Plugin name")]

    def __call__(self) -> None:
        """Execute the plugins info command.

        Raises
        ------
        SystemExit
            If the plugin is not found.
        """
        manager = get_plugin_manager()
        plugin = manager.get_plugin(self.name)

        if plugin is None:
            sys.stdout.write(f"Plugin not found: {self.name}\n")
            raise SystemExit(1)

        console = Console()
        console.print(f"[bold]Name:[/bold] {plugin.name}")
        console.print(f"[bold]Version:[/bold] {plugin.version}")
        console.print(f"[bold]Description:[/bold] {plugin.description}")
        console.print(f"[bold]Path:[/bold] {plugin.path}")
        console.print(f"[bold]Operations:[/bold] {plugin.operations}")
        console.print(f"[bold]Enabled:[/bold] {plugin.enabled}")


@plugins_app.command(name="paths")
@dataclass
class PluginsPathsCommand:
    """Show plugin search paths.

    Display all directories where plugins are searched,
    with indicators for which paths exist.
    """

    def __call__(self) -> None:
        """Execute the plugins paths command."""
        manager = get_plugin_manager()

        sys.stdout.write("Plugin directories:\n")
        for plugin_dir in manager.plugin_dirs:
            exists = "✓" if plugin_dir.exists() else "✗"
            sys.stdout.write(f"  {exists} {plugin_dir}\n")


__all__ = [
    "plugins_app",
]
