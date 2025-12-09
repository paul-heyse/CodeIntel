"""Plugin management commands.

Provide commands to discover, list, and inspect CLI plugins.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import App, Parameter
from rich.console import Console
from rich.table import Table

from codeintel.cli.plugins import (
    PluginManifest,
    PluginTestHarness,
    create_plugin_scaffold,
    get_plugin_manager,
)

# Plugin name validation pattern
_PLUGIN_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")

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
            exists = "\u2713" if plugin_dir.exists() else "\u2717"
            sys.stdout.write(f"  {exists} {plugin_dir}\n")


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

    def __call__(self) -> None:
        """Execute the plugins new command.

        Raises
        ------
        SystemExit
            If plugin name is invalid.
        """
        output_dir = self.output or Path.cwd()

        # Validate name
        if not _PLUGIN_NAME_PATTERN.match(self.name):
            sys.stderr.write(
                "Error: Plugin name must be lowercase alphanumeric with hyphens/underscores\n",
            )
            raise SystemExit(1)

        plugin_dir = create_plugin_scaffold(self.name, output_dir)
        sys.stdout.write(f"Created plugin scaffold at: {plugin_dir}\n")
        sys.stdout.write("\nNext steps:\n")
        sys.stdout.write(f"  1. cd {plugin_dir}\n")
        sys.stdout.write(f"  2. Edit {self.name}/main.py to add your operations\n")
        sys.stdout.write("  3. Run tests: pytest tests/\n")
        sys.stdout.write(f"  4. Install: cp -r {self.name} ~/.codeintel/plugins/\n")


@plugins_app.command(name="test")
@dataclass
class PluginsTestCommand:
    """Test a plugin.

    Run the test harness to validate plugin manifest, loading,
    and operation registration.
    """

    path: Annotated[Path, Parameter(help="Plugin directory")]

    def __call__(self) -> None:
        """Execute the plugins test command.

        Raises
        ------
        SystemExit
            If plugin.json not found or tests fail.
        """
        manifest_path = self.path / "plugin.json"
        if not manifest_path.exists():
            sys.stderr.write(f"Error: No plugin.json found in {self.path}\n")
            raise SystemExit(1)

        manifest = PluginManifest.load(manifest_path)
        harness = PluginTestHarness(manifest)
        results = harness.run_all_tests()

        sys.stdout.write(f"Testing plugin: {manifest.name} v{manifest.version}\n\n")

        all_passed = True
        for result in results:
            status = "\u2713" if result.success else "\u2717"
            sys.stdout.write(f"  {status} {result.message}\n")

            for error in result.errors:
                sys.stdout.write(f"      Error: {error}\n")
                all_passed = False

            for warning in result.warnings:
                sys.stdout.write(f"      Warning: {warning}\n")

        sys.stdout.write("\n")
        summary = harness.get_summary()
        sys.stdout.write(f"Tests: {summary['passed']}/{summary['tests_run']} passed\n")

        if summary["registered_operations"]:
            ops = ", ".join(summary["registered_operations"])
            sys.stdout.write(f"Operations: {ops}\n")

        if not all_passed:
            raise SystemExit(1)


@plugins_app.command(name="validate")
@dataclass
class PluginsValidateCommand:
    """Validate plugin manifest.

    Check that the plugin manifest is valid and compatible
    with the current CLI version.
    """

    path: Annotated[Path, Parameter(help="Plugin directory")]

    def __call__(self) -> None:
        """Execute the plugins validate command.

        Raises
        ------
        SystemExit
            If plugin.json not found or validation fails.
        """
        manifest_path = self.path / "plugin.json"
        if not manifest_path.exists():
            sys.stderr.write(f"Error: No plugin.json found in {self.path}\n")
            raise SystemExit(1)

        try:
            manifest = PluginManifest.load(manifest_path)
        except (KeyError, ValueError) as e:
            sys.stderr.write(f"Error loading manifest: {e}\n")
            raise SystemExit(1) from None

        errors = manifest.validate()

        if errors:
            sys.stderr.write("Manifest validation failed:\n")
            for error in errors:
                sys.stderr.write(f"  - {error}\n")
            raise SystemExit(1)

        sys.stdout.write("\u2713 Manifest is valid\n")
        sys.stdout.write(f"  Name: {manifest.name}\n")
        sys.stdout.write(f"  Version: {manifest.version}\n")
        sys.stdout.write(f"  API Version: {manifest.api_version}\n")

        if manifest.capabilities:
            caps = ", ".join(cap.value for cap in manifest.capabilities)
            sys.stdout.write(f"  Capabilities: {caps}\n")


__all__ = [
    "plugins_app",
]
