"""Plugin management commands.

Provide commands to discover, list, and inspect CLI plugins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.commands._common import SHARED_FLAGS_METADATA, SharedFlags
from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.plugins import (
    plugins_discover_handler,
    plugins_info_handler,
    plugins_list_handler,
    plugins_new_handler,
    plugins_paths_handler,
    plugins_test_handler,
    plugins_validate_handler,
)

plugins_app = App(name="plugins", help="Manage CLI plugins")

# Config for plugins commands - no runtime or gateway needed
_PLUGINS_CONFIG = CommandConfig(require_runtime=False, require_gateway=False)


@cli_command("plugins.list", handler=plugins_list_handler, config=_PLUGINS_CONFIG)
@plugins_app.command(name="list")
@dataclass
class PluginsListCommand:
    """List installed plugins.

    Display a table of all loaded plugins with their version,
    operation count, and description.
    """

    flags: SharedFlags = field(default_factory=SharedFlags, metadata=SHARED_FLAGS_METADATA)


@cli_command("plugins.discover", handler=plugins_discover_handler, config=_PLUGINS_CONFIG)
@plugins_app.command(name="discover")
@dataclass
class PluginsDiscoverCommand:
    """Discover available plugins.

    Search plugin directories for available plugins and show
    their loading status.
    """

    flags: SharedFlags = field(default_factory=SharedFlags, metadata=SHARED_FLAGS_METADATA)


@cli_command("plugins.info", handler=plugins_info_handler, config=_PLUGINS_CONFIG)
@plugins_app.command(name="info")
@dataclass
class PluginsInfoCommand:
    """Show details about a plugin.

    Display detailed information about a specific loaded plugin
    including its operations and file path.
    """

    name: Annotated[str, Parameter(help="Plugin name")]
    flags: SharedFlags = field(default_factory=SharedFlags, metadata=SHARED_FLAGS_METADATA)


@cli_command("plugins.paths", handler=plugins_paths_handler, config=_PLUGINS_CONFIG)
@plugins_app.command(name="paths")
@dataclass
class PluginsPathsCommand:
    """Show plugin search paths.

    Display all directories where plugins are searched,
    with indicators for which paths exist.
    """

    flags: SharedFlags = field(default_factory=SharedFlags, metadata=SHARED_FLAGS_METADATA)


@cli_command("plugins.new", handler=plugins_new_handler, config=_PLUGINS_CONFIG)
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
    flags: SharedFlags = field(default_factory=SharedFlags, metadata=SHARED_FLAGS_METADATA)


@cli_command("plugins.test", handler=plugins_test_handler, config=_PLUGINS_CONFIG)
@plugins_app.command(name="test")
@dataclass
class PluginsTestCommand:
    """Test a plugin.

    Run the test harness to validate plugin manifest, loading,
    and operation registration.
    """

    path: Annotated[Path, Parameter(help="Plugin directory")]
    flags: SharedFlags = field(default_factory=SharedFlags, metadata=SHARED_FLAGS_METADATA)


@cli_command("plugins.validate", handler=plugins_validate_handler, config=_PLUGINS_CONFIG)
@plugins_app.command(name="validate")
@dataclass
class PluginsValidateCommand:
    """Validate plugin manifest.

    Check that the plugin manifest is valid and compatible
    with the current CLI version.
    """

    path: Annotated[Path, Parameter(help="Plugin directory")]
    flags: SharedFlags = field(default_factory=SharedFlags, metadata=SHARED_FLAGS_METADATA)


__all__ = [
    "plugins_app",
]
