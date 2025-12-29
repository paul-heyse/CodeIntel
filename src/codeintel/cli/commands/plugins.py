"""Plugin inspection commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from cyclopts import App

from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.plugins import plugins_info_handler, plugins_list_handler
from codeintel.cli.options.registry import PLUGINS_NAME
from codeintel.cli.options.shared_flags import SharedFlagsProtocol, shared_flags_field
from codeintel.cli.options.types import CommandPath, option_param

plugins_app = App(
    name="plugins",
    help="Plugin inspection commands.",
)

_PLUGINS_CONFIG = CommandConfig(require_runtime=True, require_gateway=False)

PLUGINS_LIST_PATH: CommandPath = ("plugins", "list")
PLUGINS_INFO_PATH: CommandPath = ("plugins", "info")

_PLUGINS_LIST_FLAGS_FIELD = shared_flags_field(PLUGINS_LIST_PATH)
_PLUGINS_INFO_FLAGS_FIELD = shared_flags_field(PLUGINS_INFO_PATH)


@cli_command("plugins.list", handler=plugins_list_handler, config=_PLUGINS_CONFIG)
@plugins_app.command(name="list")
@dataclass
class PluginsListCommand:
    """List discovered plugin packs."""

    flags: SharedFlagsProtocol = _PLUGINS_LIST_FLAGS_FIELD


@cli_command("plugins.info", handler=plugins_info_handler, config=_PLUGINS_CONFIG)
@plugins_app.command(name="info")
@dataclass
class PluginsInfoCommand:
    """Show detailed information for a plugin pack."""

    name: Annotated[
        str,
        option_param(PLUGINS_NAME, command_path=PLUGINS_INFO_PATH),
    ]
    flags: SharedFlagsProtocol = _PLUGINS_INFO_FLAGS_FIELD


__all__ = [
    "PluginsInfoCommand",
    "PluginsListCommand",
    "plugins_app",
]
