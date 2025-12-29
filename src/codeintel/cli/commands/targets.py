"""Target inspection commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from cyclopts import App

from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.targets import targets_list_handler
from codeintel.cli.options.registry import TARGETS_SHOW_ORIGIN
from codeintel.cli.options.shared_flags import SharedFlagsProtocol, shared_flags_field
from codeintel.cli.options.types import CommandPath, option_param

targets_app = App(
    name="targets",
    help="Runtime target inspection commands.",
)

_TARGETS_CONFIG = CommandConfig(require_runtime=True, require_gateway=True)

TARGETS_LIST_PATH: CommandPath = ("targets", "list")

_TARGETS_LIST_FLAGS_FIELD = shared_flags_field(TARGETS_LIST_PATH)


@cli_command("targets.list", handler=targets_list_handler, config=_TARGETS_CONFIG)
@targets_app.command(name="list")
@dataclass
class TargetsListCommand:
    """List runtime targets."""

    show_origin: Annotated[
        bool,
        option_param(TARGETS_SHOW_ORIGIN, command_path=TARGETS_LIST_PATH),
    ] = False
    flags: SharedFlagsProtocol = _TARGETS_LIST_FLAGS_FIELD


__all__ = [
    "TargetsListCommand",
    "targets_app",
]
