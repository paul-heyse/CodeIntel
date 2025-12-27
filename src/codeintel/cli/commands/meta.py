"""Metadata catalog commands."""

from __future__ import annotations

from dataclasses import dataclass

from cyclopts import App

from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.meta import meta_sync_handler
from codeintel.cli.options.shared_flags import SharedFlagsProtocol, shared_flags_field
from codeintel.cli.options.types import CommandPath

meta_app = App(
    name="meta",
    help="Metadata catalog maintenance commands.",
)

_META_SYNC_CONFIG = CommandConfig(require_runtime=True, require_gateway=True)

META_SYNC_PATH: CommandPath = ("meta", "sync")
_META_SYNC_FLAGS_FIELD = shared_flags_field(META_SYNC_PATH)


@cli_command("meta.sync", handler=meta_sync_handler, config=_META_SYNC_CONFIG)
@meta_app.command(name="sync")
@dataclass
class MetaSyncCommand:
    """Regenerate and persist canonical meta catalogs."""

    flags: SharedFlagsProtocol = _META_SYNC_FLAGS_FIELD
