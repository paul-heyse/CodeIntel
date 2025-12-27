"""Metadata catalog commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from cyclopts import App

from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.meta import (
    meta_override_pin_handler,
    meta_registry_health_handler,
    meta_sync_handler,
)
from codeintel.cli.options.registry import (
    DATASET_TABLE_KEY,
    META_OVERRIDE_SCHEMA_DIGEST,
    META_OVERRIDE_VERSION_ID,
)
from codeintel.cli.options.shared_flags import SharedFlagsProtocol, shared_flags_field
from codeintel.cli.options.types import CommandPath, option_param

meta_app = App(
    name="meta",
    help="Metadata catalog maintenance commands.",
)

_META_SYNC_CONFIG = CommandConfig(require_runtime=True, require_gateway=True)
_META_OVERRIDE_CONFIG = CommandConfig(require_runtime=False, require_gateway=True)

META_SYNC_PATH: CommandPath = ("meta", "sync")
META_HEALTH_PATH: CommandPath = ("meta", "health")
META_OVERRIDE_PIN_PATH: CommandPath = ("meta", "override-pin")
_META_SYNC_FLAGS_FIELD = shared_flags_field(META_SYNC_PATH)
_META_HEALTH_FLAGS_FIELD = shared_flags_field(META_HEALTH_PATH)
_META_OVERRIDE_FLAGS_FIELD = shared_flags_field(META_OVERRIDE_PIN_PATH)


@cli_command("meta.sync", handler=meta_sync_handler, config=_META_SYNC_CONFIG)
@meta_app.command(name="sync")
@dataclass
class MetaSyncCommand:
    """Regenerate and persist canonical meta catalogs."""

    flags: SharedFlagsProtocol = _META_SYNC_FLAGS_FIELD


@cli_command("meta.health", handler=meta_registry_health_handler, config=_META_OVERRIDE_CONFIG)
@meta_app.command(name="health")
@dataclass
class MetaHealthCommand:
    """Report schema registry health for the attached meta catalog."""

    flags: SharedFlagsProtocol = _META_HEALTH_FLAGS_FIELD


@cli_command("meta.override-pin", handler=meta_override_pin_handler, config=_META_OVERRIDE_CONFIG)
@meta_app.command(name="override-pin")
@dataclass
class MetaOverridePinCommand:
    """Pin the override registry to a specific schema version."""

    table_key: Annotated[
        str,
        option_param(DATASET_TABLE_KEY, command_path=META_OVERRIDE_PIN_PATH),
    ]
    schema_digest: Annotated[
        str | None,
        option_param(META_OVERRIDE_SCHEMA_DIGEST, command_path=META_OVERRIDE_PIN_PATH),
    ] = None
    version_id: Annotated[
        str | None,
        option_param(META_OVERRIDE_VERSION_ID, command_path=META_OVERRIDE_PIN_PATH),
    ] = None
    flags: SharedFlagsProtocol = _META_OVERRIDE_FLAGS_FIELD
