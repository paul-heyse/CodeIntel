"""Iceberg CLI commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cyclopts import App

from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.iceberg import (
    iceberg_add_files_handler,
    iceberg_expire_snapshots_handler,
    iceberg_inspect_handler,
    iceberg_manage_snapshots_handler,
    iceberg_refs_handler,
    iceberg_time_travel_handler,
)
from codeintel.cli.options.iceberg import (
    ICEBERG_CONFIRM,
    ICEBERG_DATA_FORMAT,
    ICEBERG_DATA_PATH,
    ICEBERG_DRY_RUN,
    ICEBERG_INSPECT_ENTRIES,
    ICEBERG_INSPECT_MANIFESTS,
    ICEBERG_INSPECT_REFS,
    ICEBERG_INSPECT_SNAPSHOTS,
    ICEBERG_OUTPUT,
    ICEBERG_REF_NAME,
    ICEBERG_REF_REMOVE,
    ICEBERG_REF_TYPE,
    ICEBERG_REFRESH_CACHE,
    ICEBERG_RETENTION_DAYS,
    ICEBERG_SNAPSHOT_ID,
    ICEBERG_TABLE_KEY,
)
from codeintel.cli.options.shared_flags import SharedFlagsProtocol, shared_flags_field
from codeintel.cli.options.types import CommandPath, option_param

iceberg_app = App(
    name="iceberg",
    help="Iceberg catalog utilities.",
)

_ICEBERG_CONFIG = CommandConfig(require_runtime=True, require_gateway=False)

ICEBERG_INSPECT_PATH: CommandPath = ("iceberg", "inspect")
ICEBERG_REFS_PATH: CommandPath = ("iceberg", "refs")
ICEBERG_EXPIRE_PATH: CommandPath = ("iceberg", "expire-snapshots")
ICEBERG_TIME_TRAVEL_PATH: CommandPath = ("iceberg", "time-travel")
ICEBERG_MANAGE_PATH: CommandPath = ("iceberg", "manage-snapshots")
ICEBERG_ADD_FILES_PATH: CommandPath = ("iceberg", "add-files")

_ICEBERG_INSPECT_FLAGS_FIELD = shared_flags_field(ICEBERG_INSPECT_PATH)
_ICEBERG_REFS_FLAGS_FIELD = shared_flags_field(ICEBERG_REFS_PATH)
_ICEBERG_EXPIRE_FLAGS_FIELD = shared_flags_field(ICEBERG_EXPIRE_PATH)
_ICEBERG_TIME_TRAVEL_FLAGS_FIELD = shared_flags_field(ICEBERG_TIME_TRAVEL_PATH)
_ICEBERG_MANAGE_FLAGS_FIELD = shared_flags_field(ICEBERG_MANAGE_PATH)
_ICEBERG_ADD_FILES_FLAGS_FIELD = shared_flags_field(ICEBERG_ADD_FILES_PATH)


@cli_command("iceberg.inspect", handler=iceberg_inspect_handler, config=_ICEBERG_CONFIG)
@iceberg_app.command(name="inspect")
@dataclass
class IcebergInspectCommand:
    """Inspect Iceberg metadata for a table."""

    table_key: Annotated[
        str | None,
        option_param(ICEBERG_TABLE_KEY, command_path=ICEBERG_INSPECT_PATH),
    ] = None
    snapshot_id: Annotated[
        int | None,
        option_param(ICEBERG_SNAPSHOT_ID, command_path=ICEBERG_INSPECT_PATH),
    ] = None
    snapshots: Annotated[
        bool,
        option_param(ICEBERG_INSPECT_SNAPSHOTS, command_path=ICEBERG_INSPECT_PATH),
    ] = False
    manifests: Annotated[
        bool,
        option_param(ICEBERG_INSPECT_MANIFESTS, command_path=ICEBERG_INSPECT_PATH),
    ] = False
    entries: Annotated[
        bool,
        option_param(ICEBERG_INSPECT_ENTRIES, command_path=ICEBERG_INSPECT_PATH),
    ] = False
    refs: Annotated[
        bool,
        option_param(ICEBERG_INSPECT_REFS, command_path=ICEBERG_INSPECT_PATH),
    ] = False
    flags: SharedFlagsProtocol = _ICEBERG_INSPECT_FLAGS_FIELD


@cli_command("iceberg.refs", handler=iceberg_refs_handler, config=_ICEBERG_CONFIG)
@iceberg_app.command(name="refs")
@dataclass
class IcebergRefsCommand:
    """List Iceberg refs for a table."""

    table_key: Annotated[
        str | None,
        option_param(ICEBERG_TABLE_KEY, command_path=ICEBERG_REFS_PATH),
    ] = None
    flags: SharedFlagsProtocol = _ICEBERG_REFS_FLAGS_FIELD


@cli_command(
    "iceberg.expire_snapshots",
    handler=iceberg_expire_snapshots_handler,
    config=_ICEBERG_CONFIG,
)
@iceberg_app.command(name="expire-snapshots")
@dataclass
class IcebergExpireSnapshotsCommand:
    """Expire Iceberg snapshots based on retention."""

    table_key: Annotated[
        str | None,
        option_param(ICEBERG_TABLE_KEY, command_path=ICEBERG_EXPIRE_PATH),
    ] = None
    retention_days: Annotated[
        int | None,
        option_param(ICEBERG_RETENTION_DAYS, command_path=ICEBERG_EXPIRE_PATH),
    ] = None
    dry_run: Annotated[
        bool,
        option_param(ICEBERG_DRY_RUN, command_path=ICEBERG_EXPIRE_PATH),
    ] = False
    confirm: Annotated[
        bool,
        option_param(ICEBERG_CONFIRM, command_path=ICEBERG_EXPIRE_PATH),
    ] = False
    flags: SharedFlagsProtocol = _ICEBERG_EXPIRE_FLAGS_FIELD


@cli_command("iceberg.time_travel", handler=iceberg_time_travel_handler, config=_ICEBERG_CONFIG)
@iceberg_app.command(name="time-travel")
@dataclass
class IcebergTimeTravelCommand:
    """Export an Iceberg snapshot to IPC or Parquet."""

    table_key: Annotated[
        str | None,
        option_param(ICEBERG_TABLE_KEY, command_path=ICEBERG_TIME_TRAVEL_PATH),
    ] = None
    snapshot_id: Annotated[
        int | None,
        option_param(ICEBERG_SNAPSHOT_ID, command_path=ICEBERG_TIME_TRAVEL_PATH),
    ] = None
    output: Annotated[
        Path | None,
        option_param(ICEBERG_OUTPUT, command_path=ICEBERG_TIME_TRAVEL_PATH),
    ] = None
    data_format: Annotated[
        str | None,
        option_param(ICEBERG_DATA_FORMAT, command_path=ICEBERG_TIME_TRAVEL_PATH),
    ] = None
    flags: SharedFlagsProtocol = _ICEBERG_TIME_TRAVEL_FLAGS_FIELD


@cli_command(
    "iceberg.manage_snapshots",
    handler=iceberg_manage_snapshots_handler,
    config=_ICEBERG_CONFIG,
)
@iceberg_app.command(name="manage-snapshots")
@dataclass
class IcebergManageSnapshotsCommand:
    """Create or remove Iceberg snapshot refs."""

    table_key: Annotated[
        str | None,
        option_param(ICEBERG_TABLE_KEY, command_path=ICEBERG_MANAGE_PATH),
    ] = None
    snapshot_id: Annotated[
        int | None,
        option_param(ICEBERG_SNAPSHOT_ID, command_path=ICEBERG_MANAGE_PATH),
    ] = None
    ref_name: Annotated[
        str | None,
        option_param(ICEBERG_REF_NAME, command_path=ICEBERG_MANAGE_PATH),
    ] = None
    ref_type: Annotated[
        str | None,
        option_param(ICEBERG_REF_TYPE, command_path=ICEBERG_MANAGE_PATH),
    ] = None
    ref_remove: Annotated[
        bool,
        option_param(ICEBERG_REF_REMOVE, command_path=ICEBERG_MANAGE_PATH),
    ] = False
    confirm: Annotated[
        bool,
        option_param(ICEBERG_CONFIRM, command_path=ICEBERG_MANAGE_PATH),
    ] = False
    flags: SharedFlagsProtocol = _ICEBERG_MANAGE_FLAGS_FIELD


@cli_command(
    "iceberg.add_files",
    handler=iceberg_add_files_handler,
    config=_ICEBERG_CONFIG,
)
@iceberg_app.command(name="add-files")
@dataclass
class IcebergAddFilesCommand:
    """Add Parquet files to an Iceberg table."""

    table_key: Annotated[
        str | None,
        option_param(ICEBERG_TABLE_KEY, command_path=ICEBERG_ADD_FILES_PATH),
    ] = None
    data_path: Annotated[
        Path | None,
        option_param(ICEBERG_DATA_PATH, command_path=ICEBERG_ADD_FILES_PATH),
    ] = None
    refresh_cache: Annotated[
        bool,
        option_param(ICEBERG_REFRESH_CACHE, command_path=ICEBERG_ADD_FILES_PATH),
    ] = False
    flags: SharedFlagsProtocol = _ICEBERG_ADD_FILES_FLAGS_FIELD


__all__ = ["iceberg_app"]
