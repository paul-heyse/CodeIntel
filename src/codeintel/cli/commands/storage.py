"""Storage commands for macro validation and profiling.

Note: Storage commands require runtime/gateway access via handler pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from cyclopts import App

from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.storage import (
    export_database_handler,
    import_database_handler,
    ingest_cache_logs_handler,
    profile_storage_handler,
    validate_macros_handler,
)
from codeintel.cli.options.registry import (
    STORAGE_CACHE_LOG_DIR,
    STORAGE_CACHE_LOG_PATHS,
    STORAGE_DB_PATH,
    STORAGE_INCLUDE_VIEWS,
    STORAGE_INPUT_DIR,
    STORAGE_OUTPUT_DIR,
    STORAGE_VALIDATION_MODE,
)
from codeintel.cli.options.shared_flags import SharedFlagsProtocol, shared_flags_field
from codeintel.cli.options.types import CommandPath, option_param
from codeintel.cli.rendering.types import OutputFormat

storage_app = App(
    name="storage",
    help="Storage utilities.",
)


_STORAGE_CONFIG = CommandConfig(require_runtime=True, require_gateway=True)
STORAGE_VALIDATE_PATH: CommandPath = ("storage", "validate-macros")
STORAGE_PROFILE_PATH: CommandPath = ("storage", "profile")
STORAGE_EXPORT_PATH: CommandPath = ("storage", "export-db")
STORAGE_IMPORT_PATH: CommandPath = ("storage", "import-db")
STORAGE_INGEST_CACHE_LOGS_PATH: CommandPath = ("storage", "ingest-cache-logs")

_STORAGE_VALIDATE_FLAGS_FIELD = shared_flags_field(STORAGE_VALIDATE_PATH)
_STORAGE_PROFILE_FLAGS_FIELD = shared_flags_field(STORAGE_PROFILE_PATH)
_STORAGE_EXPORT_FLAGS_FIELD = shared_flags_field(STORAGE_EXPORT_PATH)
_STORAGE_IMPORT_FLAGS_FIELD = shared_flags_field(STORAGE_IMPORT_PATH)
_STORAGE_INGEST_CACHE_LOGS_FLAGS_FIELD = shared_flags_field(
    STORAGE_INGEST_CACHE_LOGS_PATH,
    default_output_format=OutputFormat.JSONL,
)


@cli_command("storage.validate_macros", handler=validate_macros_handler, config=_STORAGE_CONFIG)
@storage_app.command(name="validate-macros")
@dataclass
class ValidateMacrosCommand:
    """Validate the schema catalog registry tables."""

    db_path: Annotated[
        Path | None,
        option_param(STORAGE_DB_PATH, command_path=STORAGE_VALIDATE_PATH),
    ] = None
    validation_mode: Annotated[
        str | None,
        option_param(STORAGE_VALIDATION_MODE, command_path=STORAGE_VALIDATE_PATH),
    ] = None
    flags: SharedFlagsProtocol = _STORAGE_VALIDATE_FLAGS_FIELD


@cli_command("storage.profile", handler=profile_storage_handler, config=_STORAGE_CONFIG)
@storage_app.command(name="profile")
@dataclass
class ProfileStorageCommand:
    """Profile storage paths and sizes."""

    db_path: Annotated[
        Path | None,
        option_param(STORAGE_DB_PATH, command_path=STORAGE_PROFILE_PATH),
    ] = None
    output_dir: Annotated[
        Path,
        option_param(STORAGE_OUTPUT_DIR, command_path=STORAGE_PROFILE_PATH),
    ] = field(default_factory=lambda: Path("build/storage_profile"))
    include_views: Annotated[
        bool,
        option_param(STORAGE_INCLUDE_VIEWS, command_path=STORAGE_PROFILE_PATH),
    ] = False
    flags: SharedFlagsProtocol = _STORAGE_PROFILE_FLAGS_FIELD


@cli_command("storage.export_db", handler=export_database_handler, config=_STORAGE_CONFIG)
@storage_app.command(name="export-db")
@dataclass
class ExportDatabaseCommand:
    """Export the DuckDB database directory using EXPORT DATABASE."""

    db_path: Annotated[
        Path | None,
        option_param(STORAGE_DB_PATH, command_path=STORAGE_EXPORT_PATH),
    ] = None
    output_dir: Annotated[
        Path,
        option_param(STORAGE_OUTPUT_DIR, command_path=STORAGE_EXPORT_PATH),
    ] = field(default_factory=lambda: Path("build/db_export"))
    flags: SharedFlagsProtocol = _STORAGE_EXPORT_FLAGS_FIELD


@cli_command("storage.import_db", handler=import_database_handler, config=_STORAGE_CONFIG)
@storage_app.command(name="import-db")
@dataclass
class ImportDatabaseCommand:
    """Import a DuckDB database directory using IMPORT DATABASE."""

    db_path: Annotated[
        Path | None,
        option_param(STORAGE_DB_PATH, command_path=STORAGE_IMPORT_PATH),
    ] = None
    input_dir: Annotated[
        Path,
        option_param(STORAGE_INPUT_DIR, command_path=STORAGE_IMPORT_PATH),
    ] = field(default_factory=lambda: Path("build/db_export"))
    flags: SharedFlagsProtocol = _STORAGE_IMPORT_FLAGS_FIELD


@cli_command("storage.ingest_cache_logs", handler=ingest_cache_logs_handler, config=_STORAGE_CONFIG)
@storage_app.command(name="ingest-cache-logs")
@dataclass
class IngestCacheLogsCommand:
    """Ingest Hamilton cache JSONL logs into DuckDB."""

    db_path: Annotated[
        Path | None,
        option_param(STORAGE_DB_PATH, command_path=STORAGE_INGEST_CACHE_LOGS_PATH),
    ] = None
    cache_dir: Annotated[
        Path | None,
        option_param(STORAGE_CACHE_LOG_DIR, command_path=STORAGE_INGEST_CACHE_LOGS_PATH),
    ] = None
    jsonl_paths: Annotated[
        list[Path] | None,
        option_param(STORAGE_CACHE_LOG_PATHS, command_path=STORAGE_INGEST_CACHE_LOGS_PATH),
    ] = None
    flags: SharedFlagsProtocol = _STORAGE_INGEST_CACHE_LOGS_FLAGS_FIELD


__all__ = ["storage_app"]
