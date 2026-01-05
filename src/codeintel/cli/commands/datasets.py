"""Dataset management commands.

This module wires Cyclopts command classes to unified handlers via @cli_command.

Note: Dataset commands use the handler pattern. Only lint requires runtime/gateway
access; listing and diffing rely on build schema contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import App

from codeintel.cli.commands.dataset_ops import DatasetListCommand
from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.datasets import (
    datasets_diff_handler,
    datasets_lint_handler,
    datasets_migrate_parquet_handler,
    datasets_scaffold_handler,
    datasets_snapshot_handler,
)
from codeintel.cli.options.registry import (
    DATASETS_DIFF_AGAINST_REF,
    DATASETS_DIFF_BASELINE,
    DATASETS_DIFF_BASELINE_PATH,
    DATASETS_DIFF_OUTPUT,
    DATASETS_MIGRATE_DATASET_ROOT,
    DATASETS_MIGRATE_DROP_DUCKDB,
    DATASETS_MIGRATE_OVERWRITE,
    DATASETS_MIGRATE_SNAPSHOT_ID,
    DATASETS_MIGRATE_TABLE_KEYS,
    DATASETS_SAMPLING,
    DATASETS_SCAFFOLD_DRY_RUN,
    DATASETS_SCAFFOLD_NAME,
    DATASETS_SCAFFOLD_REGISTRY_CHECK,
    DATASETS_SCHEMA_DIR,
    DATASETS_SNAPSHOT_OUTPUT,
)
from codeintel.cli.options.shared_flags import SharedFlagsProtocol, shared_flags_field
from codeintel.cli.options.types import CommandPath, option_param

datasets_ext_app = App(
    name="datasets",
    help="Extended dataset management commands.",
)

datasets_ext_app.command(DatasetListCommand, name="list")


class SamplingStrictness(Enum):
    """Strictness policy when sampling rows."""

    STRICT = "strict"
    LENIENT = "lenient"


class OverwritePolicy(Enum):
    """Behavior when scaffold outputs already exist."""

    OVERWRITE = "overwrite"
    SKIP = "skip"
    ERROR = "error"


class BootstrapSnippet(Enum):
    """Whether to emit a bootstrap snippet during scaffold."""

    EMIT = "emit"
    SKIP = "skip"


_DATASETS_LINT_CONFIG = CommandConfig(require_runtime=True, require_gateway=True)
_DATASETS_READONLY_CONFIG = CommandConfig(require_runtime=False, require_gateway=False)
_SCAFFOLD_CONFIG = CommandConfig(require_runtime=False, require_gateway=False)
_DATASETS_MIGRATE_CONFIG = CommandConfig(require_runtime=True, require_gateway=True)

DATASETS_LINT_PATH: CommandPath = ("datasets", "lint")
DATASETS_SNAPSHOT_PATH: CommandPath = ("datasets", "snapshot")
DATASETS_DIFF_PATH: CommandPath = ("datasets", "diff")
DATASETS_SCAFFOLD_PATH: CommandPath = ("datasets", "scaffold")
DATASETS_MIGRATE_PATH: CommandPath = ("datasets", "migrate-parquet")

_DATASETS_LINT_FLAGS_FIELD = shared_flags_field(DATASETS_LINT_PATH)
_DATASETS_SNAPSHOT_FLAGS_FIELD = shared_flags_field(DATASETS_SNAPSHOT_PATH)
_DATASETS_DIFF_FLAGS_FIELD = shared_flags_field(DATASETS_DIFF_PATH)
_DATASETS_SCAFFOLD_FLAGS_FIELD = shared_flags_field(DATASETS_SCAFFOLD_PATH)
_DATASETS_MIGRATE_FLAGS_FIELD = shared_flags_field(DATASETS_MIGRATE_PATH)


@cli_command("datasets.lint", handler=datasets_lint_handler, config=_DATASETS_LINT_CONFIG)
@datasets_ext_app.command(name="lint")
@dataclass
class LintCommand:
    """Validate dataset contract health."""

    schema_dir: Annotated[
        Path,
        option_param(DATASETS_SCHEMA_DIR, command_path=DATASETS_LINT_PATH),
    ] = Path("config/schemas/export")
    sampling: Annotated[
        str,
        option_param(DATASETS_SAMPLING, command_path=DATASETS_LINT_PATH),
    ] = "disabled"
    flags: SharedFlagsProtocol = _DATASETS_LINT_FLAGS_FIELD


@cli_command(
    "datasets.snapshot", handler=datasets_snapshot_handler, config=_DATASETS_READONLY_CONFIG
)
@datasets_ext_app.command(name="snapshot")
@dataclass
class SnapshotCommand:
    """Write current dataset specs to a JSON snapshot file."""

    output: Annotated[
        Path,
        option_param(DATASETS_SNAPSHOT_OUTPUT, command_path=DATASETS_SNAPSHOT_PATH),
    ] = Path("build/dataset_specs.json")
    flags: SharedFlagsProtocol = _DATASETS_SNAPSHOT_FLAGS_FIELD


@cli_command("datasets.diff", handler=datasets_diff_handler, config=_DATASETS_READONLY_CONFIG)
@datasets_ext_app.command(name="diff")
@dataclass
class DiffCommand:
    """Diff current dataset specs against a baseline."""

    baseline: Annotated[
        Path | None,
        option_param(DATASETS_DIFF_BASELINE, command_path=DATASETS_DIFF_PATH),
    ] = None
    output: Annotated[
        Path | None,
        option_param(DATASETS_DIFF_OUTPUT, command_path=DATASETS_DIFF_PATH),
    ] = None
    against_ref: Annotated[
        str | None,
        option_param(DATASETS_DIFF_AGAINST_REF, command_path=DATASETS_DIFF_PATH),
    ] = None
    baseline_path: Annotated[
        Path,
        option_param(DATASETS_DIFF_BASELINE_PATH, command_path=DATASETS_DIFF_PATH),
    ] = Path("build/dataset_specs.json")
    flags: SharedFlagsProtocol = _DATASETS_DIFF_FLAGS_FIELD


@cli_command("datasets.scaffold", handler=datasets_scaffold_handler, config=_SCAFFOLD_CONFIG)
@datasets_ext_app.command(name="scaffold")
@dataclass
class ScaffoldDatasetCommand:
    """Scaffold a new dataset definition."""

    name: Annotated[
        str,
        option_param(DATASETS_SCAFFOLD_NAME, command_path=DATASETS_SCAFFOLD_PATH),
    ]
    registry_check: Annotated[
        Literal["enabled", "disabled"],
        option_param(DATASETS_SCAFFOLD_REGISTRY_CHECK, command_path=DATASETS_SCAFFOLD_PATH),
    ] = "enabled"
    dry_run: Annotated[
        bool,
        option_param(DATASETS_SCAFFOLD_DRY_RUN, command_path=DATASETS_SCAFFOLD_PATH),
    ] = False
    flags: SharedFlagsProtocol = _DATASETS_SCAFFOLD_FLAGS_FIELD


@cli_command(
    "datasets.migrate_parquet",
    handler=datasets_migrate_parquet_handler,
    config=_DATASETS_MIGRATE_CONFIG,
)
@datasets_ext_app.command(name="migrate-parquet")
@dataclass
class MigrateParquetCommand:
    """Materialize DuckDB dataset tables as Parquet snapshots."""

    dataset_root_dir: Annotated[
        Path | None,
        option_param(DATASETS_MIGRATE_DATASET_ROOT, command_path=DATASETS_MIGRATE_PATH),
    ] = None
    snapshot_id: Annotated[
        str | None,
        option_param(DATASETS_MIGRATE_SNAPSHOT_ID, command_path=DATASETS_MIGRATE_PATH),
    ] = None
    table_keys: Annotated[
        list[str] | None,
        option_param(DATASETS_MIGRATE_TABLE_KEYS, command_path=DATASETS_MIGRATE_PATH),
    ] = None
    overwrite: Annotated[
        bool,
        option_param(DATASETS_MIGRATE_OVERWRITE, command_path=DATASETS_MIGRATE_PATH),
    ] = False
    drop_duckdb_tables: Annotated[
        bool,
        option_param(DATASETS_MIGRATE_DROP_DUCKDB, command_path=DATASETS_MIGRATE_PATH),
    ] = False
    flags: SharedFlagsProtocol = _DATASETS_MIGRATE_FLAGS_FIELD


__all__ = ["datasets_ext_app"]
