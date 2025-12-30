"""Option registry for Iceberg CLI commands."""

from __future__ import annotations

from codeintel.cli.options.types import OptionSpec

ICEBERG_TABLE_KEY = OptionSpec(
    arg_name="table_key",
    names=("--table",),
    help="Iceberg table key (e.g., 'core.functions').",
)
ICEBERG_SNAPSHOT_ID = OptionSpec(
    arg_name="snapshot_id",
    names=("--snapshot-id",),
    help="Snapshot id to inspect or time travel.",
)
ICEBERG_RETENTION_DAYS = OptionSpec(
    arg_name="retention_days",
    names=("--retention-days",),
    help="Expire snapshots older than this many days.",
)
ICEBERG_DRY_RUN = OptionSpec(
    arg_name="dry_run",
    names=("--dry-run",),
    help="Compute actions without committing changes.",
)
ICEBERG_CONFIRM = OptionSpec(
    arg_name="confirm",
    names=("--confirm",),
    help="Confirm destructive snapshot expiration.",
)
ICEBERG_OUTPUT = OptionSpec(
    arg_name="output",
    names=("--output", "-o"),
    help="Output path for time travel exports.",
)
ICEBERG_DATA_FORMAT = OptionSpec(
    arg_name="data_format",
    names=("--format",),
    help="Output format: ipc or parquet.",
)
ICEBERG_DATA_PATH = OptionSpec(
    arg_name="data_path",
    names=("--data-path",),
    help="Parquet file or directory to add to the Iceberg table.",
)
ICEBERG_INSPECT_SNAPSHOTS = OptionSpec(
    arg_name="snapshots",
    names=("--snapshots",),
    help="Include snapshot metadata.",
)
ICEBERG_INSPECT_MANIFESTS = OptionSpec(
    arg_name="manifests",
    names=("--manifests",),
    help="Include manifest metadata.",
)
ICEBERG_INSPECT_ENTRIES = OptionSpec(
    arg_name="entries",
    names=("--entries",),
    help="Include manifest entries.",
)
ICEBERG_INSPECT_REFS = OptionSpec(
    arg_name="refs",
    names=("--refs",),
    help="Include ref metadata.",
)
ICEBERG_REF_NAME = OptionSpec(
    arg_name="ref_name",
    names=("--ref-name", "--ref"),
    help="Snapshot ref name to create or remove.",
)
ICEBERG_REF_TYPE = OptionSpec(
    arg_name="ref_type",
    names=("--ref-type",),
    help="Snapshot ref type: tag or branch.",
)
ICEBERG_REF_REMOVE = OptionSpec(
    arg_name="ref_remove",
    names=("--remove",),
    help="Remove the specified snapshot ref.",
)
ICEBERG_REFRESH_CACHE = OptionSpec(
    arg_name="refresh_cache",
    names=("--refresh-cache",),
    help="Refresh the DuckDB metadata cache after add_files.",
)


__all__ = [
    "ICEBERG_CONFIRM",
    "ICEBERG_DATA_FORMAT",
    "ICEBERG_DATA_PATH",
    "ICEBERG_DRY_RUN",
    "ICEBERG_INSPECT_ENTRIES",
    "ICEBERG_INSPECT_MANIFESTS",
    "ICEBERG_INSPECT_REFS",
    "ICEBERG_INSPECT_SNAPSHOTS",
    "ICEBERG_OUTPUT",
    "ICEBERG_REFRESH_CACHE",
    "ICEBERG_REF_NAME",
    "ICEBERG_REF_REMOVE",
    "ICEBERG_REF_TYPE",
    "ICEBERG_RETENTION_DAYS",
    "ICEBERG_SNAPSHOT_ID",
    "ICEBERG_TABLE_KEY",
]
