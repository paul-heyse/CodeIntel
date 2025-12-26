"""Migrate legacy build tracking columns from plugin to impl_kind."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from codeintel.storage.backend import DuckDBSession
from codeintel.storage.gateway.config import StorageConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.storage.gateway.protocol import DuckDBConnection

LOG = logging.getLogger(__name__)

_TABLES: tuple[tuple[str, str], ...] = (
    ("build", "output_manifests"),
    ("build", "run_targets"),
)


def _table_exists(con: DuckDBConnection, *, schema: str, table: str) -> bool:
    row = con.execute(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = ? AND table_name = ?
        LIMIT 1
        """,
        [schema, table],
    ).fetchone()
    return row is not None


def _table_columns(con: DuckDBConnection, *, schema: str, table: str) -> set[str]:
    rows = con.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = ? AND table_name = ?
        """,
        [schema, table],
    ).fetchall()
    return {str(row[0]) for row in rows}


def _rename_column(
    con: DuckDBConnection,
    *,
    schema: str,
    table: str,
    old_name: str,
    new_name: str,
) -> None:
    qualified = f'"{schema}"."{table}"'
    con.execute(f'ALTER TABLE {qualified} RENAME COLUMN "{old_name}" TO "{new_name}"')


def _add_column(
    con: DuckDBConnection,
    *,
    schema: str,
    table: str,
    name: str,
    col_type: str,
) -> None:
    qualified = f'"{schema}"."{table}"'
    con.execute(f'ALTER TABLE {qualified} ADD COLUMN "{name}" {col_type}')


def _backfill_impl_kind(
    con: DuckDBConnection,
    *,
    schema: str,
    table: str,
) -> None:
    qualified = f'"{schema}"."{table}"'
    con.execute(f'UPDATE {qualified} SET impl_kind = ? WHERE impl_kind IS NULL', ["native"])


def _migrate_table(
    con: DuckDBConnection,
    *,
    schema: str,
    table: str,
    check_only: bool,
) -> tuple[int, list[str]]:
    issues: list[str] = []
    renamed = 0
    if not _table_exists(con, schema=schema, table=table):
        issues.append(f"Missing table {schema}.{table}")
        return renamed, issues
    columns = _table_columns(con, schema=schema, table=table)
    if "plugin" in columns and "impl_kind" in columns:
        issues.append(f"Table {schema}.{table} has both plugin and impl_kind columns")
        return renamed, issues
    if "plugin" in columns:
        if check_only:
            issues.append(f"Table {schema}.{table} still has legacy plugin column")
            return renamed, issues
        _rename_column(con, schema=schema, table=table, old_name="plugin", new_name="impl_kind")
        LOG.info("Renamed %s.%s plugin column to impl_kind", schema, table)
        renamed = 1
    elif "impl_kind" in columns:
        LOG.info("Table %s.%s already uses impl_kind", schema, table)
    else:
        if check_only:
            issues.append(f"Table {schema}.{table} missing impl_kind column")
            return renamed, issues
        _add_column(con, schema=schema, table=table, name="impl_kind", col_type="VARCHAR")
        _backfill_impl_kind(con, schema=schema, table=table)
        LOG.info("Added impl_kind column to %s.%s", schema, table)
        renamed = 1
    return renamed, issues


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate build tracking tables from plugin to impl_kind columns."
    )
    parser.add_argument(
        "--db-path",
        required=True,
        help="Path to the DuckDB database file.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Do not modify the database; report whether migration is needed.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the build tracking column migration.

    Parameters
    ----------
    argv
        Optional argv override for programmatic execution.

    Returns
    -------
    int
        Exit code (0 for success, 1 for errors).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args(argv)
    db_path = Path(args.db_path)
    if not db_path.exists():
        LOG.error("Database path does not exist: %s", db_path)
        return 1

    config = StorageConfig(
        db_path=db_path,
        read_only=False,
        apply_schema=False,
        ensure_views=False,
        validate_schema=False,
    )
    session = DuckDBSession(config)
    con = session.open()
    try:
        renamed = 0
        issues: list[str] = []
        for schema, table in _TABLES:
            renamed_delta, table_issues = _migrate_table(
                con,
                schema=schema,
                table=table,
                check_only=args.check_only,
            )
            renamed += renamed_delta
            issues.extend(table_issues)
    finally:
        con.close()

    if issues:
        for issue in issues:
            LOG.error(issue)
        return 1
    if args.check_only:
        LOG.info("No legacy plugin columns found.")
    else:
        LOG.info("Migration complete. Tables updated: %s", renamed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
