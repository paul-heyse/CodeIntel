"""Storage-owned export utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.core.config.settings import ExportAuditSettings
from codeintel.storage.protocols import ExportRelation
from codeintel.storage.protocols.duckdb_export import adapt_duckdb_relation

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.storage.duckdb_types import DuckDBConnection
    from codeintel.storage.gateway.protocol import MinimalGateway


@dataclass(frozen=True, slots=True)
class ExportAuditRecord:
    """Metadata about a completed export for audit logging."""

    table_name: str
    macro: str
    rows: int | None
    duration_s: float
    output_path: Path


@dataclass(frozen=True, slots=True)
class ExportAuditContext:
    """Context bundle for export audit logging."""

    con: DuckDBConnection
    settings: ExportAuditSettings
    ensure_table: Callable[[], None] | None = None


@dataclass(frozen=True, slots=True)
class ExportService:
    """Export helpers bound to a storage gateway."""

    gateway: MinimalGateway

    def build_export_relation(self, *, sql: str) -> ExportRelation:
        """Build a DuckDB relation for export.

        Parameters
        ----------
        sql
            DuckDB SQL to execute for the export relation.

        Returns
        -------
        ExportRelation
            Export relation adapter.
        """
        return build_export_relation(self.gateway.con, sql=sql)

    @staticmethod
    def audit_enabled(settings: ExportAuditSettings) -> bool:
        """Return True when audit logging is enabled.

        Parameters
        ----------
        settings
            Export audit settings.

        Returns
        -------
        bool
            True when audit logging is enabled.
        """
        return audit_enabled(settings)

    def write_export_audit(
        self,
        record: ExportAuditRecord,
        *,
        settings: ExportAuditSettings,
        sql: str | None = None,
        plan: str | None = None,
    ) -> None:
        """Write an audit entry for an export operation.

        Parameters
        ----------
        record
            Audit record describing the export.
        settings
            Export audit settings.
        sql
            Optional SQL statement for the export.
        plan
            Optional query plan text.
        """
        write_export_audit(
            record=record,
            context=ExportAuditContext(
                con=self.gateway.con,
                settings=settings,
                ensure_table=self.gateway.policy.ensure_export_audit_table,
            ),
            sql=sql,
            plan=plan,
        )


def audit_enabled(settings: ExportAuditSettings) -> bool:
    """Return True when audit logging is enabled.

    Parameters
    ----------
    settings
        Export audit settings.

    Returns
    -------
    bool
        True when audit logging is enabled.
    """
    return settings.log_path is not None or settings.table_enabled


def build_export_relation(
    con: DuckDBConnection,
    *,
    sql: str,
) -> ExportRelation:
    """Wrap a DuckDB relation as an ExportRelation.

    Parameters
    ----------
    con
        DuckDB connection for relation creation.
    sql
        DuckDB SQL to execute for the export relation.

    Returns
    -------
    ExportRelation
        Export relation adapter.
    """
    relation = con.sql(sql)
    return adapt_duckdb_relation(relation)


def write_export_audit(
    record: ExportAuditRecord,
    *,
    context: ExportAuditContext,
    sql: str | None = None,
    plan: str | None = None,
) -> None:
    """Write an audit entry for an export operation.

    Parameters
    ----------
    record
        Audit record describing the export.
    context
        Export audit context for connections and settings.
    sql
        Optional SQL statement for the export.
    plan
        Optional query plan text.
    """
    if not audit_enabled(context.settings):
        return

    json_record = {
        "table": record.table_name,
        "macro": record.macro,
        "rows": record.rows,
        "duration_s": record.duration_s,
        "output": str(record.output_path),
    }
    if context.settings.log_path is not None:
        with context.settings.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_record))
            handle.write("\n")

    if context.settings.table_enabled:
        if context.ensure_table is not None:
            context.ensure_table()
        created_at = datetime.now(tz=UTC)
        context.con.execute(
            """
            INSERT INTO metadata.export_audit
                (dataset, macro, rows, duration_s, output_path, sql, plan, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                record.table_name,
                record.macro,
                record.rows,
                record.duration_s,
                str(record.output_path),
                sql,
                plan,
                created_at,
            ],
        )


__all__ = [
    "ExportAuditContext",
    "ExportAuditRecord",
    "ExportService",
    "audit_enabled",
    "build_export_relation",
    "write_export_audit",
]
