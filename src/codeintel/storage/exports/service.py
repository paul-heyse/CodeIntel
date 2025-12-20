"""Storage-owned export utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.core.config.settings import ExportAuditSettings
from codeintel.storage.protocols import ExportRelation
from codeintel.storage.protocols.duckdb_export import adapt_duckdb_relation

if TYPE_CHECKING:
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

    def audit_enabled(self, settings: ExportAuditSettings) -> bool:
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
            con=self.gateway.con,
            settings=settings,
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
    con: DuckDBConnection,
    settings: ExportAuditSettings,
    sql: str | None = None,
    plan: str | None = None,
) -> None:
    """Write an audit entry for an export operation.

    Parameters
    ----------
    record
        Audit record describing the export.
    con
        DuckDB connection used for metadata logging.
    settings
        Export audit settings.
    sql
        Optional SQL statement for the export.
    plan
        Optional query plan text.
    """
    if not audit_enabled(settings):
        return

    json_record = {
        "table": record.table_name,
        "macro": record.macro,
        "rows": record.rows,
        "duration_s": record.duration_s,
        "output": str(record.output_path),
    }
    if settings.log_path is not None:
        with settings.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_record))
            handle.write("\n")

    if settings.table_enabled:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata.export_audit (
                dataset TEXT,
                macro TEXT,
                rows BIGINT,
                duration_s DOUBLE,
                output_path TEXT,
                sql TEXT,
                plan TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
        )
        con.execute(
            """
            INSERT INTO metadata.export_audit
                (dataset, macro, rows, duration_s, output_path, sql, plan)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                record.table_name,
                record.macro,
                record.rows,
                record.duration_s,
                str(record.output_path),
                sql,
                plan,
            ],
        )


__all__ = [
    "ExportAuditRecord",
    "ExportService",
    "audit_enabled",
    "build_export_relation",
    "write_export_audit",
]
