"""Manifest store implementations for plugin execution records."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from codeintel.core.plugins.execution.manifest import ManifestStore
from codeintel.core.plugins.types.result import PluginExecutionRecord, PluginStatus

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

log = logging.getLogger(__name__)


class DuckDBManifestStore(ManifestStore):
    """ManifestStore backed by DuckDB."""

    def __init__(
        self,
        con: DuckDBPyConnection,
        *,
        table_name: str = "core.plugin_execution_manifest",
    ) -> None:
        self._con = con
        self._table_name = self._validate_table_name(table_name)

    @staticmethod
    def _validate_table_name(name: str) -> str:
        """Validate table name to avoid injection in SQL fragments.

        Returns
        -------
        str
            Validated table name.

        Raises
        ------
        ValueError
            If the table name contains unsafe characters.
        """
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\\.]*", name):
            message = f"Invalid manifest table name: {name}"
            raise ValueError(message)
        return name

    def ensure_schema(self) -> None:
        """Create manifest table and indexes if they do not exist."""
        self._con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table_name} (
                id INTEGER PRIMARY KEY,
                plugin_name VARCHAR NOT NULL,
                repo VARCHAR NOT NULL,
                commit VARCHAR NOT NULL,
                scope_id VARCHAR,
                variant VARCHAR,
                status VARCHAR NOT NULL,
                started_at TIMESTAMP NOT NULL,
                ended_at TIMESTAMP NOT NULL,
                duration_ms DOUBLE NOT NULL,
                options_hash VARCHAR,
                input_hash VARCHAR,
                error VARCHAR,
                meta_json VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._con.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_manifest_lookup
            ON {self._table_name} (plugin_name, repo, commit, scope_id, variant)
            """
        )

    def load_last_record(
        self,
        *,
        plugin_name: str,
        repo: str,
        commit: str,
        scope_id: str | None,
        variant: str | None,
    ) -> PluginExecutionRecord | None:
        """Load the most recent record matching the key.

        Returns
        -------
        PluginExecutionRecord | None
            Latest matching record when present.
        """
        query = """
            SELECT
                plugin_name,
                status,
                started_at,
                ended_at,
                duration_ms,
                options_hash,
                input_hash,
                error,
                meta_json
            FROM {table_name}
            WHERE plugin_name = ?
              AND repo = ?
              AND commit = ?
              AND scope_id IS ?
              AND variant IS ?
            ORDER BY ended_at DESC
            LIMIT 1
            """
        row = self._con.execute(
            query.format(table_name=self._table_name),
            [plugin_name, repo, commit, scope_id, variant],
        ).fetchone()

        if not row:
            return None

        (
            name,
            status,
            started_at,
            ended_at,
            duration_ms,
            options_hash,
            input_hash,
            error,
            meta_json,
        ) = row
        meta: dict[str, Any] = {}
        if meta_json:
            try:
                meta = json.loads(str(meta_json))
            except json.JSONDecodeError:
                log.warning("manifest_store: failed to decode meta_json for %s", name)

        meta.update(
            {
                "repo": repo,
                "commit": commit,
                "scope_id": scope_id,
                "variant": variant,
                "options_hash": options_hash,
                "input_hash": input_hash,
            }
        )

        return PluginExecutionRecord(
            plugin_name=str(name),
            status=PluginStatus(status),
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=float(duration_ms),
            error=str(error) if error else None,
            meta=meta,
        )

    def append_record(self, record: PluginExecutionRecord) -> None:
        """Persist a new PluginExecutionRecord."""
        meta_json = json.dumps(dict(record.meta), default=str)
        insert_sql = """
            INSERT INTO {table_name} (
                plugin_name,
                repo,
                commit,
                scope_id,
                variant,
                status,
                started_at,
                ended_at,
                duration_ms,
                options_hash,
                input_hash,
                error,
                meta_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
        self._con.execute(
            insert_sql.format(table_name=self._table_name),
            [
                record.plugin_name,
                record.meta.get("repo", ""),
                record.meta.get("commit", ""),
                record.meta.get("scope_id"),
                record.meta.get("variant"),
                record.status,
                record.started_at,
                record.ended_at,
                record.duration_ms,
                record.meta.get("options_hash"),
                record.meta.get("input_hash"),
                record.error,
                meta_json,
            ],
        )


class InMemoryManifestStore(ManifestStore):
    """In-memory ManifestStore for testing."""

    def __init__(self) -> None:
        self._records: list[PluginExecutionRecord] = []

    def load_last_record(
        self,
        *,
        plugin_name: str,
        repo: str,
        commit: str,
        scope_id: str | None,
        variant: str | None,
    ) -> PluginExecutionRecord | None:
        """Return the most recent record for the key.

        Returns
        -------
        PluginExecutionRecord | None
            Latest record matching identity fields.
        """

        def matches(record: PluginExecutionRecord) -> bool:
            meta = record.meta
            return (
                record.plugin_name == plugin_name
                and meta.get("repo") == repo
                and meta.get("commit") == commit
                and meta.get("scope_id") == scope_id
                and meta.get("variant") == variant
            )

        matching: list[PluginExecutionRecord] = [r for r in self._records if matches(r)]
        if not matching:
            return None
        return sorted(matching, key=lambda r: r.ended_at, reverse=True)[0]

    def append_record(self, record: PluginExecutionRecord) -> None:
        """Append a record to the in-memory collection."""
        self._records.append(record)

    def all_records(self) -> list[PluginExecutionRecord]:
        """Return all stored records.

        Returns
        -------
        list[PluginExecutionRecord]
            Stored records in insertion order.
        """
        return list(self._records)


__all__ = [
    "DuckDBManifestStore",
    "InMemoryManifestStore",
]
