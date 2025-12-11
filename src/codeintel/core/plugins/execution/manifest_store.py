"""Manifest store implementations for plugin execution records."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, cast

from codeintel.core.plugins.execution.manifest import ManifestStore
from codeintel.core.plugins.types.result import PluginExecutionRecord, PluginStatus

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

log = logging.getLogger(__name__)

MANIFEST_TABLE = "core.plugin_execution_manifest"


class DuckDBManifestStore(ManifestStore):
    """ManifestStore backed by DuckDB."""

    def __init__(
        self,
        con: DuckDBPyConnection,
        *,
        table_name: str = MANIFEST_TABLE,
    ) -> None:
        if table_name != MANIFEST_TABLE:
            message = f"DuckDBManifestStore only supports table {MANIFEST_TABLE}"
            raise ValueError(message)

        self._con = con
        self._table_name = table_name
        self._insert_sql = (
            "INSERT INTO core.plugin_execution_manifest ("
            "plugin_name, repo, commit, scope_id, variant, status, started_at, "
            "ended_at, duration_ms, options_hash, input_hash, error, meta_json"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )

    def ensure_schema(self) -> None:
        """Create manifest table and indexes if they do not exist."""
        self._con.execute("CREATE SCHEMA IF NOT EXISTS core")
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS core.plugin_execution_manifest (
                id BIGINT,
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
            """
            CREATE INDEX IF NOT EXISTS idx_manifest_lookup
            ON core.plugin_execution_manifest (plugin_name, repo, commit, scope_id, variant)
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
        params: list[Any] = [plugin_name, repo, commit, scope_id, scope_id, variant, variant]
        row = self._con.execute(
            """
            SELECT
                plugin_name, status, started_at, ended_at, duration_ms,
                options_hash, input_hash, error, meta_json
            FROM core.plugin_execution_manifest
            WHERE plugin_name = ?
            AND repo = ?
            AND commit = ?
            AND ((scope_id IS NULL AND ? IS NULL) OR scope_id = ?)
            AND ((variant IS NULL AND ? IS NULL) OR variant = ?)
            ORDER BY created_at DESC
            LIMIT 1
            """,
            params,
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
            status=cast("PluginStatus", str(status)),
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=float(duration_ms),
            error=str(error) if error else None,
            meta=meta,
        )

    def append_record(self, record: PluginExecutionRecord) -> None:
        """Persist a new PluginExecutionRecord."""
        meta = dict(record.meta)
        repo = str(meta.pop("repo", ""))
        commit = str(meta.pop("commit", ""))
        scope_value = meta.pop("scope_id", None)
        variant_value = meta.pop("variant", None)
        options_hash = meta.pop("options_hash", None)
        input_hash = meta.pop("input_hash", None)
        meta_json = json.dumps(meta, default=str) if meta else None
        self._con.execute(
            self._insert_sql,
            [
                record.plugin_name,
                repo,
                commit,
                str(scope_value) if scope_value is not None else None,
                str(variant_value) if variant_value is not None else None,
                record.status,
                record.started_at,
                record.ended_at,
                record.duration_ms,
                options_hash,
                input_hash,
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
