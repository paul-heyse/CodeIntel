"""Manifest store implementations for plugin execution records.

This module provides manifest storage using the Ibis-first architecture:
- Queries via Ibis expressions through StorageGateway
- Inserts via DuckDBPolicyBackend.bulk_insert()
- DDL via DuckDBPolicyBackend
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, cast

from ibis.expr.types import BooleanValue

from codeintel.core.plugins.execution.manifest import ManifestStore
from codeintel.core.plugins.types.result import PluginExecutionRecord, PluginStatus

if TYPE_CHECKING:
    from codeintel.storage.gateway.protocol import StorageGateway

log = logging.getLogger(__name__)

MANIFEST_TABLE = "core.plugin_execution_manifest"

# Column names for bulk insert
_MANIFEST_COLUMNS: tuple[str, ...] = (
    "plugin_name", "repo", "commit", "scope_id", "variant", "status",
    "started_at", "ended_at", "duration_ms", "options_hash", "input_hash",
    "error", "meta_json",
)


class DuckDBManifestStore(ManifestStore):
    """ManifestStore backed by DuckDB via StorageGateway.

    Uses Ibis expressions for queries and DuckDBPolicyBackend for inserts.
    """

    def __init__(
        self,
        gateway: StorageGateway,
        *,
        table_name: str = MANIFEST_TABLE,
    ) -> None:
        """Initialize the manifest store.

        Parameters
        ----------
        gateway
            Storage gateway providing Ibis access.
        table_name
            Qualified table name (must be core.plugin_execution_manifest).

        Raises
        ------
        ValueError
            If table_name is not the expected manifest table.
        """
        if table_name != MANIFEST_TABLE:
            message = f"DuckDBManifestStore only supports table {MANIFEST_TABLE}"
            raise ValueError(message)

        self._gateway = gateway
        self._table_name = table_name

    def ensure_schema(self) -> None:
        """Create manifest table and indexes if they do not exist.

        Uses DuckDBPolicyBackend for DDL operations.
        """
        from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend  # noqa: PLC0415

        backend = DuckDBPolicyBackend(self._gateway)
        backend.create_schema_if_not_exists("core")

        # Create table using raw SQL via policy backend (no TableSchema yet)
        # This is acceptable as an infrastructure table
        self._gateway.con.execute(
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
        self._gateway.con.execute(
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

        Uses Ibis expressions for the query.

        Returns
        -------
        PluginExecutionRecord | None
            Latest matching record when present.
        """
        t = self._gateway.ibis.table(self._table_name)

        # Build base filter - cast to BooleanValue for type safety
        base_filter = cast(
            "BooleanValue",
            (t.plugin_name == plugin_name) & (t.repo == repo) & (t.commit == commit),
        )

        # Handle nullable scope_id and variant
        scope_filter = cast(
            "BooleanValue",
            t.scope_id.isnull() if scope_id is None else t.scope_id == scope_id,
        )
        variant_filter = cast(
            "BooleanValue",
            t.variant.isnull() if variant is None else t.variant == variant,
        )

        # Combine conditions and query
        combined_filter = cast("BooleanValue", base_filter & scope_filter & variant_filter)
        expr = (
            t.filter(combined_filter)
            .order_by(t.created_at.desc())
            .limit(1)
            .select(
                t.plugin_name,
                t.status,
                t.started_at,
                t.ended_at,
                t.duration_ms,
                t.options_hash,
                t.input_hash,
                t.error,
                t.meta_json,
            )
        )

        df = expr.to_pandas()
        if df.empty:
            return None

        row = df.iloc[0]
        meta: dict[str, Any] = {}
        if row["meta_json"]:
            try:
                meta = json.loads(str(row["meta_json"]))
            except json.JSONDecodeError:
                log.warning(
                    "manifest_store: failed to decode meta_json for %s",
                    row["plugin_name"],
                )

        meta.update(
            {
                "repo": repo,
                "commit": commit,
                "scope_id": scope_id,
                "variant": variant,
                "options_hash": row["options_hash"],
                "input_hash": row["input_hash"],
            }
        )

        return PluginExecutionRecord(
            plugin_name=str(row["plugin_name"]),
            status=cast("PluginStatus", str(row["status"])),
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            duration_ms=float(row["duration_ms"]),
            error=str(row["error"]) if row["error"] else None,
            meta=meta,
        )

    def append_record(self, record: PluginExecutionRecord) -> None:
        """Persist a new PluginExecutionRecord using policy backend."""
        from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend  # noqa: PLC0415

        meta = dict(record.meta)
        repo = str(meta.pop("repo", ""))
        commit = str(meta.pop("commit", ""))
        scope_value = meta.pop("scope_id", None)
        variant_value = meta.pop("variant", None)
        options_hash = meta.pop("options_hash", None)
        input_hash = meta.pop("input_hash", None)
        meta_json = json.dumps(meta, default=str) if meta else None

        row = (
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
        )

        backend = DuckDBPolicyBackend(self._gateway)
        backend.bulk_insert(self._table_name, [row], columns=list(_MANIFEST_COLUMNS))


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
