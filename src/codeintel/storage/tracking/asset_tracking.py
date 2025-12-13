"""Asset catalog tracking for build observability.

This module provides persistence and querying for the asset catalog,
enabling "what exists?" visibility into the build state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend
from codeintel.storage.helpers.json import decode_json_dict, encode_json_compact
from codeintel.storage.helpers.time import utc_now

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.storage.gateway.protocol import StorageGateway


@dataclass(frozen=True)
class AssetRecord:
    """Record of a materialized asset.

    Attributes
    ----------
    asset_key
        Unique asset identifier (e.g., "analytics.function_metrics").
    asset_type
        Asset type: "table", "view", or "artifact".
    repo
        Repository slug.
    commit
        Commit SHA.
    owner_target
        Target that produced this asset.
    schema_version
        Schema version if applicable.
    row_count
        Row count for tables.
    file_size_bytes
        File size for artifacts.
    materialized_at
        When asset was created.
    input_hash
        Input hash from manifest.
    metadata
        Additional metadata as JSON-serializable dict.
    """

    asset_key: str
    asset_type: str
    repo: str
    commit: str
    owner_target: str
    schema_version: str | None = None
    row_count: int | None = None
    file_size_bytes: int | None = None
    materialized_at: datetime | None = None
    input_hash: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class AssetVersionRecord:
    """Record of a content-addressed asset version."""

    asset_kind: str
    asset_key: str
    version_hash: str
    repo: str
    commit: str
    status: str
    run_id: str | None = None
    target: str | None = None
    impl_kind: str | None = None
    location: str | None = None
    input_hash: str | None = None
    options_hash: str | None = None
    schema_hash: str | None = None
    row_count: int | None = None
    bytes: int | None = None
    created_at: datetime | None = None
    meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class RunAssetVersionRecord:
    """Record linking a run_id to a resolved asset version."""

    run_id: str
    repo: str
    commit: str
    asset_kind: str
    asset_key: str
    version_hash: str
    resolution_kind: str
    recorded_at: datetime | None = None
    target: str | None = None
    meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class AssetLineageEdgeRecord:
    """Version-level lineage edge between two asset versions."""

    downstream_kind: str
    downstream_key: str
    downstream_version: str
    upstream_kind: str
    upstream_key: str
    upstream_version: str
    edge_kind: str
    created_at: datetime | None = None
    meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class AssetAliasRecord:
    """Alias pointer to an asset version."""

    alias: str
    asset_kind: str
    asset_key: str
    version_hash: str
    set_at: datetime | None = None
    set_by_run_id: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class AssetDiffRecord:
    """Cached diff between two versions of the same asset."""

    asset_kind: str
    asset_key: str
    from_version_hash: str
    to_version_hash: str
    diff_kind: str
    computed_at: datetime | None = None
    computed_by_run_id: str | None = None
    summary: dict[str, Any] | None = None


@dataclass(frozen=True)
class RunEnvironmentRecord:
    """Captured environment for a build run.

    Attributes
    ----------
    run_id
        Build run identifier.
    python_version
        Python version string.
    os_name
        Operating system name.
    os_version
        Operating system release.
    tool_versions
        Mapping of tool names to versions.
    config_hash
        Hash of build configuration.
    git_dirty
        Whether git working tree had uncommitted changes.
    captured_at
        When environment was captured.
    """

    run_id: str
    python_version: str
    os_name: str
    os_version: str
    tool_versions: dict[str, str] | None = None
    config_hash: str | None = None
    git_dirty: bool = False
    captured_at: datetime | None = None


class AssetTracking:
    """Accessor for build asset catalog.

    Provides CRUD operations for the build.assets table,
    enabling observability into what has been materialized.

    Parameters
    ----------
    gateway
        Storage gateway providing database access.

    Examples
    --------
    >>> tracking = AssetTracking(gateway)
    >>> tracking.record_asset(AssetRecord(...))
    >>> assets = tracking.list_assets("org/repo", "abc123")
    """

    def __init__(self, gateway: StorageGateway) -> None:
        """Initialize asset tracking accessor.

        Parameters
        ----------
        gateway
            Storage gateway providing database access.
        """
        self._gateway = gateway
        self._con = gateway.con
        self._backend = DuckDBPolicyBackend(gateway)

    def record_asset(self, record: AssetRecord) -> None:
        """Record or update an asset in the catalog.

        Uses upsert to insert or update the asset record.

        Parameters
        ----------
        record
            Asset record to save.
        """
        materialized_at = record.materialized_at or utc_now()
        metadata_json = encode_json_compact(record.metadata or {})

        self._backend.upsert(
            "build.assets",
            [
                (
                    record.asset_key,
                    record.asset_type,
                    record.repo,
                    record.commit,
                    record.owner_target,
                    record.schema_version,
                    record.row_count,
                    record.file_size_bytes,
                    materialized_at,
                    record.input_hash,
                    metadata_json,
                )
            ],
            columns=(
                "asset_key",
                "asset_type",
                "repo",
                "commit",
                "owner_target",
                "schema_version",
                "row_count",
                "file_size_bytes",
                "materialized_at",
                "input_hash",
                "metadata",
            ),
            conflict_columns=("asset_key", "repo", "commit"),
            update_columns=(
                "asset_type",
                "owner_target",
                "schema_version",
                "row_count",
                "file_size_bytes",
                "materialized_at",
                "input_hash",
                "metadata",
            ),
        )

    def record_assets_batch(
        self,
        records: Sequence[AssetRecord],
    ) -> int:
        """Record multiple assets in a single batch.

        Parameters
        ----------
        records
            Sequence of AssetRecord objects to save.

        Returns
        -------
        int
            Number of records persisted.
        """
        if not records:
            return 0

        now = utc_now()
        rows = [
            (
                r.asset_key,
                r.asset_type,
                r.repo,
                r.commit,
                r.owner_target,
                r.schema_version,
                r.row_count,
                r.file_size_bytes,
                r.materialized_at or now,
                r.input_hash,
                encode_json_compact(r.metadata or {}),
            )
            for r in records
        ]

        return self._backend.upsert(
            "build.assets",
            rows,
            columns=(
                "asset_key",
                "asset_type",
                "repo",
                "commit",
                "owner_target",
                "schema_version",
                "row_count",
                "file_size_bytes",
                "materialized_at",
                "input_hash",
                "metadata",
            ),
            conflict_columns=("asset_key", "repo", "commit"),
            update_columns=(
                "asset_type",
                "owner_target",
                "schema_version",
                "row_count",
                "file_size_bytes",
                "materialized_at",
                "input_hash",
                "metadata",
            ),
        )

    def record_asset_versions_batch(self, records: Sequence[AssetVersionRecord]) -> int:
        """Upsert multiple asset version records.

        Returns
        -------
        int
            Number of rows written to the asset_versions table.
        """
        if not records:
            return 0

        now = utc_now()
        rows = [
            (
                r.asset_kind,
                r.asset_key,
                r.version_hash,
                r.repo,
                r.commit,
                r.run_id,
                r.target,
                r.impl_kind,
                r.status,
                r.location,
                r.input_hash,
                r.options_hash,
                r.schema_hash,
                r.row_count,
                r.bytes,
                r.created_at or now,
                encode_json_compact(r.meta or {}),
            )
            for r in records
        ]

        return self._backend.upsert(
            "build.asset_versions",
            rows,
            columns=(
                "asset_kind",
                "asset_key",
                "version_hash",
                "repo",
                "commit",
                "run_id",
                "target",
                "impl_kind",
                "status",
                "location",
                "input_hash",
                "options_hash",
                "schema_hash",
                "row_count",
                "bytes",
                "created_at",
                "meta",
            ),
            conflict_columns=("asset_kind", "asset_key", "version_hash"),
            update_columns=(
                "repo",
                "commit",
                "run_id",
                "target",
                "impl_kind",
                "status",
                "location",
                "input_hash",
                "options_hash",
                "schema_hash",
                "row_count",
                "bytes",
                "created_at",
                "meta",
            ),
        )

    def record_run_asset_versions_batch(self, records: Sequence[RunAssetVersionRecord]) -> int:
        """Upsert mappings from run_id to resolved asset versions.

        Returns
        -------
        int
            Number of rows written to the run_asset_versions table.
        """
        if not records:
            return 0

        now = utc_now()
        rows = [
            (
                r.run_id,
                r.repo,
                r.commit,
                r.asset_kind,
                r.asset_key,
                r.version_hash,
                r.target,
                r.resolution_kind,
                r.recorded_at or now,
                encode_json_compact(r.meta or {}),
            )
            for r in records
        ]

        return self._backend.upsert(
            "build.run_asset_versions",
            rows,
            columns=(
                "run_id",
                "repo",
                "commit",
                "asset_kind",
                "asset_key",
                "version_hash",
                "target",
                "resolution_kind",
                "recorded_at",
                "meta",
            ),
            conflict_columns=("run_id", "asset_kind", "asset_key"),
            update_columns=("version_hash", "target", "resolution_kind", "recorded_at", "meta"),
        )

    def record_lineage_edges_batch(self, edges: Sequence[AssetLineageEdgeRecord]) -> int:
        """Upsert lineage edges between asset versions.

        Returns
        -------
        int
            Number of rows written to the asset_lineage table.
        """
        if not edges:
            return 0

        now = utc_now()
        rows = [
            (
                e.downstream_kind,
                e.downstream_key,
                e.downstream_version,
                e.upstream_kind,
                e.upstream_key,
                e.upstream_version,
                e.edge_kind,
                e.created_at or now,
                encode_json_compact(e.meta or {}),
            )
            for e in edges
        ]

        return self._backend.upsert(
            "build.asset_lineage",
            rows,
            columns=(
                "downstream_kind",
                "downstream_key",
                "downstream_version",
                "upstream_kind",
                "upstream_key",
                "upstream_version",
                "edge_kind",
                "created_at",
                "meta",
            ),
            conflict_columns=(
                "downstream_kind",
                "downstream_key",
                "downstream_version",
                "upstream_kind",
                "upstream_key",
                "upstream_version",
                "edge_kind",
            ),
            update_columns=("created_at", "meta"),
        )

    def set_alias(self, record: AssetAliasRecord) -> None:
        """Set (upsert) an alias for an asset version."""
        set_at = record.set_at or utc_now()
        self._backend.upsert(
            "build.asset_aliases",
            [
                (
                    record.alias,
                    record.asset_kind,
                    record.asset_key,
                    record.version_hash,
                    record.set_by_run_id,
                    set_at,
                    record.note,
                )
            ],
            columns=(
                "alias",
                "asset_kind",
                "asset_key",
                "version_hash",
                "set_by_run_id",
                "set_at",
                "note",
            ),
            conflict_columns=("alias", "asset_kind", "asset_key"),
            update_columns=("version_hash", "set_by_run_id", "set_at", "note"),
        )

    def resolve_alias(self, *, alias: str, asset_kind: str, asset_key: str) -> str | None:
        """Resolve an alias to a version_hash.

        Returns
        -------
        str | None
            Resolved version hash, or ``None`` when the alias is unknown.
        """
        result = self._con.execute(
            """
            SELECT version_hash
            FROM build.asset_aliases
            WHERE alias = ? AND asset_kind = ? AND asset_key = ?
            """,
            [alias, asset_kind, asset_key],
        ).fetchone()
        if result is None:
            return None
        return str(result[0])

    def get_asset_versions(
        self,
        *,
        repo: str,
        commit: str,
        asset_kind: str,
        asset_key: str,
        limit: int = 50,
    ) -> list[AssetVersionRecord]:
        """List versions for an asset within a repo/commit scope.

        Returns
        -------
        list[AssetVersionRecord]
            Parsed asset version records ordered by recency.
        """
        rows = self._con.execute(
            """
            SELECT asset_kind, asset_key, version_hash, repo, commit,
                   status, run_id, target, impl_kind, location, input_hash,
                   options_hash, schema_hash, row_count, bytes, created_at, meta
            FROM build.asset_versions
            WHERE repo = ? AND commit = ? AND asset_kind = ? AND asset_key = ?
            ORDER BY created_at DESC, version_hash DESC
            LIMIT ?
            """,
            [repo, commit, asset_kind, asset_key, limit],
        ).fetchall()
        return [self._parse_asset_version_row(row) for row in rows]

    def get_latest_version_hash(
        self,
        *,
        repo: str,
        commit: str,
        asset_kind: str,
        asset_key: str,
    ) -> str | None:
        """Return the latest version_hash for an asset, if present.

        Returns
        -------
        str | None
            Newest version hash, or ``None`` when no versions exist.
        """
        row = self._con.execute(
            """
            SELECT version_hash
            FROM build.asset_versions
            WHERE repo = ? AND commit = ? AND asset_kind = ? AND asset_key = ?
            ORDER BY created_at DESC, version_hash DESC
            LIMIT 1
            """,
            [repo, commit, asset_kind, asset_key],
        ).fetchone()
        if row is None:
            return None
        return str(row[0])

    def get_run_asset_versions(self, *, run_id: str) -> list[RunAssetVersionRecord]:
        """List run->asset version mappings for a given run_id.

        Returns
        -------
        list[RunAssetVersionRecord]
            Run asset version mappings sorted by asset kind/key.
        """
        rows = self._con.execute(
            """
            SELECT run_id, repo, commit, asset_kind, asset_key, version_hash,
                   target, resolution_kind, recorded_at, meta
            FROM build.run_asset_versions
            WHERE run_id = ?
            ORDER BY asset_kind, asset_key
            """,
            [run_id],
        ).fetchall()
        return [
            RunAssetVersionRecord(
                run_id=str(row[0]),
                repo=str(row[1]),
                commit=str(row[2]),
                asset_kind=str(row[3]),
                asset_key=str(row[4]),
                version_hash=str(row[5]),
                target=str(row[6]) if row[6] else None,
                resolution_kind=str(row[7]),
                recorded_at=row[8],
                meta=decode_json_dict(row[9]) if row[9] else None,
            )
            for row in rows
        ]

    def get_cached_diff(
        self,
        *,
        asset_kind: str,
        asset_key: str,
        from_version_hash: str,
        to_version_hash: str,
        diff_kind: str,
    ) -> AssetDiffRecord | None:
        """Return a cached diff summary if present.

        Returns
        -------
        AssetDiffRecord | None
            Cached diff record when found, otherwise ``None``.
        """
        row = self._con.execute(
            """
            SELECT asset_kind, asset_key, from_version_hash, to_version_hash,
                   diff_kind, summary, computed_at, computed_by_run_id
            FROM build.asset_diffs
            WHERE asset_kind = ? AND asset_key = ? AND from_version_hash = ?
              AND to_version_hash = ? AND diff_kind = ?
            """,
            [asset_kind, asset_key, from_version_hash, to_version_hash, diff_kind],
        ).fetchone()
        if row is None:
            return None
        return AssetDiffRecord(
            asset_kind=str(row[0]),
            asset_key=str(row[1]),
            from_version_hash=str(row[2]),
            to_version_hash=str(row[3]),
            diff_kind=str(row[4]),
            summary=decode_json_dict(row[5]) if row[5] else None,
            computed_at=row[6],
            computed_by_run_id=str(row[7]) if row[7] else None,
        )

    def save_cached_diff(self, record: AssetDiffRecord) -> None:
        """Upsert a cached diff summary."""
        computed_at = record.computed_at or utc_now()
        summary_json = encode_json_compact(record.summary or {})
        self._backend.upsert(
            "build.asset_diffs",
            [
                (
                    record.asset_kind,
                    record.asset_key,
                    record.from_version_hash,
                    record.to_version_hash,
                    record.diff_kind,
                    summary_json,
                    computed_at,
                    record.computed_by_run_id,
                )
            ],
            columns=(
                "asset_kind",
                "asset_key",
                "from_version_hash",
                "to_version_hash",
                "diff_kind",
                "summary",
                "computed_at",
                "computed_by_run_id",
            ),
            conflict_columns=(
                "asset_kind",
                "asset_key",
                "from_version_hash",
                "to_version_hash",
                "diff_kind",
            ),
            update_columns=("summary", "computed_at", "computed_by_run_id"),
        )

    def list_assets(
        self,
        repo: str,
        commit: str,
        *,
        asset_type: str | None = None,
        owner_target: str | None = None,
    ) -> list[AssetRecord]:
        """List assets for a repo/commit with optional filters.

        Parameters
        ----------
        repo
            Repository slug.
        commit
            Commit SHA.
        asset_type
            Optional filter by asset type (table, view, artifact).
        owner_target
            Optional filter by owner target name.

        Returns
        -------
        list[AssetRecord]
            List of asset records matching the filters.
        """
        query = """
            SELECT asset_key, asset_type, repo, commit, owner_target,
                   schema_version, row_count, file_size_bytes,
                   materialized_at, input_hash, metadata
            FROM build.assets
            WHERE repo = ? AND commit = ?
        """
        params: list[Any] = [repo, commit]

        if asset_type:
            query += " AND asset_type = ?"
            params.append(asset_type)

        if owner_target:
            query += " AND owner_target = ?"
            params.append(owner_target)

        query += " ORDER BY asset_key"

        results = self._con.execute(query, params).fetchall()
        return [self._parse_asset_row(row) for row in results]

    @staticmethod
    def _parse_asset_row(row: tuple[Any, ...]) -> AssetRecord:
        """Parse a DuckDB row into AssetRecord.

        Parameters
        ----------
        row
            DuckDB row tuple from build.assets table.

        Returns
        -------
        AssetRecord
            Parsed asset record.
        """
        return AssetRecord(
            asset_key=str(row[0]),
            asset_type=str(row[1]),
            repo=str(row[2]),
            commit=str(row[3]),
            owner_target=str(row[4]),
            schema_version=str(row[5]) if row[5] else None,
            row_count=int(row[6]) if row[6] is not None else None,
            file_size_bytes=int(row[7]) if row[7] is not None else None,
            materialized_at=row[8],
            input_hash=str(row[9]) if row[9] else None,
            metadata=decode_json_dict(row[10]) if row[10] else None,
        )

    @staticmethod
    def _parse_asset_version_row(row: tuple[Any, ...]) -> AssetVersionRecord:
        return AssetVersionRecord(
            asset_kind=str(row[0]),
            asset_key=str(row[1]),
            version_hash=str(row[2]),
            repo=str(row[3]),
            commit=str(row[4]),
            status=str(row[5]),
            run_id=str(row[6]) if row[6] else None,
            target=str(row[7]) if row[7] else None,
            impl_kind=str(row[8]) if row[8] else None,
            location=str(row[9]) if row[9] else None,
            input_hash=str(row[10]) if row[10] else None,
            options_hash=str(row[11]) if row[11] else None,
            schema_hash=str(row[12]) if row[12] else None,
            row_count=int(row[13]) if row[13] is not None else None,
            bytes=int(row[14]) if row[14] is not None else None,
            created_at=row[15],
            meta=decode_json_dict(row[16]) if row[16] else None,
        )

    def get_downstream_edges(
        self,
        *,
        upstream_kind: str,
        upstream_key: str,
        upstream_version: str | None = None,
    ) -> list[AssetLineageEdgeRecord]:
        """Query lineage edges where the given asset is upstream.

        Parameters
        ----------
        upstream_kind
            Kind of upstream asset (table, artifact).
        upstream_key
            Key of upstream asset.
        upstream_version
            Specific version, or None for all versions.

        Returns
        -------
        list[AssetLineageEdgeRecord]
            Lineage edges with the asset as upstream.
        """
        if upstream_version:
            query = """
                SELECT downstream_kind, downstream_key, downstream_version,
                       upstream_kind, upstream_key, upstream_version,
                       edge_kind, created_at, meta
                FROM build.asset_lineage
                WHERE upstream_kind = ? AND upstream_key = ? AND upstream_version = ?
            """
            params: list[object] = [upstream_kind, upstream_key, upstream_version]
        else:
            query = """
                SELECT downstream_kind, downstream_key, downstream_version,
                       upstream_kind, upstream_key, upstream_version,
                       edge_kind, created_at, meta
                FROM build.asset_lineage
                WHERE upstream_kind = ? AND upstream_key = ?
            """
            params = [upstream_kind, upstream_key]

        rows = self._con.execute(query, params).fetchall()
        return [
            AssetLineageEdgeRecord(
                downstream_kind=str(row[0]),
                downstream_key=str(row[1]),
                downstream_version=str(row[2]),
                upstream_kind=str(row[3]),
                upstream_key=str(row[4]),
                upstream_version=str(row[5]),
                edge_kind=str(row[6]),
                created_at=row[7],
                meta=decode_json_dict(row[8]) if row[8] else None,
            )
            for row in rows
        ]

    def get_asset_target(
        self,
        asset_kind: str,
        asset_key: str,
    ) -> str | None:
        """Look up the target that produces an asset.

        Parameters
        ----------
        asset_kind
            Kind of asset (table, artifact).
        asset_key
            Key of asset.

        Returns
        -------
        str | None
            Target name, or None if not found.
        """
        row = self._con.execute(
            """
            SELECT target
            FROM build.asset_versions
            WHERE asset_kind = ? AND asset_key = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            [asset_kind, asset_key],
        ).fetchone()

        if row is None or row[0] is None:
            return None
        return str(row[0])

    def record_run_environment(self, record: RunEnvironmentRecord) -> None:
        """Record the environment for a build run.

        Parameters
        ----------
        record
            Run environment record to save.
        """
        captured_at = record.captured_at or utc_now()
        tool_versions_json = encode_json_compact(record.tool_versions or {})

        self._backend.upsert(
            "build.run_environments",
            [
                (
                    record.run_id,
                    record.python_version,
                    record.os_name,
                    record.os_version,
                    tool_versions_json,
                    record.config_hash,
                    record.git_dirty,
                    captured_at,
                )
            ],
            columns=(
                "run_id",
                "python_version",
                "os_name",
                "os_version",
                "tool_versions",
                "config_hash",
                "git_dirty",
                "captured_at",
            ),
            conflict_columns=("run_id",),
            update_columns=(
                "python_version",
                "os_name",
                "os_version",
                "tool_versions",
                "config_hash",
                "git_dirty",
                "captured_at",
            ),
        )

    def get_run_environment(self, run_id: str) -> RunEnvironmentRecord | None:
        """Get the environment record for a build run.

        Parameters
        ----------
        run_id
            Build run identifier.

        Returns
        -------
        RunEnvironmentRecord | None
            Environment record, or None if not found.
        """
        row = self._con.execute(
            """
            SELECT run_id, python_version, os_name, os_version,
                   tool_versions, config_hash, git_dirty, captured_at
            FROM build.run_environments
            WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()

        if row is None:
            return None

        # Parse tool_versions, ensuring string values
        tool_versions_raw = decode_json_dict(row[4]) if row[4] else None
        tool_versions: dict[str, str] | None = None
        if tool_versions_raw:
            tool_versions = {k: str(v) for k, v in tool_versions_raw.items()}

        return RunEnvironmentRecord(
            run_id=str(row[0]),
            python_version=str(row[1]),
            os_name=str(row[2]),
            os_version=str(row[3]),
            tool_versions=tool_versions,
            config_hash=str(row[5]) if row[5] else None,
            git_dirty=bool(row[6]),
            captured_at=row[7],
        )


__all__ = [
    "AssetAliasRecord",
    "AssetDiffRecord",
    "AssetLineageEdgeRecord",
    "AssetRecord",
    "AssetTracking",
    "AssetVersionRecord",
    "RunAssetVersionRecord",
    "RunEnvironmentRecord",
]
