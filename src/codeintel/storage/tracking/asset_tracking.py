"""Asset catalog tracking for build observability.

This module provides persistence and querying for the versioned asset catalog,
enabling "what exists?" visibility into the build state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa
from sqlglot import exp

from codeintel.core.columnar.conversion import reader_to_table, table_to_reader
from codeintel.core.columnar.expr_vocab import E
from codeintel.core.columnar.finalize_ops import finalize_spec_for_table, finalize_table
from codeintel.core.columnar.iter import iter_rows
from codeintel.core.columnar.kernels import SortKey
from codeintel.core.columnar.plan_ops import ScanPlanOptions, build_scan_plan
from codeintel.core.columnar.streaming import sample_reader
from codeintel.core.gateway import (
    AssetLineageEdgeRecordProtocol,
    AssetVersionEventRecordProtocol,
    AssetVersionRecordProtocol,
    RunAssetVersionRecordProtocol,
    RunEnvironmentRecordProtocol,
)
from codeintel.core.serialization.json import decode_json_dict
from codeintel.core.serialization.payload import encode_payload
from codeintel.core.sqlglot_tools import render_sql_duckdb, table_expr_from_ref
from codeintel.core.time import utc_now
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.storage.datasets.manifest_index import dataset_for_entry
from codeintel.storage.query_results import (
    coerce_optional_datetime,
    coerce_optional_int,
    coerce_optional_str,
    coerce_str,
    iter_tuples_from_arrow_reader,
)
from codeintel.storage.upsert import UpsertSpec

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from codeintel.core.columnar.expr_vocab import Expression
    from codeintel.storage.datasets.manifest_index import DatasetManifestEntry
    from codeintel.storage.gateway.protocol import StorageGateway


@dataclass(frozen=True)
class AssetVersionRecord:
    """Record of a content-addressed asset version."""

    asset_kind: str
    asset_key: str
    version_hash: str
    schema_hash: str | None = None
    row_count: int | None = None
    bytes: int | None = None
    created_at: datetime | None = None
    meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class AssetVersionEventRecord:
    """Run-scoped event for an asset version."""

    run_id: str
    repo: str
    commit: str
    asset_kind: str
    asset_key: str
    version_hash: str
    status: str
    target: str | None = None
    impl_kind: str | None = None
    location: str | None = None
    input_hash: str | None = None
    options_hash: str | None = None
    recorded_at: datetime | None = None
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
class AssetVersionHistoryRecord:
    """Versioned asset record joined with run event context."""

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
    recorded_at: datetime | None = None
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

    Provides persistence and querying for the versioned asset catalog tables.

    Parameters
    ----------
    gateway
        Storage gateway providing database access.
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
        self._backend = gateway.policy

    def _manifest_entry_for_table(self, table_key: str) -> DatasetManifestEntry | None:
        snapshot_id = self._gateway.config.commit
        if snapshot_id is None:
            return None
        return self._gateway.datasets.manifest_entry_for_table(table_key, snapshot_id=snapshot_id)

    @staticmethod
    def _combine_conditions(conditions: Sequence[exp.Expression]) -> exp.Expression | None:
        if not conditions:
            return None
        combined = conditions[0]
        for condition in conditions[1:]:
            combined = exp.and_(combined, condition)
        return combined

    @staticmethod
    def _aliased_table(table_ref: str, alias: str) -> exp.Table:
        table_expr = table_expr_from_ref(table_ref)
        aliased = table_expr.copy()
        aliased.set("alias", exp.TableAlias(this=exp.to_identifier(alias)))
        return aliased

    @staticmethod
    def _arrow_scan_table(
        *,
        entry: DatasetManifestEntry,
        columns: list[str],
        filter_expr: Expression | None,
        order_by: Sequence[SortKey] | None,
        limit: int | None,
    ) -> pa.Table:
        dataset = dataset_for_entry(entry)
        plan = build_scan_plan(
            dataset,
            options=ScanPlanOptions(
                columns=columns,
                filter_expr=filter_expr,
                implicit_ordering=True,
                require_sequenced_output=True,
                order_by=order_by,
            ),
        )
        reader = plan.to_reader(use_threads=True)
        if limit is not None:
            reader = sample_reader(reader, max_rows=limit)
        table = reader_to_table(reader)
        finalized = finalize_table(
            table,
            spec=finalize_spec_for_table(entry.manifest.table_key, mode="tolerant"),
        )
        return finalized.good

    def record_asset_versions_batch(self, records: Sequence[AssetVersionRecordProtocol]) -> int:
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
                r.schema_hash,
                r.row_count,
                r.bytes,
                r.created_at or now,
                encode_payload(r.meta or {}),
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
                "schema_hash",
                "row_count",
                "bytes",
                "created_at",
                "meta",
            ),
            upsert=UpsertSpec(
                conflict_columns=("asset_kind", "asset_key", "version_hash"),
                update_columns=(
                    "schema_hash",
                    "row_count",
                    "bytes",
                    "created_at",
                    "meta",
                ),
            ),
        )

    def record_asset_version_events_batch(
        self,
        records: Sequence[AssetVersionEventRecordProtocol],
    ) -> int:
        """Upsert run-scoped asset version events.

        Returns
        -------
        int
            Number of rows written to the asset_version_events table.
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
                r.impl_kind,
                r.status,
                r.location,
                r.input_hash,
                r.options_hash,
                r.recorded_at or now,
                encode_payload(r.meta or {}),
            )
            for r in records
        ]

        return self._backend.upsert(
            "build.asset_version_events",
            rows,
            columns=(
                "run_id",
                "repo",
                "commit",
                "asset_kind",
                "asset_key",
                "version_hash",
                "target",
                "impl_kind",
                "status",
                "location",
                "input_hash",
                "options_hash",
                "recorded_at",
                "meta",
            ),
            upsert=UpsertSpec(
                conflict_columns=("run_id", "asset_kind", "asset_key"),
                update_columns=(
                    "repo",
                    "commit",
                    "version_hash",
                    "target",
                    "impl_kind",
                    "status",
                    "location",
                    "input_hash",
                    "options_hash",
                    "recorded_at",
                    "meta",
                ),
            ),
        )

    def record_run_asset_versions_batch(
        self,
        records: Sequence[RunAssetVersionRecordProtocol],
    ) -> int:
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
                encode_payload(r.meta or {}),
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
            upsert=UpsertSpec(
                conflict_columns=("run_id", "asset_kind", "asset_key"),
                update_columns=(
                    "version_hash",
                    "target",
                    "resolution_kind",
                    "recorded_at",
                    "meta",
                ),
            ),
        )

    def record_lineage_edges_batch(
        self,
        edges: Sequence[AssetLineageEdgeRecordProtocol],
    ) -> int:
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
                encode_payload(e.meta or {}),
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
            upsert=UpsertSpec(
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
            ),
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
            upsert=UpsertSpec(
                conflict_columns=("alias", "asset_kind", "asset_key"),
                update_columns=("version_hash", "set_by_run_id", "set_at", "note"),
            ),
        )

    def resolve_alias(self, *, alias: str, asset_kind: str, asset_key: str) -> str | None:
        """Resolve an alias to a version_hash.

        Returns
        -------
        str | None
            Resolved version hash, or ``None`` when the alias is unknown.
        """
        entry = self._manifest_entry_for_table("build.asset_aliases")
        if entry is not None:
            table = self._arrow_scan_table(
                entry=entry,
                columns=["version_hash"],
                filter_expr=E.and_(
                    E.field("alias") == E.scalar(alias),
                    E.field("asset_kind") == E.scalar(asset_kind),
                    E.field("asset_key") == E.scalar(asset_key),
                ),
                order_by=None,
                limit=1,
            )
            if table.num_rows == 0:
                return None
            value = table.column("version_hash")[0].as_py()
            return str(value) if value is not None else None
        where_expr = self._combine_conditions(
            [
                exp.EQ(
                    this=exp.Column(this=exp.to_identifier("alias")),
                    expression=exp.Placeholder(),
                ),
                exp.EQ(
                    this=exp.Column(this=exp.to_identifier("asset_kind")),
                    expression=exp.Placeholder(),
                ),
                exp.EQ(
                    this=exp.Column(this=exp.to_identifier("asset_key")),
                    expression=exp.Placeholder(),
                ),
            ]
        )
        query = (
            exp.select(exp.Column(this=exp.to_identifier("version_hash")))
            .from_(table_expr_from_ref("build.asset_aliases"))
            .where(where_expr)
        )
        result = self._con.execute(
            render_sql_duckdb(query),
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
    ) -> list[AssetVersionHistoryRecord]:
        """List versions for an asset within a repo/commit scope.

        Returns
        -------
        list[AssetVersionHistoryRecord]
            Parsed asset version records ordered by recency.
        """
        events_table = self._aliased_table("build.asset_version_events", "e")
        versions_table = self._aliased_table("build.asset_versions", "v")
        join_condition = exp.and_(
            exp.EQ(
                this=exp.column("asset_kind", table="v"),
                expression=exp.column("asset_kind", table="e"),
            ),
            exp.EQ(
                this=exp.column("asset_key", table="v"),
                expression=exp.column("asset_key", table="e"),
            ),
            exp.EQ(
                this=exp.column("version_hash", table="v"),
                expression=exp.column("version_hash", table="e"),
            ),
        )
        where_expr = self._combine_conditions(
            [
                exp.EQ(
                    this=exp.column("repo", table="e"),
                    expression=exp.Placeholder(),
                ),
                exp.EQ(
                    this=exp.column("commit", table="e"),
                    expression=exp.Placeholder(),
                ),
                exp.EQ(
                    this=exp.column("asset_kind", table="e"),
                    expression=exp.Placeholder(),
                ),
                exp.EQ(
                    this=exp.column("asset_key", table="e"),
                    expression=exp.Placeholder(),
                ),
            ]
        )
        query = (
            exp.select(
                exp.column("asset_kind", table="v"),
                exp.column("asset_key", table="v"),
                exp.column("version_hash", table="v"),
                exp.column("repo", table="e"),
                exp.column("commit", table="e"),
                exp.column("status", table="e"),
                exp.column("run_id", table="e"),
                exp.column("target", table="e"),
                exp.column("impl_kind", table="e"),
                exp.column("location", table="e"),
                exp.column("input_hash", table="e"),
                exp.column("options_hash", table="e"),
                exp.column("schema_hash", table="v"),
                exp.column("row_count", table="v"),
                exp.column("bytes", table="v"),
                exp.column("created_at", table="v"),
                exp.column("recorded_at", table="e"),
                exp.column("meta", table="v"),
            )
            .from_(events_table)
            .join(versions_table, on=join_condition)
            .where(where_expr)
            .order_by(
                exp.Ordered(this=exp.column("recorded_at", table="e"), desc=True),
                exp.Ordered(this=exp.column("version_hash", table="v"), desc=True),
            )
            .limit(exp.Placeholder())
        )
        reader = self._con.execute(
            render_sql_duckdb(query),
            [repo, commit, asset_kind, asset_key, limit],
        ).fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)
        return [
            self._parse_asset_version_history_row(row)
            for row in iter_tuples_from_arrow_reader(reader)
        ]

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
        entry = self._manifest_entry_for_table("build.asset_version_events")
        if entry is not None:
            table = self._arrow_scan_table(
                entry=entry,
                columns=["version_hash", "recorded_at"],
                filter_expr=E.and_(
                    E.field("repo") == E.scalar(repo),
                    E.field("commit") == E.scalar(commit),
                    E.field("asset_kind") == E.scalar(asset_kind),
                    E.field("asset_key") == E.scalar(asset_key),
                ),
                order_by=[
                    ("recorded_at", "descending"),
                    ("version_hash", "descending"),
                ],
                limit=1,
            )
            if table.num_rows == 0:
                return None
            value = table.column("version_hash")[0].as_py()
            return str(value) if value is not None else None
        where_expr = self._combine_conditions(
            [
                exp.EQ(
                    this=exp.Column(this=exp.to_identifier("repo")),
                    expression=exp.Placeholder(),
                ),
                exp.EQ(
                    this=exp.Column(this=exp.to_identifier("commit")),
                    expression=exp.Placeholder(),
                ),
                exp.EQ(
                    this=exp.Column(this=exp.to_identifier("asset_kind")),
                    expression=exp.Placeholder(),
                ),
                exp.EQ(
                    this=exp.Column(this=exp.to_identifier("asset_key")),
                    expression=exp.Placeholder(),
                ),
            ]
        )
        query = (
            exp.select(exp.Column(this=exp.to_identifier("version_hash")))
            .from_(table_expr_from_ref("build.asset_version_events"))
            .where(where_expr)
            .order_by(
                exp.Ordered(this=exp.Column(this=exp.to_identifier("recorded_at")), desc=True),
                exp.Ordered(this=exp.Column(this=exp.to_identifier("version_hash")), desc=True),
            )
            .limit(exp.Literal.number(1))
        )
        row = self._con.execute(
            render_sql_duckdb(query),
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
        entry = self._manifest_entry_for_table("build.run_asset_versions")
        if entry is not None:
            columns = [
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
            ]
            table = self._arrow_scan_table(
                entry=entry,
                columns=columns,
                filter_expr=E.field("run_id") == E.scalar(run_id),
                order_by=[("asset_kind", "ascending"), ("asset_key", "ascending")],
                limit=None,
            )
            reader = table_to_reader(table, batch_size=DEFAULT_ARROW_BATCH_SIZE)
            return [
                RunAssetVersionRecord(
                    run_id=coerce_str(row[0], ctx="run_asset_versions.run_id"),
                    repo=coerce_str(row[1], ctx="run_asset_versions.repo"),
                    commit=coerce_str(row[2], ctx="run_asset_versions.commit"),
                    asset_kind=coerce_str(row[3], ctx="run_asset_versions.asset_kind"),
                    asset_key=coerce_str(row[4], ctx="run_asset_versions.asset_key"),
                    version_hash=coerce_str(row[5], ctx="run_asset_versions.version_hash"),
                    target=coerce_optional_str(row[6], ctx="run_asset_versions.target"),
                    resolution_kind=coerce_str(row[7], ctx="run_asset_versions.resolution_kind"),
                    recorded_at=coerce_optional_datetime(
                        row[8],
                        ctx="run_asset_versions.recorded_at",
                    ),
                    meta=decode_json_dict(row[9]) if row[9] else None,
                )
                for row in iter_tuples_from_arrow_reader(reader)
            ]
        query = (
            exp.select(
                exp.Column(this=exp.to_identifier("run_id")),
                exp.Column(this=exp.to_identifier("repo")),
                exp.Column(this=exp.to_identifier("commit")),
                exp.Column(this=exp.to_identifier("asset_kind")),
                exp.Column(this=exp.to_identifier("asset_key")),
                exp.Column(this=exp.to_identifier("version_hash")),
                exp.Column(this=exp.to_identifier("target")),
                exp.Column(this=exp.to_identifier("resolution_kind")),
                exp.Column(this=exp.to_identifier("recorded_at")),
                exp.Column(this=exp.to_identifier("meta")),
            )
            .from_(table_expr_from_ref("build.run_asset_versions"))
            .where(
                exp.EQ(
                    this=exp.Column(this=exp.to_identifier("run_id")),
                    expression=exp.Placeholder(),
                )
            )
            .order_by(
                exp.Ordered(this=exp.Column(this=exp.to_identifier("asset_kind"))),
                exp.Ordered(this=exp.Column(this=exp.to_identifier("asset_key"))),
            )
        )
        reader = self._con.execute(
            render_sql_duckdb(query),
            [run_id],
        ).fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)
        return [
            RunAssetVersionRecord(
                run_id=coerce_str(row[0], ctx="run_asset_versions.run_id"),
                repo=coerce_str(row[1], ctx="run_asset_versions.repo"),
                commit=coerce_str(row[2], ctx="run_asset_versions.commit"),
                asset_kind=coerce_str(row[3], ctx="run_asset_versions.asset_kind"),
                asset_key=coerce_str(row[4], ctx="run_asset_versions.asset_key"),
                version_hash=coerce_str(row[5], ctx="run_asset_versions.version_hash"),
                target=coerce_optional_str(row[6], ctx="run_asset_versions.target"),
                resolution_kind=coerce_str(row[7], ctx="run_asset_versions.resolution_kind"),
                recorded_at=coerce_optional_datetime(
                    row[8],
                    ctx="run_asset_versions.recorded_at",
                ),
                meta=decode_json_dict(row[9]) if row[9] else None,
            )
            for row in iter_tuples_from_arrow_reader(reader)
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
        entry = self._manifest_entry_for_table("build.asset_diffs")
        if entry is not None:
            columns = [
                "asset_kind",
                "asset_key",
                "from_version_hash",
                "to_version_hash",
                "diff_kind",
                "summary",
                "computed_at",
                "computed_by_run_id",
            ]
            table = self._arrow_scan_table(
                entry=entry,
                columns=columns,
                filter_expr=E.and_(
                    E.field("asset_kind") == E.scalar(asset_kind),
                    E.field("asset_key") == E.scalar(asset_key),
                    E.field("from_version_hash") == E.scalar(from_version_hash),
                    E.field("to_version_hash") == E.scalar(to_version_hash),
                    E.field("diff_kind") == E.scalar(diff_kind),
                ),
                order_by=None,
                limit=1,
            )
            reader = table_to_reader(table, batch_size=DEFAULT_ARROW_BATCH_SIZE)
            row = next(iter_tuples_from_arrow_reader(reader), None)
            if row is None:
                return None
            return AssetDiffRecord(
                asset_kind=coerce_str(row[0], ctx="asset_diffs.asset_kind"),
                asset_key=coerce_str(row[1], ctx="asset_diffs.asset_key"),
                from_version_hash=coerce_str(row[2], ctx="asset_diffs.from_version_hash"),
                to_version_hash=coerce_str(row[3], ctx="asset_diffs.to_version_hash"),
                diff_kind=coerce_str(row[4], ctx="asset_diffs.diff_kind"),
                summary=decode_json_dict(row[5]) if row[5] else None,
                computed_at=coerce_optional_datetime(row[6], ctx="asset_diffs.computed_at"),
                computed_by_run_id=coerce_optional_str(
                    row[7],
                    ctx="asset_diffs.computed_by_run_id",
                ),
            )
        where_expr = self._combine_conditions(
            [
                exp.EQ(
                    this=exp.Column(this=exp.to_identifier("asset_kind")),
                    expression=exp.Placeholder(),
                ),
                exp.EQ(
                    this=exp.Column(this=exp.to_identifier("asset_key")),
                    expression=exp.Placeholder(),
                ),
                exp.EQ(
                    this=exp.Column(this=exp.to_identifier("from_version_hash")),
                    expression=exp.Placeholder(),
                ),
                exp.EQ(
                    this=exp.Column(this=exp.to_identifier("to_version_hash")),
                    expression=exp.Placeholder(),
                ),
                exp.EQ(
                    this=exp.Column(this=exp.to_identifier("diff_kind")),
                    expression=exp.Placeholder(),
                ),
            ]
        )
        query = (
            exp.select(
                exp.Column(this=exp.to_identifier("asset_kind")),
                exp.Column(this=exp.to_identifier("asset_key")),
                exp.Column(this=exp.to_identifier("from_version_hash")),
                exp.Column(this=exp.to_identifier("to_version_hash")),
                exp.Column(this=exp.to_identifier("diff_kind")),
                exp.Column(this=exp.to_identifier("summary")),
                exp.Column(this=exp.to_identifier("computed_at")),
                exp.Column(this=exp.to_identifier("computed_by_run_id")),
            )
            .from_(table_expr_from_ref("build.asset_diffs"))
            .where(where_expr)
        )
        row = self._con.execute(
            render_sql_duckdb(query),
            [asset_kind, asset_key, from_version_hash, to_version_hash, diff_kind],
        ).fetchone()
        if row is None:
            return None
        return AssetDiffRecord(
            asset_kind=coerce_str(row[0], ctx="asset_diffs.asset_kind"),
            asset_key=coerce_str(row[1], ctx="asset_diffs.asset_key"),
            from_version_hash=coerce_str(row[2], ctx="asset_diffs.from_version_hash"),
            to_version_hash=coerce_str(row[3], ctx="asset_diffs.to_version_hash"),
            diff_kind=coerce_str(row[4], ctx="asset_diffs.diff_kind"),
            summary=decode_json_dict(row[5]) if row[5] else None,
            computed_at=coerce_optional_datetime(row[6], ctx="asset_diffs.computed_at"),
            computed_by_run_id=coerce_optional_str(
                row[7],
                ctx="asset_diffs.computed_by_run_id",
            ),
        )

    def save_cached_diff(self, record: AssetDiffRecord) -> None:
        """Upsert a cached diff summary."""
        computed_at = record.computed_at or utc_now()
        summary_json = record.summary or {}
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
            upsert=UpsertSpec(
                conflict_columns=(
                    "asset_kind",
                    "asset_key",
                    "from_version_hash",
                    "to_version_hash",
                    "diff_kind",
                ),
                update_columns=("summary", "computed_at", "computed_by_run_id"),
            ),
        )

    @staticmethod
    def _parse_asset_version_history_row(
        row: tuple[Any, ...],
    ) -> AssetVersionHistoryRecord:
        return AssetVersionHistoryRecord(
            asset_kind=coerce_str(row[0], ctx="asset_versions.asset_kind"),
            asset_key=coerce_str(row[1], ctx="asset_versions.asset_key"),
            version_hash=coerce_str(row[2], ctx="asset_versions.version_hash"),
            repo=coerce_str(row[3], ctx="asset_version_events.repo"),
            commit=coerce_str(row[4], ctx="asset_version_events.commit"),
            status=coerce_str(row[5], ctx="asset_version_events.status"),
            run_id=coerce_optional_str(row[6], ctx="asset_version_events.run_id"),
            target=coerce_optional_str(row[7], ctx="asset_version_events.target"),
            impl_kind=coerce_optional_str(row[8], ctx="asset_version_events.impl_kind"),
            location=coerce_optional_str(row[9], ctx="asset_version_events.location"),
            input_hash=coerce_optional_str(row[10], ctx="asset_version_events.input_hash"),
            options_hash=coerce_optional_str(row[11], ctx="asset_version_events.options_hash"),
            schema_hash=coerce_optional_str(row[12], ctx="asset_versions.schema_hash"),
            row_count=coerce_optional_int(row[13], ctx="asset_versions.row_count"),
            bytes=coerce_optional_int(row[14], ctx="asset_versions.bytes"),
            created_at=coerce_optional_datetime(row[15], ctx="asset_versions.created_at"),
            recorded_at=coerce_optional_datetime(row[16], ctx="asset_version_events.recorded_at"),
            meta=decode_json_dict(row[17]) if row[17] else None,
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
        entry = self._manifest_entry_for_table("build.asset_lineage")
        if entry is not None:
            columns = [
                "downstream_kind",
                "downstream_key",
                "downstream_version",
                "upstream_kind",
                "upstream_key",
                "upstream_version",
                "edge_kind",
                "created_at",
                "meta",
            ]
            exprs = [
                E.field("upstream_kind") == E.scalar(upstream_kind),
                E.field("upstream_key") == E.scalar(upstream_key),
            ]
            if upstream_version:
                exprs.append(E.field("upstream_version") == E.scalar(upstream_version))
            table = self._arrow_scan_table(
                entry=entry,
                columns=columns,
                filter_expr=E.and_(*exprs),
                order_by=None,
                limit=None,
            )
            reader = table_to_reader(table, batch_size=DEFAULT_ARROW_BATCH_SIZE)
            return [
                AssetLineageEdgeRecord(
                    downstream_kind=coerce_str(row[0], ctx="asset_lineage.downstream_kind"),
                    downstream_key=coerce_str(row[1], ctx="asset_lineage.downstream_key"),
                    downstream_version=coerce_str(row[2], ctx="asset_lineage.downstream_version"),
                    upstream_kind=coerce_str(row[3], ctx="asset_lineage.upstream_kind"),
                    upstream_key=coerce_str(row[4], ctx="asset_lineage.upstream_key"),
                    upstream_version=coerce_str(row[5], ctx="asset_lineage.upstream_version"),
                    edge_kind=coerce_str(row[6], ctx="asset_lineage.edge_kind"),
                    created_at=coerce_optional_datetime(row[7], ctx="asset_lineage.created_at"),
                    meta=decode_json_dict(row[8]) if row[8] else None,
                )
                for row in iter_tuples_from_arrow_reader(reader)
            ]
        conditions: list[exp.Expression] = [
            exp.EQ(
                this=exp.Column(this=exp.to_identifier("upstream_kind")),
                expression=exp.Placeholder(),
            ),
            exp.EQ(
                this=exp.Column(this=exp.to_identifier("upstream_key")),
                expression=exp.Placeholder(),
            ),
        ]
        params: list[object] = [upstream_kind, upstream_key]
        if upstream_version:
            conditions.append(
                exp.EQ(
                    this=exp.Column(this=exp.to_identifier("upstream_version")),
                    expression=exp.Placeholder(),
                )
            )
            params.append(upstream_version)
        query = (
            exp.select(
                exp.Column(this=exp.to_identifier("downstream_kind")),
                exp.Column(this=exp.to_identifier("downstream_key")),
                exp.Column(this=exp.to_identifier("downstream_version")),
                exp.Column(this=exp.to_identifier("upstream_kind")),
                exp.Column(this=exp.to_identifier("upstream_key")),
                exp.Column(this=exp.to_identifier("upstream_version")),
                exp.Column(this=exp.to_identifier("edge_kind")),
                exp.Column(this=exp.to_identifier("created_at")),
                exp.Column(this=exp.to_identifier("meta")),
            )
            .from_(table_expr_from_ref("build.asset_lineage"))
            .where(self._combine_conditions(conditions))
        )
        reader = self._con.execute(
            render_sql_duckdb(query),
            params,
        ).fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)
        return [
            AssetLineageEdgeRecord(
                downstream_kind=coerce_str(row[0], ctx="asset_lineage.downstream_kind"),
                downstream_key=coerce_str(row[1], ctx="asset_lineage.downstream_key"),
                downstream_version=coerce_str(row[2], ctx="asset_lineage.downstream_version"),
                upstream_kind=coerce_str(row[3], ctx="asset_lineage.upstream_kind"),
                upstream_key=coerce_str(row[4], ctx="asset_lineage.upstream_key"),
                upstream_version=coerce_str(row[5], ctx="asset_lineage.upstream_version"),
                edge_kind=coerce_str(row[6], ctx="asset_lineage.edge_kind"),
                created_at=coerce_optional_datetime(row[7], ctx="asset_lineage.created_at"),
                meta=decode_json_dict(row[8]) if row[8] else None,
            )
            for row in iter_tuples_from_arrow_reader(reader)
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
        entry = self._manifest_entry_for_table("build.asset_version_events")
        if entry is not None:
            table = self._arrow_scan_table(
                entry=entry,
                columns=["target", "recorded_at"],
                filter_expr=E.and_(
                    E.field("asset_kind") == E.scalar(asset_kind),
                    E.field("asset_key") == E.scalar(asset_key),
                ),
                order_by=[("recorded_at", "descending")],
                limit=1,
            )
            if table.num_rows == 0:
                return None
            value = table.column("target")[0].as_py()
            return str(value) if value is not None else None
        query = (
            exp.select(exp.Column(this=exp.to_identifier("target")))
            .from_(table_expr_from_ref("build.asset_version_events"))
            .where(
                self._combine_conditions(
                    [
                        exp.EQ(
                            this=exp.Column(this=exp.to_identifier("asset_kind")),
                            expression=exp.Placeholder(),
                        ),
                        exp.EQ(
                            this=exp.Column(this=exp.to_identifier("asset_key")),
                            expression=exp.Placeholder(),
                        ),
                    ]
                )
            )
            .order_by(
                exp.Ordered(this=exp.Column(this=exp.to_identifier("recorded_at")), desc=True)
            )
            .limit(exp.Literal.number(1))
        )
        row = self._con.execute(
            render_sql_duckdb(query),
            [asset_kind, asset_key],
        ).fetchone()

        if row is None or row[0] is None:
            return None
        return str(row[0])

    def record_run_environment(self, record: RunEnvironmentRecordProtocol) -> None:
        """Record the environment for a build run.

        Parameters
        ----------
        record
            Run environment record to save.
        """
        captured_at = record.captured_at or utc_now()
        tool_versions_json = record.tool_versions or {}

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
            upsert=UpsertSpec(
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
        entry = self._manifest_entry_for_table("build.run_environments")
        if entry is not None:
            columns = [
                "run_id",
                "python_version",
                "os_name",
                "os_version",
                "tool_versions",
                "config_hash",
                "git_dirty",
                "captured_at",
            ]
            table = self._arrow_scan_table(
                entry=entry,
                columns=columns,
                filter_expr=E.field("run_id") == E.scalar(run_id),
                order_by=None,
                limit=1,
            )
            if table.num_rows == 0:
                return None
            row_iter = iter_rows(table, columns=columns)
            row = next(row_iter, None)
            if row is None:
                return None
            tool_versions_payload = row.get("tool_versions")
            tool_versions_raw = (
                decode_json_dict(tool_versions_payload) if tool_versions_payload else None
            )
            tool_versions: dict[str, str] | None = None
            if tool_versions_raw:
                tool_versions = {k: str(v) for k, v in tool_versions_raw.items()}
            return RunEnvironmentRecord(
                run_id=str(row.get("run_id")),
                python_version=str(row.get("python_version")),
                os_name=str(row.get("os_name")),
                os_version=str(row.get("os_version")),
                tool_versions=tool_versions,
                config_hash=str(row.get("config_hash")) if row.get("config_hash") else None,
                git_dirty=bool(row.get("git_dirty")),
                captured_at=coerce_optional_datetime(
                    row.get("captured_at"),
                    ctx="build.run_environments.captured_at",
                ),
            )
        query = (
            exp.select(
                exp.Column(this=exp.to_identifier("run_id")),
                exp.Column(this=exp.to_identifier("python_version")),
                exp.Column(this=exp.to_identifier("os_name")),
                exp.Column(this=exp.to_identifier("os_version")),
                exp.Column(this=exp.to_identifier("tool_versions")),
                exp.Column(this=exp.to_identifier("config_hash")),
                exp.Column(this=exp.to_identifier("git_dirty")),
                exp.Column(this=exp.to_identifier("captured_at")),
            )
            .from_(table_expr_from_ref("build.run_environments"))
            .where(
                exp.EQ(
                    this=exp.Column(this=exp.to_identifier("run_id")),
                    expression=exp.Placeholder(),
                )
            )
        )
        row = self._con.execute(render_sql_duckdb(query), [run_id]).fetchone()

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
    "AssetTracking",
    "AssetVersionEventRecord",
    "AssetVersionHistoryRecord",
    "AssetVersionRecord",
    "RunAssetVersionRecord",
    "RunEnvironmentRecord",
]
