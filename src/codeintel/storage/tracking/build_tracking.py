"""Build manifest and run tracking persistence for DuckDB.

This module provides persistent tracking of build output manifests and
build runs, enabling cache invalidation and observability of the build
system.

All DuckDB access is encapsulated here, following the storage layer pattern.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from codeintel.core.build_manifest import BuildRunRecord, OutputManifest
from codeintel.core.time import utc_now
from codeintel.storage.helpers.json import (
    decode_json_dict,
    decode_json_list,
    deserialize_str_tuple,
    encode_json_compact,
    serialize_str_sequence,
)
from codeintel.storage.upsert import UpsertSpec

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from codeintel.core.build_manifest import BuildStatus
    from codeintel.core.hamilton.records import NodeExecutionRecord, TargetRunRecord
    from codeintel.storage.gateway.protocol import StorageGateway

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScipRunRecord:
    """Structured record for build.scip_runs telemetry rows."""

    run_id: str
    repo: str
    commit: str
    mode: str
    options_hash: str | None
    tool_version: str | None
    total_modules: int
    changed_modules: int
    deleted_modules: int
    changed_ratio: float | None
    batch_size: int | None
    batch_count: int
    decision: str | None
    ratio_gate_applied: bool | None
    ratio_gate_min_modules: int | None
    ratio_gate_min_changed: int | None
    hash_source: str | None
    hash_source_breakdown: str | None
    hash_reused: int
    hash_computed: int
    plan_ms: float | None
    hash_ms: float | None
    tool_ms: float | None
    parse_ms: float | None
    merge_ms: float | None
    write_ms: float | None
    total_ms: float | None
    status: str
    error_summary: str | None
    output_scip: str | None
    recorded_at: datetime


def _parse_manifest_row(row: tuple[Any, ...]) -> OutputManifest:
    """Parse a DuckDB row into an OutputManifest.

    Centralizes type coercion from DuckDB result tuples to typed dataclass.

    Parameters
    ----------
    row
        DuckDB row tuple from output_manifests table.
        Expected column order: target, repo, commit, impl_kind, computed_at,
        duration_ms, input_hash, output_hash, row_count, options_hash,
        change_delta

    Returns
    -------
    OutputManifest
        Typed manifest dataclass.
    """
    return OutputManifest(
        target=str(row[0]),
        repo=str(row[1]),
        commit=str(row[2]),
        impl_kind=str(row[3]),
        computed_at=cast("datetime", row[4]),
        duration_ms=float(row[5]),
        input_hash=str(row[6]),
        output_hash=str(row[7]) if row[7] is not None else None,
        row_count=int(row[8]) if row[8] is not None else None,
        options_hash=str(row[9]) if row[9] is not None else None,
        change_delta=decode_json_dict(row[10]) if row[10] is not None else None,
    )


def _parse_run_row(row: tuple[Any, ...]) -> BuildRunRecord:
    """Parse a DuckDB row into a BuildRunRecord.

    Centralizes type coercion from DuckDB result tuples to typed dataclass.

    Parameters
    ----------
    row
        DuckDB row tuple from build.runs table.
        Expected column order: run_id, repo, commit, requested_targets,
        computed_targets, skipped_targets, started_at, completed_at,
        status, error_summary, duration_ms

    Returns
    -------
    BuildRunRecord
        Typed run record dataclass.
    """
    return BuildRunRecord(
        run_id=str(row[0]),
        repo=str(row[1]),
        commit=str(row[2]),
        requested_targets=deserialize_str_tuple(cast("str | None", row[3])),
        computed_targets=deserialize_str_tuple(cast("str | None", row[4])),
        skipped_targets=deserialize_str_tuple(cast("str | None", row[5])),
        started_at=cast("datetime", row[6]),
        completed_at=cast("datetime | None", row[7]),
        status=cast("BuildStatus", row[8]),
        error_summary=str(row[9]) if row[9] is not None else None,
        duration_ms=float(row[10]) if row[10] is not None else None,
    )


class BuildTracking:
    """Accessor for build manifest and run tracking tables.

    This class provides CRUD operations for:
    - ``build.output_manifests``: Records of computed targets
    - ``build.runs``: Records of build system runs

    All operations are performed directly on the DuckDB connection
    without caching, following the storage accessor pattern.

    Parameters
    ----------
    con
        DuckDB connection to use for queries.

    Examples
    --------
    >>> tracking = BuildTracking(gateway)
    >>> tracking.save_manifest(manifest)
    >>> loaded = tracking.load_manifest("risk_factors", "org/repo", "abc123")
    """

    def __init__(self, gateway: StorageGateway) -> None:
        """Initialize build tracking accessor.

        Parameters
        ----------
        gateway
            Storage gateway providing database access.
        """
        self._gateway = gateway
        self._con = gateway.con
        self._backend = gateway.policy

    def save_manifest(self, manifest: OutputManifest) -> None:
        """Save or update an output manifest.

        Uses upsert to insert or update the manifest record.

        Parameters
        ----------
        manifest
            The manifest to save.
        """
        change_delta = (
            encode_json_compact(manifest.change_delta)
            if manifest.change_delta is not None
            else None
        )
        self._backend.upsert(
            "build.output_manifests",
            [
                (
                    manifest.target,
                    manifest.repo,
                    manifest.commit,
                    manifest.impl_kind,
                    manifest.computed_at,
                    manifest.duration_ms,
                    manifest.input_hash,
                    manifest.output_hash,
                    manifest.row_count,
                    manifest.options_hash,
                    change_delta,
                )
            ],
            columns=(
                "target",
                "repo",
                "commit",
                "impl_kind",
                "computed_at",
                "duration_ms",
                "input_hash",
                "output_hash",
                "row_count",
                "options_hash",
                "change_delta",
            ),
            upsert=UpsertSpec(
                conflict_columns=("target", "repo", "commit"),
                update_columns=(
                    "impl_kind",
                    "computed_at",
                    "duration_ms",
                    "input_hash",
                    "output_hash",
                    "row_count",
                    "options_hash",
                    "change_delta",
                ),
            ),
        )

    def load_manifest(self, target: str, repo: str, commit: str) -> OutputManifest | None:
        """Load an output manifest by primary key.

        Parameters
        ----------
        target
            Target name.
        repo
            Repository slug.
        commit
            Commit SHA.

        Returns
        -------
        OutputManifest | None
            The manifest if found, None otherwise.
        """
        result = self._con.execute(
            """
            SELECT target, repo, commit, impl_kind, computed_at, duration_ms,
                   input_hash, output_hash, row_count, options_hash
                   , change_delta
            FROM build.output_manifests
            WHERE target = ? AND repo = ? AND commit = ?
            """,
            [target, repo, commit],
        ).fetchone()

        if result is None:
            return None

        return _parse_manifest_row(result)

    def list_manifests(self, repo: str, commit: str) -> tuple[OutputManifest, ...]:
        """List all manifests for a repo/commit.

        Parameters
        ----------
        repo
            Repository slug.
        commit
            Commit SHA.

        Returns
        -------
        tuple[OutputManifest, ...]
            All manifests for the given repo/commit.
        """
        results = self._con.execute(
            """
            SELECT target, repo, commit, impl_kind, computed_at, duration_ms,
                   input_hash, output_hash, row_count, options_hash
                   , change_delta
            FROM build.output_manifests
            WHERE repo = ? AND commit = ?
            ORDER BY target
            """,
            [repo, commit],
        ).fetchall()

        return tuple(_parse_manifest_row(row) for row in results)

    def delete_manifests(self, repo: str, commit: str) -> None:
        """Delete all manifests for a repo/commit.

        Parameters
        ----------
        repo
            Repository slug.
        commit
            Commit SHA.
        """
        self._con.execute(
            """
            DELETE FROM build.output_manifests
            WHERE repo = ? AND commit = ?
            """,
            [repo, commit],
        )

    def start_run(self, record: BuildRunRecord) -> None:
        """Record the start of a build run.

        Parameters
        ----------
        record
            The run record to save.
        """
        inserted = self._backend.upsert(
            "build.runs",
            [
                (
                    record.run_id,
                    record.repo,
                    record.commit,
                    serialize_str_sequence(record.requested_targets),
                    serialize_str_sequence(record.computed_targets),
                    serialize_str_sequence(record.skipped_targets),
                    record.started_at,
                    record.completed_at,
                    record.status,
                    record.error_summary,
                    record.duration_ms,
                )
            ],
            columns=(
                "run_id",
                "repo",
                "commit",
                "requested_targets",
                "computed_targets",
                "skipped_targets",
                "started_at",
                "completed_at",
                "status",
                "error_summary",
                "duration_ms",
            ),
            upsert=UpsertSpec(
                conflict_columns=("run_id",),
                update_columns=(),
            ),
        )
        if inserted == 0:
            log.warning("build.run start ignored due to duplicate run_id: %s", record.run_id)

    def complete_run(
        self,
        run_id: str,
        status: BuildStatus,
        computed_targets: tuple[str, ...],
        skipped_targets: tuple[str, ...],
        error_summary: str | None = None,
    ) -> None:
        """Update a run record upon completion.

        Parameters
        ----------
        run_id
            Run identifier.
        status
            Final status (succeeded or failed).
        computed_targets
            Targets that were computed.
        skipped_targets
            Targets that were skipped.
        error_summary
            Error summary if failed.
        """
        completed_at = utc_now()

        result = self._con.execute(
            "SELECT started_at FROM build.runs WHERE run_id = ?",
            [run_id],
        ).fetchone()

        duration_ms: float | None = None
        if result is not None and result[0] is not None:
            started_at: datetime = cast("datetime", result[0])
            duration_ms = (completed_at - started_at).total_seconds() * 1000

        self._con.execute(
            """
            UPDATE build.runs
            SET completed_at = ?,
                status = ?,
                computed_targets = ?,
                skipped_targets = ?,
                error_summary = ?,
                duration_ms = ?
            WHERE run_id = ?
            """,
            [
                completed_at,
                status,
                serialize_str_sequence(computed_targets),
                serialize_str_sequence(skipped_targets),
                error_summary,
                duration_ms,
                run_id,
            ],
        )

    def fetch_run(self, run_id: str) -> BuildRunRecord | None:
        """Fetch a run record by ID.

        Parameters
        ----------
        run_id
            Run identifier.

        Returns
        -------
        BuildRunRecord | None
            The run record if found, None otherwise.
        """
        result = self._con.execute(
            """
            SELECT run_id, repo, commit, requested_targets, computed_targets,
                   skipped_targets, started_at, completed_at, status,
                   error_summary, duration_ms
            FROM build.runs
            WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()

        if result is None:
            return None

        return _parse_run_row(result)

    def list_runs(self, repo: str, limit: int = 100) -> tuple[BuildRunRecord, ...]:
        """List recent runs for a repository.

        Parameters
        ----------
        repo
            Repository slug.
        limit
            Maximum number of runs to return.

        Returns
        -------
        tuple[BuildRunRecord, ...]
            Recent runs, newest first.
        """
        results = self._con.execute(
            """
            SELECT run_id, repo, commit, requested_targets, computed_targets,
                   skipped_targets, started_at, completed_at, status,
                   error_summary, duration_ms
            FROM build.runs
            WHERE repo = ?
            ORDER BY started_at DESC
            LIMIT ?
            """,
            [repo, limit],
        ).fetchall()

        return tuple(_parse_run_row(row) for row in results)

    def save_run_targets(
        self,
        run_id: str,
        repo: str,
        commit: str,
        records: Sequence[TargetRunRecord],
    ) -> int:
        """Save per-target execution records for a build run.

        Parameters
        ----------
        run_id
            Parent run identifier.
        repo
            Repository slug.
        commit
            Commit SHA.
        records
            Sequence of TargetRunRecord objects from execution.

        Returns
        -------
        int
            Number of records inserted.
        """
        if not records:
            return 0

        recorded_at = utc_now()
        rows: list[tuple[object, ...]] = []

        for rec in records:
            row_counts_json = encode_json_compact(dict(rec.row_counts) if rec.row_counts else {})
            rows.append(
                (
                    run_id,
                    repo,
                    commit,
                    rec.target,
                    rec.impl_kind,
                    rec.status,
                    rec.input_hash,
                    rec.options_hash,
                    rec.duration_ms,
                    row_counts_json,
                    rec.error,
                    recorded_at,
                )
            )

        return self._backend.bulk_insert(
            "build.run_targets",
            rows,
            columns=(
                "run_id",
                "repo",
                "commit",
                "target",
                "impl_kind",
                "status",
                "input_hash",
                "options_hash",
                "duration_ms",
                "row_counts",
                "error",
                "recorded_at",
            ),
        )

    def list_run_targets(self, run_id: str) -> list[dict[str, Any]]:
        """List per-target records for a specific run.

        Parameters
        ----------
        run_id
            Run identifier to fetch targets for.

        Returns
        -------
        list[dict[str, Any]]
            List of target record dictionaries.
        """
        results = self._con.execute(
            """
            SELECT target, impl_kind, status, input_hash, options_hash,
                   duration_ms, row_counts, error, recorded_at
            FROM build.run_targets
            WHERE run_id = ?
            ORDER BY target
            """,
            [run_id],
        ).fetchall()

        return [
            {
                "target": row[0],
                "impl_kind": row[1],
                "status": row[2],
                "input_hash": row[3],
                "options_hash": row[4],
                "duration_ms": row[5],
                "row_counts": decode_json_list(row[6]) if row[6] else {},
                "error": row[7],
                "recorded_at": row[8],
            }
            for row in results
        ]

    def save_run_nodes(
        self,
        run_id: str,
        records: Sequence[NodeExecutionRecord],
    ) -> int:
        """Save node-level execution records for a build run.

        Parameters
        ----------
        run_id
            Parent run identifier.
        records
            Sequence of NodeExecutionRecord objects.

        Returns
        -------
        int
            Number of records inserted.
        """
        if not records:
            return 0

        rows = [
            (
                run_id,
                r.node_name,
                r.target,
                r.node_type,
                r.status,
                r.started_at,
                r.completed_at,
                r.duration_ms,
                r.error,
                encode_json_compact(r.tags or {}),
            )
            for r in records
        ]

        return self._backend.bulk_insert(
            "build.run_nodes",
            rows,
            columns=(
                "run_id",
                "node_name",
                "target",
                "node_type",
                "status",
                "started_at",
                "completed_at",
                "duration_ms",
                "error",
                "tags",
            ),
        )

    def record_scip_run(self, record: ScipRunRecord) -> None:
        """Upsert a SCIP telemetry record into build.scip_runs."""
        self._backend.ensure_table("build.scip_runs", create_if_missing=True)
        self._backend.upsert(
            "build.scip_runs",
            [
                (
                    record.run_id,
                    record.repo,
                    record.commit,
                    record.mode,
                    record.options_hash,
                    record.tool_version,
                    record.total_modules,
                    record.changed_modules,
                    record.deleted_modules,
                    record.changed_ratio,
                    record.batch_size,
                    record.batch_count,
                    record.decision,
                    record.ratio_gate_applied,
                    record.ratio_gate_min_modules,
                    record.ratio_gate_min_changed,
                    record.hash_source,
                    record.hash_source_breakdown,
                    record.hash_reused,
                    record.hash_computed,
                    record.plan_ms,
                    record.hash_ms,
                    record.tool_ms,
                    record.parse_ms,
                    record.merge_ms,
                    record.write_ms,
                    record.total_ms,
                    record.status,
                    record.error_summary,
                    record.output_scip,
                    record.recorded_at,
                )
            ],
            columns=(
                "run_id",
                "repo",
                "commit",
                "mode",
                "options_hash",
                "tool_version",
                "total_modules",
                "changed_modules",
                "deleted_modules",
                "changed_ratio",
                "batch_size",
                "batch_count",
                "decision",
                "ratio_gate_applied",
                "ratio_gate_min_modules",
                "ratio_gate_min_changed",
                "hash_source",
                "hash_source_breakdown",
                "hash_reused",
                "hash_computed",
                "plan_ms",
                "hash_ms",
                "tool_ms",
                "parse_ms",
                "merge_ms",
                "write_ms",
                "total_ms",
                "status",
                "error_summary",
                "output_scip",
                "recorded_at",
            ),
            upsert=UpsertSpec(
                conflict_columns=("run_id",),
                update_columns=(
                    "mode",
                    "options_hash",
                    "tool_version",
                    "total_modules",
                    "changed_modules",
                    "deleted_modules",
                    "changed_ratio",
                    "batch_size",
                    "batch_count",
                    "decision",
                    "ratio_gate_applied",
                    "ratio_gate_min_modules",
                    "ratio_gate_min_changed",
                    "hash_source",
                    "hash_source_breakdown",
                    "hash_reused",
                    "hash_computed",
                    "plan_ms",
                    "hash_ms",
                    "tool_ms",
                    "parse_ms",
                    "merge_ms",
                    "write_ms",
                    "total_ms",
                    "status",
                    "error_summary",
                    "output_scip",
                    "recorded_at",
                ),
            ),
        )

    def list_run_nodes(
        self,
        run_id: str,
        *,
        target: str | None = None,
    ) -> list[dict[str, Any]]:
        """List node records for a specific run.

        Parameters
        ----------
        run_id
            Run identifier to fetch nodes for.
        target
            Optional target filter.

        Returns
        -------
        list[dict[str, Any]]
            List of node record dictionaries.
        """
        query = """
            SELECT node_name, target, node_type, status, started_at,
                   completed_at, duration_ms, error, tags
            FROM build.run_nodes
            WHERE run_id = ?
        """
        params: list[Any] = [run_id]

        if target:
            query += " AND target = ?"
            params.append(target)

        query += " ORDER BY started_at"

        results = self._con.execute(query, params).fetchall()

        return [
            {
                "node_name": row[0],
                "target": row[1],
                "node_type": row[2],
                "status": row[3],
                "started_at": row[4],
                "completed_at": row[5],
                "duration_ms": row[6],
                "error": row[7],
                "tags": decode_json_list(row[8]) if row[8] else {},
            }
            for row in results
        ]


__all__ = [
    "BuildTracking",
    "ScipRunRecord",
]
