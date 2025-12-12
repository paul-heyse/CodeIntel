"""Build manifest and run tracking persistence for DuckDB.

This module provides persistent tracking of build output manifests and
build runs, enabling cache invalidation and observability of the build
system.

All DuckDB access is encapsulated here, following the storage layer pattern.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from codeintel.build.manifest import BuildRunRecord, OutputManifest
from codeintel.storage.helpers.json import decode_json_list, encode_json_compact

if TYPE_CHECKING:
    from collections.abc import Sequence

    from duckdb import DuckDBPyConnection

    from codeintel.build.hamilton.manifest_hook import TargetRunRecord
    from codeintel.build.manifest import BuildStatus


# =============================================================================
# Row Parsing Helpers
# =============================================================================


def _parse_manifest_row(row: tuple[Any, ...]) -> OutputManifest:
    """Parse a DuckDB row into an OutputManifest.

    Centralizes type coercion from DuckDB result tuples to typed dataclass.

    Parameters
    ----------
    row
        DuckDB row tuple from output_manifests table.
        Expected column order: target, repo, commit, plugin, computed_at,
        duration_ms, input_hash, output_hash, row_count, options_hash

    Returns
    -------
    OutputManifest
        Typed manifest dataclass.
    """
    return OutputManifest(
        target=str(row[0]),
        repo=str(row[1]),
        commit=str(row[2]),
        plugin=str(row[3]),
        computed_at=cast("datetime", row[4]),
        duration_ms=float(row[5]),
        input_hash=str(row[6]),
        output_hash=str(row[7]) if row[7] is not None else None,
        row_count=int(row[8]) if row[8] is not None else None,
        options_hash=str(row[9]) if row[9] is not None else None,
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
        requested_targets=_deserialize_targets(cast("str | None", row[3])),
        computed_targets=_deserialize_targets(cast("str | None", row[4])),
        skipped_targets=_deserialize_targets(cast("str | None", row[5])),
        started_at=cast("datetime", row[6]),
        completed_at=cast("datetime | None", row[7]),
        status=cast("BuildStatus", row[8]),
        error_summary=str(row[9]) if row[9] is not None else None,
        duration_ms=float(row[10]) if row[10] is not None else None,
    )


def _now() -> datetime:
    """Return current UTC timestamp.

    Returns
    -------
    datetime
        Current datetime with UTC timezone.
    """
    return datetime.now(tz=UTC)


def _serialize_targets(targets: tuple[str, ...]) -> str:
    """Serialize targets to JSON array.

    Parameters
    ----------
    targets
        Tuple of target names.

    Returns
    -------
    str
        JSON-encoded array of target names.
    """
    return encode_json_compact(list(targets))


def _deserialize_targets(raw: str | None) -> tuple[str, ...]:
    """Deserialize targets from JSON array.

    Parameters
    ----------
    raw
        JSON-encoded array or None.

    Returns
    -------
    tuple[str, ...]
        Tuple of target names.
    """
    if not raw:
        return ()
    items = decode_json_list(raw)
    return tuple(str(x) for x in items)


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
    >>> tracking = BuildTracking(connection)
    >>> tracking.save_manifest(manifest)
    >>> loaded = tracking.load_manifest("risk_factors", "org/repo", "abc123")
    """

    def __init__(self, con: DuckDBPyConnection) -> None:
        """Initialize build tracking accessor.

        Parameters
        ----------
        con
            DuckDB connection to use for all operations.
        """
        self._con = con

    # =========================================================================
    # Manifest Operations
    # =========================================================================

    def save_manifest(self, manifest: OutputManifest) -> None:
        """Save or update an output manifest.

        Uses INSERT OR REPLACE to upsert the manifest record.

        Parameters
        ----------
        manifest
            The manifest to save.
        """
        self._con.execute(
            """
            INSERT OR REPLACE INTO build.output_manifests (
                target, repo, commit, plugin, computed_at, duration_ms,
                input_hash, output_hash, row_count, options_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                manifest.target,
                manifest.repo,
                manifest.commit,
                manifest.plugin,
                manifest.computed_at,
                manifest.duration_ms,
                manifest.input_hash,
                manifest.output_hash,
                manifest.row_count,
                manifest.options_hash,
            ],
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
            SELECT target, repo, commit, plugin, computed_at, duration_ms,
                   input_hash, output_hash, row_count, options_hash
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
            SELECT target, repo, commit, plugin, computed_at, duration_ms,
                   input_hash, output_hash, row_count, options_hash
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

    # =========================================================================
    # Run Tracking Operations
    # =========================================================================

    def start_run(self, record: BuildRunRecord) -> None:
        """Record the start of a build run.

        Parameters
        ----------
        record
            The run record to save.
        """
        self._con.execute(
            """
            INSERT INTO build.runs (
                run_id, repo, commit, requested_targets, computed_targets,
                skipped_targets, started_at, completed_at, status,
                error_summary, duration_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                record.run_id,
                record.repo,
                record.commit,
                _serialize_targets(record.requested_targets),
                _serialize_targets(record.computed_targets),
                _serialize_targets(record.skipped_targets),
                record.started_at,
                record.completed_at,
                record.status,
                record.error_summary,
                record.duration_ms,
            ],
        )

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
        completed_at = _now()
        # Calculate duration from started_at
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
                _serialize_targets(computed_targets),
                _serialize_targets(skipped_targets),
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

    # =========================================================================
    # Run Target Operations (Phase 2)
    # =========================================================================

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

        recorded_at = _now()
        rows: list[tuple[object, ...]] = []

        for rec in records:
            row_counts_json = encode_json_compact(dict(rec.row_counts) if rec.row_counts else {})
            rows.append((
                run_id,
                repo,
                commit,
                rec.target,
                rec.plugin_name,
                rec.status,
                rec.input_hash,
                rec.options_hash,
                rec.duration_ms,
                row_counts_json,
                rec.error,
                recorded_at,
            ))

        self._con.executemany(
            """
            INSERT INTO build.run_targets (
                run_id, repo, commit, target, plugin, status,
                input_hash, options_hash, duration_ms, row_counts,
                error, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

        return len(rows)

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
            SELECT target, plugin, status, input_hash, options_hash,
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
                "plugin": row[1],
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


__all__ = [
    "BuildTracking",
]
