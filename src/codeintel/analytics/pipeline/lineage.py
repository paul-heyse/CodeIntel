"""Lineage tracking for dataset provenance.

This module provides data structures and storage for tracking
the provenance of dataset computations.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import msgspec

from codeintel.storage.gateway import DuckDBError
from codeintel.storage.sql_builder import SafeTable, render_sql

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


class _LineageRaw(msgspec.Struct):
    """Raw lineage record for JSON deserialization with validation.

    This struct provides type-safe parsing of JSON lineage data.
    All fields have defaults to handle missing data gracefully.
    """

    dataset: str = ""
    run_id: str = ""
    input_datasets: list[str] = msgspec.field(default_factory=list)
    input_hashes: list[str] = msgspec.field(default_factory=list)
    output_hash: str = ""
    row_count: int = 0
    computed_at: str | None = None
    duration_ms: float = 0.0
    version: str = "1.0.0"
    metadata: dict[str, object] = msgspec.field(default_factory=dict)


@dataclass(frozen=True)
class DatasetLineage:
    """Track provenance of a dataset computation.

    Lineage records capture the inputs, outputs, and execution
    metadata for each dataset computation, enabling:
    - Debugging computation issues
    - Understanding data dependencies
    - Detecting when recomputation is needed

    Attributes
    ----------
    dataset
        Name of the computed dataset.
    run_id
        Unique identifier for the pipeline run.
    input_datasets
        Names of input datasets consumed.
    input_hashes
        Content hashes of input datasets.
    output_hash
        Content hash of the output dataset.
    row_count
        Number of rows in the output.
    computed_at
        Timestamp when computation completed.
    duration_ms
        Computation time in milliseconds.
    version
        Version of the dataset specification used.
    metadata
        Additional execution metadata.
    """

    dataset: str
    run_id: str
    input_datasets: tuple[str, ...]
    input_hashes: tuple[str, ...]
    output_hash: str
    row_count: int
    computed_at: datetime
    duration_ms: float = 0.0
    version: str = "1.0.0"
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "dataset": self.dataset,
            "run_id": self.run_id,
            "input_datasets": list(self.input_datasets),
            "input_hashes": list(self.input_hashes),
            "output_hash": self.output_hash,
            "row_count": self.row_count,
            "computed_at": self.computed_at.isoformat(),
            "duration_ms": self.duration_ms,
            "version": self.version,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> DatasetLineage:
        """Create from dictionary.

        Uses msgspec for type-safe deserialization with validation.

        Parameters
        ----------
        data
            Dictionary representation.

        Returns
        -------
        DatasetLineage
            Reconstructed instance.
        """
        # Use msgspec for type-safe conversion
        raw = msgspec.convert(data, _LineageRaw)

        # Parse timestamp
        computed_at = (
            datetime.fromisoformat(raw.computed_at) if raw.computed_at else datetime.now(tz=UTC)
        )

        return cls(
            dataset=raw.dataset,
            run_id=raw.run_id,
            input_datasets=tuple(raw.input_datasets),
            input_hashes=tuple(raw.input_hashes),
            output_hash=raw.output_hash,
            row_count=raw.row_count,
            computed_at=computed_at,
            duration_ms=raw.duration_ms,
            version=raw.version,
            metadata=dict(raw.metadata),
        )


def compute_table_hash(
    gateway: StorageGateway,
    table: str,
    *,
    repo: str | None = None,
    commit: str | None = None,
    sample_size: int = 1000,
) -> str:
    """Compute a content hash for a table.

    Uses a sample of rows to compute a stable hash that changes
    when table contents change.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    table
        Fully qualified table name.
    repo
        Optional repository filter.
    commit
        Optional commit filter.
    sample_size
        Number of rows to sample for hashing.

    Returns
    -------
    str
        Hex digest of the content hash.
    """
    # Build query with optional filtering (SafeTable validates the table name)
    safe_table = SafeTable(table)
    params: list[object] = []

    where_parts: list[str] = []
    if repo is not None:
        where_parts.append("repo = ?")
        params.append(repo)
        if commit is not None:
            where_parts.append("commit = ?")
            params.append(commit)

    query_parts: list[str] = ["SELECT * FROM", str(safe_table)]
    if where_parts:
        query_parts.extend(["WHERE", " AND ".join(where_parts)])
    query_parts.extend(["LIMIT", str(sample_size)])
    query = render_sql(query_parts)

    try:
        result = gateway.con.execute(query, params)
        rows = result.fetchall()

        # Create stable string representation
        hasher = hashlib.sha256()
        for row in rows:
            row_str = "|".join(str(v) for v in row)
            hasher.update(row_str.encode())

        return hasher.hexdigest()[:16]
    except DuckDBError:
        log.warning("Failed to hash table %s", table, exc_info=True)
        return "error"


class LineageStore:
    """Storage for dataset lineage records.

    Provides methods for recording and querying lineage information.
    """

    def __init__(self, gateway: StorageGateway) -> None:
        """Initialize the lineage store.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        """
        self._gateway = gateway
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Ensure the lineage table exists."""
        query = """
            CREATE TABLE IF NOT EXISTS analytics.dataset_lineage (
                dataset VARCHAR NOT NULL,
                run_id VARCHAR NOT NULL,
                input_datasets JSON,
                input_hashes JSON,
                output_hash VARCHAR,
                row_count INTEGER,
                computed_at TIMESTAMP,
                duration_ms DOUBLE,
                version VARCHAR,
                metadata JSON,
                PRIMARY KEY (dataset, run_id)
            )
        """
        try:
            self._gateway.con.execute(query)
        except DuckDBError:
            log.debug("Lineage table creation skipped (may already exist)")

    def record(self, lineage: DatasetLineage) -> None:
        """Record a lineage entry.

        Parameters
        ----------
        lineage
            Lineage record to store.
        """
        query = """
            INSERT OR REPLACE INTO analytics.dataset_lineage
            (dataset, run_id, input_datasets, input_hashes, output_hash,
             row_count, computed_at, duration_ms, version, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        try:
            self._gateway.con.execute(
                query,
                [
                    lineage.dataset,
                    lineage.run_id,
                    json.dumps(list(lineage.input_datasets)),
                    json.dumps(list(lineage.input_hashes)),
                    lineage.output_hash,
                    lineage.row_count,
                    lineage.computed_at,
                    lineage.duration_ms,
                    lineage.version,
                    json.dumps(lineage.metadata),
                ],
            )
        except DuckDBError:
            log.warning("Failed to record lineage for %s", lineage.dataset, exc_info=True)

    def get_latest(self, dataset: str) -> DatasetLineage | None:
        """Get the most recent lineage for a dataset.

        Parameters
        ----------
        dataset
            Dataset name.

        Returns
        -------
        DatasetLineage | None
            Most recent lineage, or None if not found.
        """
        query = """
            SELECT dataset, run_id, input_datasets, input_hashes, output_hash,
                   row_count, computed_at, duration_ms, version, metadata
            FROM analytics.dataset_lineage
            WHERE dataset = ?
            ORDER BY computed_at DESC
            LIMIT 1
        """
        try:
            result = self._gateway.con.execute(query, [dataset])
            row = result.fetchone()
            if row is None:
                return None

            return DatasetLineage(
                dataset=str(row[0]),
                run_id=str(row[1]),
                input_datasets=tuple(json.loads(row[2] or "[]")),
                input_hashes=tuple(json.loads(row[3] or "[]")),
                output_hash=str(row[4] or ""),
                row_count=int(row[5] or 0),
                computed_at=row[6] if isinstance(row[6], datetime) else datetime.now(tz=UTC),
                duration_ms=float(row[7] or 0.0),
                version=str(row[8] or "1.0.0"),
                metadata=json.loads(row[9] or "{}"),
            )
        except DuckDBError:
            log.warning("Failed to get lineage for %s", dataset, exc_info=True)
            return None

    def get_by_run(self, run_id: str) -> list[DatasetLineage]:
        """Get all lineage records for a pipeline run.

        Parameters
        ----------
        run_id
            Pipeline run identifier.

        Returns
        -------
        list[DatasetLineage]
            All lineage records for the run.
        """
        query = """
            SELECT dataset, run_id, input_datasets, input_hashes, output_hash,
                   row_count, computed_at, duration_ms, version, metadata
            FROM analytics.dataset_lineage
            WHERE run_id = ?
            ORDER BY computed_at
        """
        try:
            result = self._gateway.con.execute(query, [run_id])
        except DuckDBError:
            log.warning("Failed to get lineage for run %s", run_id, exc_info=True)
            return []
        else:
            return [
                DatasetLineage(
                    dataset=str(row[0]),
                    run_id=str(row[1]),
                    input_datasets=tuple(json.loads(row[2] or "[]")),
                    input_hashes=tuple(json.loads(row[3] or "[]")),
                    output_hash=str(row[4] or ""),
                    row_count=int(row[5] or 0),
                    computed_at=row[6] if isinstance(row[6], datetime) else datetime.now(tz=UTC),
                    duration_ms=float(row[7] or 0.0),
                    version=str(row[8] or "1.0.0"),
                    metadata=json.loads(row[9] or "{}"),
                )
                for row in result.fetchall()
            ]

    def needs_recompute(
        self,
        dataset: str,
        input_hashes: dict[str, str],
    ) -> bool:
        """Check if a dataset needs recomputation.

        Compares current input hashes to the last recorded lineage.

        Parameters
        ----------
        dataset
            Dataset name.
        input_hashes
            Current hashes of input datasets.

        Returns
        -------
        bool
            True if recomputation is needed.
        """
        latest = self.get_latest(dataset)
        if latest is None:
            return True

        # Check if input hashes match
        if len(latest.input_datasets) != len(input_hashes):
            return True

        for inp, stored_hash in zip(latest.input_datasets, latest.input_hashes, strict=True):
            if inp not in input_hashes:
                return True
            if input_hashes[inp] != stored_hash:
                return True

        return False


__all__ = [
    "DatasetLineage",
    "LineageStore",
    "compute_table_hash",
]
