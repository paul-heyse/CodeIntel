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

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


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

        Parameters
        ----------
        data
            Dictionary representation.

        Returns
        -------
        DatasetLineage
            Reconstructed instance.
        """
        computed_at = data.get("computed_at")
        if isinstance(computed_at, str):
            computed_at = datetime.fromisoformat(computed_at)
        elif not isinstance(computed_at, datetime):
            computed_at = datetime.now(tz=UTC)

        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        return cls(
            dataset=str(data.get("dataset", "")),
            run_id=str(data.get("run_id", "")),
            input_datasets=tuple(data.get("input_datasets", [])),  # type: ignore[arg-type]
            input_hashes=tuple(data.get("input_hashes", [])),  # type: ignore[arg-type]
            output_hash=str(data.get("output_hash", "")),
            row_count=int(data.get("row_count", 0)),  # type: ignore[arg-type]
            computed_at=computed_at,
            duration_ms=float(data.get("duration_ms", 0.0)),  # type: ignore[arg-type]
            version=str(data.get("version", "1.0.0")),
            metadata=metadata,
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
    # Build query with optional filtering
    query = f"SELECT * FROM {table}"  # noqa: S608
    params: list[object] = []

    where_parts: list[str] = []
    if repo is not None:
        where_parts.append("repo = ?")
        params.append(repo)
        if commit is not None:
            where_parts.append("commit = ?")
            params.append(commit)

    if where_parts:
        query += " WHERE " + " AND ".join(where_parts)

    query += f" LIMIT {sample_size}"

    try:
        result = gateway.con.execute(query, params)
        rows = result.fetchall()

        # Create stable string representation
        hasher = hashlib.sha256()
        for row in rows:
            row_str = "|".join(str(v) for v in row)
            hasher.update(row_str.encode())

        return hasher.hexdigest()[:16]
    except Exception:
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
        except Exception:
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
        except Exception:
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
                computed_at=row[6] if isinstance(row[6], datetime) else datetime.now(),
                duration_ms=float(row[7] or 0.0),
                version=str(row[8] or "1.0.0"),
                metadata=json.loads(row[9] or "{}"),
            )
        except Exception:
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
            records = []
            for row in result.fetchall():
                records.append(
                    DatasetLineage(
                        dataset=str(row[0]),
                        run_id=str(row[1]),
                        input_datasets=tuple(json.loads(row[2] or "[]")),
                        input_hashes=tuple(json.loads(row[3] or "[]")),
                        output_hash=str(row[4] or ""),
                        row_count=int(row[5] or 0),
                        computed_at=row[6] if isinstance(row[6], datetime) else datetime.now(),
                        duration_ms=float(row[7] or 0.0),
                        version=str(row[8] or "1.0.0"),
                        metadata=json.loads(row[9] or "{}"),
                    )
                )
            return records
        except Exception:
            log.warning("Failed to get lineage for run %s", run_id, exc_info=True)
            return []

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

        for inp, stored_hash in zip(
            latest.input_datasets, latest.input_hashes, strict=True
        ):
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
