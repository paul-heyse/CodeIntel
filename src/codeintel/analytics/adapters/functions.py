"""Adapters for function analytics persistence.

This module provides adapters for loading function GOIDs and persisting
function metrics and types to DuckDB.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from codeintel.analytics.adapters.base import BatchAdapter, DeleteScope
from codeintel.analytics.utilities.datasets import (
    get_analytics_dataset_contract,
    insert_analytics_rows,
)
from codeintel.storage.sql_helpers import ensure_schema

if TYPE_CHECKING:
    from codeintel.config.datasets import FunctionMetricsRow, FunctionTypesRow
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


class GoidRow(TypedDict):
    """Row structure for function GOIDs from DuckDB."""

    goid_h128: int
    urn: str
    repo: str
    commit: str
    rel_path: str
    language: str
    kind: str
    qualname: str
    start_line: int
    end_line: int | None


@dataclass(frozen=True)
class FunctionGoid:
    """Function GOID metadata loaded from database.

    Attributes
    ----------
    goid
        The global object identifier (hash).
    urn
        Uniform resource name for the function.
    repo
        Repository identifier.
    commit
        Commit identifier.
    rel_path
        Relative path to the source file.
    language
        Programming language.
    kind
        Symbol kind ("function" or "method").
    qualname
        Qualified name of the function.
    start_line
        Starting line number (1-indexed).
    end_line
        Ending line number (1-indexed).
    """

    goid: int
    urn: str
    repo: str
    commit: str
    rel_path: str
    language: str
    kind: str
    qualname: str
    start_line: int
    end_line: int

    @classmethod
    def from_row(cls, row: GoidRow) -> FunctionGoid:
        """Create a FunctionGoid from a database row.

        Parameters
        ----------
        row
            Database row dictionary.

        Returns
        -------
        FunctionGoid
            Constructed instance.
        """
        end_line_raw = row["end_line"]
        end_line = int(end_line_raw) if end_line_raw is not None else int(row["start_line"])
        return cls(
            goid=int(row["goid_h128"]),
            urn=str(row["urn"]),
            repo=str(row["repo"]),
            commit=str(row["commit"]),
            rel_path=str(row["rel_path"]).replace("\\", "/"),
            language=str(row["language"]),
            kind=str(row["kind"]),
            qualname=str(row["qualname"]),
            start_line=int(row["start_line"]),
            end_line=end_line,
        )


class FunctionGoidLoader:
    """Loader for function GOIDs from core.goids table.

    This class handles loading function and method GOIDs from the database
    and provides iteration and grouping capabilities.
    """

    def __init__(
        self,
        gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Initialize the loader.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        snapshot
            Repository snapshot reference.
        """
        self._gateway = gateway
        self._snapshot = snapshot

    def load_all(self) -> list[FunctionGoid]:
        """Load all function GOIDs for the snapshot.

        Returns
        -------
        list[FunctionGoid]
            All function and method GOIDs.
        """
        return list(self.iter_goids())

    def iter_goids(self) -> Iterator[FunctionGoid]:
        """Iterate over function GOIDs.

        Yields
        ------
        FunctionGoid
            Each function GOID in the snapshot.
        """
        query = """
            SELECT
                goid_h128,
                urn,
                repo,
                commit,
                rel_path,
                language,
                kind,
                qualname,
                start_line,
                end_line
            FROM core.goids
            WHERE repo = ? AND commit = ?
              AND kind IN ('function', 'method')
        """
        result = self._gateway.con.execute(
            query,
            [self._snapshot.repo, self._snapshot.commit],
        )

        for row in result.fetchall():
            goid_row: GoidRow = {
                "goid_h128": row[0],
                "urn": row[1],
                "repo": row[2],
                "commit": row[3],
                "rel_path": row[4],
                "language": row[5],
                "kind": row[6],
                "qualname": row[7],
                "start_line": row[8],
                "end_line": row[9],
            }
            yield FunctionGoid.from_row(goid_row)

    def group_by_file(self) -> dict[str, list[FunctionGoid]]:
        """Group GOIDs by their relative file path.

        Returns
        -------
        dict[str, list[FunctionGoid]]
            GOIDs grouped by rel_path.
        """
        by_file: dict[str, list[FunctionGoid]] = {}
        for goid in self.iter_goids():
            by_file.setdefault(goid.rel_path, []).append(goid)
        return by_file

    def resolve_abs_path(self, goid: FunctionGoid) -> Path:
        """Resolve the absolute path for a GOID.

        Parameters
        ----------
        goid
            The function GOID.

        Returns
        -------
        Path
            Absolute path to the source file.
        """
        return (self._snapshot.repo_root / goid.rel_path).resolve()


class FunctionMetricsAdapter(BatchAdapter["FunctionMetricsRow"]):
    """Adapter for analytics.function_metrics table.

    Handles loading source GOIDs and persisting function metrics rows.

    This adapter follows the ComputeAdapter pattern where:
    - `load_inputs()` loads source data (FunctionGoid) for computation
    - `load_outputs()` would load existing metrics (returns empty)
    - `persist()` writes computed FunctionMetricsRow
    """

    def __init__(
        self,
        gateway: StorageGateway,
        snapshot: SnapshotRef,
        *,
        timestamp: datetime | None = None,
    ) -> None:
        """Initialize the adapter.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        snapshot
            Repository snapshot reference.
        timestamp
            Optional timestamp for created_at field.
        """
        super().__init__(gateway, snapshot)
        self._timestamp = timestamp
        self._goid_loader = FunctionGoidLoader(gateway, snapshot)

    @property
    def table_name(self) -> str:
        """Return the target table name."""
        return "analytics.function_metrics"

    @property
    def goid_loader(self) -> FunctionGoidLoader:
        """Return the GOID loader."""
        return self._goid_loader

    def load_inputs(self) -> Iterator[FunctionGoid]:
        """Load function GOIDs as input for computation.

        This is the preferred method for loading source data.
        Use this instead of `load()` in new code.

        Returns
        -------
        Iterator[FunctionGoid]
            Iterator over function GOIDs.
        """
        return self._goid_loader.iter_goids()

    @staticmethod
    def load_outputs() -> Iterator[FunctionMetricsRow]:
        """Load existing metrics rows (not implemented - returns empty).

        Returns
        -------
        Iterator[FunctionMetricsRow]
            Empty iterator (metrics are computed, not loaded).
        """
        return iter([])

    def load(self) -> Iterator[FunctionMetricsRow]:
        """Load metrics rows from the database.

        Returns
        -------
        Iterator[FunctionMetricsRow]
            Empty iterator (metrics are computed via load_inputs(), not loaded).

        Notes
        -----
        This adapter computes metrics from GOIDs loaded via `load_inputs()`.
        The `load()` method returns empty since metrics aren't pre-existing.
        """
        return self.load_outputs()

    def persist(self, rows: Sequence[FunctionMetricsRow]) -> int:
        """Persist function metrics rows.

        Parameters
        ----------
        rows
            Rows to persist.

        Returns
        -------
        int
            Number of rows persisted.
        """
        if not rows:
            return 0

        ensure_schema(self._gateway.con, self.table_name)
        contract = get_analytics_dataset_contract(self._gateway, self.table_name)
        scope = f"{self.repo}@{self.commit}"

        insert_analytics_rows(
            self._gateway,
            contract,
            list(rows),
            delete_scope=DeleteScope(repo=self.repo, commit=self.commit),
            scope=scope,
        )

        log.info(
            "Persisted %d function metrics rows for %s@%s",
            len(rows),
            self.repo,
            self.commit,
        )
        return len(rows)


class FunctionTypesAdapter(BatchAdapter["FunctionTypesRow"]):
    """Adapter for analytics.function_types table.

    Handles persisting function type annotation rows.
    """

    def __init__(
        self,
        gateway: StorageGateway,
        snapshot: SnapshotRef,
        *,
        timestamp: datetime | None = None,
    ) -> None:
        """Initialize the adapter.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        snapshot
            Repository snapshot reference.
        timestamp
            Optional timestamp for created_at field.
        """
        super().__init__(gateway, snapshot)
        self._timestamp = timestamp

    @property
    def table_name(self) -> str:
        """Return the target table name."""
        return "analytics.function_types"

    def load(self) -> Iterator[FunctionTypesRow]:
        """Raise NotImplementedError as types are computed not loaded.

        Raises
        ------
        NotImplementedError
            This adapter is write-only.
        """
        message = "FunctionTypesAdapter does not support loading"
        raise NotImplementedError(message)

    def persist(self, rows: Sequence[FunctionTypesRow]) -> int:
        """Persist function types rows.

        Parameters
        ----------
        rows
            Rows to persist.

        Returns
        -------
        int
            Number of rows persisted.
        """
        if not rows:
            return 0

        ensure_schema(self._gateway.con, self.table_name)
        contract = get_analytics_dataset_contract(self._gateway, self.table_name)
        scope = f"{self.repo}@{self.commit}"

        insert_analytics_rows(
            self._gateway,
            contract,
            list(rows),
            delete_scope=DeleteScope(repo=self.repo, commit=self.commit),
            scope=scope,
        )

        log.info(
            "Persisted %d function types rows for %s@%s",
            len(rows),
            self.repo,
            self.commit,
        )
        return len(rows)


__all__ = [
    "FunctionGoid",
    "FunctionGoidLoader",
    "FunctionMetricsAdapter",
    "FunctionTypesAdapter",
    "GoidRow",
]
