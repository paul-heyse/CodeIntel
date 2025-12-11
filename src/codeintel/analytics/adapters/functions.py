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
from typing import TYPE_CHECKING, Any, ClassVar, TypedDict, cast

import pandas as pd

from codeintel.analytics.adapters.base import BatchAdapter
from codeintel.analytics.adapters.schema_adapter import SchemaValidationMixin
from codeintel.analytics.utilities.datasets import (
    get_analytics_dataset_contract,
)
from codeintel.config.datasets import load_columns_by_table
from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend

if TYPE_CHECKING:
    from codeintel.config.datasets import FunctionMetricsRow, FunctionTypesRow
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


def _to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Convert a pandas DataFrame to record dictionaries.

    Returns
    -------
    list[dict[str, Any]]
        Records returned by ``DataFrame.to_dict(orient="records")``.
    """
    return cast("list[dict[str, Any]]", df.to_dict(orient="records"))


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
        """Iterate over function GOIDs using Ibis.

        Yields
        ------
        FunctionGoid
            Each function GOID in the snapshot.
        """
        tbl = self._gateway.ibis.table("core.goids")
        repo_filter = cast("Any", tbl.repo == self._snapshot.repo)
        commit_filter = cast("Any", tbl.commit == self._snapshot.commit)
        kind_filter = cast("Any", tbl.kind.isin(cast("Any", ["function", "method"])))
        expr = tbl.filter(repo_filter & commit_filter & kind_filter).select(
            "goid_h128",
            "urn",
            "repo",
            "commit",
            "rel_path",
            "language",
            "kind",
            "qualname",
            "start_line",
            "end_line",
        )
        df = cast("pd.DataFrame", expr.execute())

        for record in _to_records(df):
            goid_row: GoidRow = {
                "goid_h128": record["goid_h128"],
                "urn": record["urn"],
                "repo": record["repo"],
                "commit": record["commit"],
                "rel_path": record["rel_path"],
                "language": record["language"],
                "kind": record["kind"],
                "qualname": record["qualname"],
                "start_line": record["start_line"],
                "end_line": record["end_line"],
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


class FunctionMetricsAdapter(BatchAdapter["FunctionMetricsRow"], SchemaValidationMixin):
    """Adapter for analytics.function_metrics table.

    Handles loading source GOIDs and persisting function metrics rows.

    This adapter follows the ComputeAdapter pattern where:
    - `load_inputs()` loads source data (FunctionGoid) for computation
    - `load_outputs()` would load existing metrics (returns empty)
    - `persist()` writes computed FunctionMetricsRow

    Includes schema validation via SchemaValidationMixin.
    """

    table_key: ClassVar[str] = "analytics.function_metrics"

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
        return type(self).table_key

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

        contract = get_analytics_dataset_contract(self._gateway, self.table_name)
        columns = (
            contract.schema.column_names()
            if contract.schema is not None
            else tuple(load_columns_by_table().get(self.table_name, []))
        )
        tuple_rows = [contract.to_tuple(row) for row in rows]

        backend = DuckDBPolicyBackend(self._gateway)
        backend.delete_for_snapshot(self.table_name, repo=self.repo, commit=self.commit)
        backend.bulk_insert(
            self.table_name,
            tuple_rows,
            columns=columns,
        )

        log.info(
            "Persisted %d function metrics rows for %s@%s",
            len(rows),
            self.repo,
            self.commit,
        )
        return len(rows)

    def persist_with_validation(
        self,
        df: pd.DataFrame,
        *,
        strict: bool = False,
    ) -> int:
        """Persist a DataFrame with schema validation.

        Parameters
        ----------
        df
            DataFrame to validate and persist.
        strict
            If True, raise on validation failure. If False, log and proceed.

        Returns
        -------
        int
            Number of rows persisted.
        """
        validated_df = self.validate_dataframe(df) if strict else self.try_validate_dataframe(df)
        rows = cast("list[FunctionMetricsRow]", _to_records(validated_df))
        return self.persist(rows)


class FunctionTypesAdapter(BatchAdapter["FunctionTypesRow"], SchemaValidationMixin):
    """Adapter for analytics.function_types table.

    Handles persisting function type annotation rows.

    Includes schema validation via SchemaValidationMixin.
    """

    table_key: ClassVar[str] = "analytics.function_types"

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
        return type(self).table_key

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

        contract = get_analytics_dataset_contract(self._gateway, self.table_name)
        columns = (
            contract.schema.column_names()
            if contract.schema is not None
            else tuple(load_columns_by_table().get(self.table_name, []))
        )
        tuple_rows = [contract.to_tuple(row) for row in rows]

        backend = DuckDBPolicyBackend(self._gateway)
        backend.delete_for_snapshot(self.table_name, repo=self.repo, commit=self.commit)
        backend.bulk_insert(
            self.table_name,
            tuple_rows,
            columns=columns,
        )

        log.info(
            "Persisted %d function types rows for %s@%s",
            len(rows),
            self.repo,
            self.commit,
        )
        return len(rows)

    def persist_with_validation(
        self,
        df: pd.DataFrame,
        *,
        strict: bool = False,
    ) -> int:
        """Persist a DataFrame with schema validation.

        Parameters
        ----------
        df
            DataFrame to validate and persist.
        strict
            If True, raise on validation failure. If False, log and proceed.

        Returns
        -------
        int
            Number of rows persisted.
        """
        validated_df = self.validate_dataframe(df) if strict else self.try_validate_dataframe(df)
        rows = cast("list[FunctionTypesRow]", _to_records(validated_df))
        return self.persist(rows)


__all__ = [
    "FunctionGoid",
    "FunctionGoidLoader",
    "FunctionMetricsAdapter",
    "FunctionTypesAdapter",
    "GoidRow",
]
