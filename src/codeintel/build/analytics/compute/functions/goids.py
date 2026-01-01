"""Function GOID types and loading utilities.

This module provides data types and utilities for working with function
global object identifiers (GOIDs), including loading from the database
and grouping by file path.

Example
-------
>>> from codeintel.build.analytics.compute.functions.goids import FunctionGoidLoader
>>> loader = FunctionGoidLoader(gateway, snapshot)
>>> for goid in loader.iter_goids():
...     print(f"{goid.qualname}: {goid.start_line}-{goid.end_line}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

from codeintel.storage.duckdb_types import ColumnExpression, ConstantExpression
from codeintel.storage.query_results import (
    coerce_int,
    coerce_optional_int,
    records_from_relation,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway


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
        """Iterate over function GOIDs using DuckDB relations.

        Yields
        ------
        FunctionGoid
            Each function GOID in the snapshot.
        """
        predicate = (ColumnExpression("repo") == ConstantExpression(self._snapshot.repo)) & (
            ColumnExpression("commit") == ConstantExpression(self._snapshot.commit)
        )
        kind_literals = [ConstantExpression("function"), ConstantExpression("method")]
        relation = (
            self._gateway.relation_from_table_key("core.goids")
            .filter(predicate)
            .filter(ColumnExpression("kind").isin(*kind_literals))
            .select(
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
        )
        for record in records_from_relation(relation):
            goid_row: GoidRow = {
                "goid_h128": coerce_int(record["goid_h128"], ctx="goid_h128"),
                "urn": str(record["urn"]),
                "repo": str(record["repo"]),
                "commit": str(record["commit"]),
                "rel_path": str(record["rel_path"]),
                "language": str(record["language"]),
                "kind": str(record["kind"]),
                "qualname": str(record["qualname"]),
                "start_line": coerce_int(record["start_line"], ctx="start_line"),
                "end_line": coerce_optional_int(record.get("end_line"), ctx="end_line"),
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


__all__ = [
    "FunctionGoid",
    "FunctionGoidLoader",
    "GoidRow",
]
