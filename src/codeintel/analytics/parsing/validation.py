"""Validation reporters shared across analytics domains."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TypeVar, cast

from codeintel.storage.gateway import StorageGateway
from codeintel.storage.rows import (
    FunctionValidationRow,
    GraphValidationRow,
    function_validation_row_to_tuple,
    graph_validation_row_to_tuple,
)

RowT = TypeVar("RowT")


@dataclass
class BaseValidationReporter[RowT]:
    """Collect validation rows and flush them to DuckDB."""

    repo: str
    commit: str
    rows: list[RowT] = field(default_factory=list)
    total: int = 0

    def flush(self, gateway: StorageGateway) -> None:
        """Persist collected rows."""
        raise NotImplementedError


@dataclass
class FunctionValidationReporter(BaseValidationReporter[FunctionValidationRow]):
    """Validation reporter for function-level parsing/span issues."""

    parse_failed: int = 0
    span_not_found: int = 0
    unknown_functions: int = 0

    def record(
        self,
        *,
        function_goid_h128: int,
        rel_path: str,
        qualname: str,
        issue: str,
        detail: str,
    ) -> None:
        """Record a validation finding for a function GOID."""
        self.total += 1
        if issue == "parse_failed":
            self.parse_failed += 1
        elif issue == "span_not_found":
            self.span_not_found += 1
        elif issue == "unknown_function":
            self.unknown_functions += 1

        row: FunctionValidationRow = {
            "repo": self.repo,
            "commit": self.commit,
            "function_goid_h128": function_goid_h128,
            "rel_path": rel_path,
            "qualname": qualname,
            "issue": issue,
            "detail": detail,
            "created_at": gateway_timestamp(),
        }
        self.rows.append(row)

    def flush(self, gateway: StorageGateway) -> None:
        """Persist recorded function validation rows."""
        if not self.rows:
            return
        tuples = [function_validation_row_to_tuple(r) for r in self.rows]
        con = gateway.con
        con.executemany(
            """
            INSERT INTO analytics.function_validation (
                repo, commit, function_goid_h128, rel_path, qualname, issue, detail, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            tuples,
        )
        self.rows.clear()


@dataclass
class GraphValidationReporter(BaseValidationReporter[GraphValidationRow]):
    """Validation reporter for graph-level issues."""

    def record(
        self,
        *,
        graph_name: str,
        issue: str,
        detail: str,
        entity_id: str | None = None,
        extras: Mapping[str, object | None] | None = None,
    ) -> None:
        """Record a graph validation finding."""
        self.total += 1
        severity = cast("str | None", extras.get("severity") if extras is not None else None)
        rel_path = cast("str | None", extras.get("rel_path") if extras is not None else None)
        metadata = extras.get("metadata") if extras is not None else None
        entity = entity_id or graph_name
        row: GraphValidationRow = {
            "repo": self.repo,
            "commit": self.commit,
            "graph_name": graph_name,
            "entity_id": entity,
            "issue": issue,
            "severity": severity,
            "rel_path": rel_path,
            "detail": detail,
            "metadata": metadata,
            "created_at": gateway_timestamp(),
        }
        self.rows.append(row)

    def flush(self, gateway: StorageGateway) -> None:
        """Persist recorded graph validation rows."""
        if not self.rows:
            return
        tuples = [graph_validation_row_to_tuple(r) for r in self.rows]
        con = gateway.con
        con.executemany(
            """
            INSERT INTO analytics.graph_validation (
                repo, commit, graph_name, entity_id, issue, severity, rel_path, detail, metadata, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            tuples,
        )
        self.rows.clear()


def gateway_timestamp() -> datetime:
    """
    Return a timezone-aware timestamp for validation rows.

    Returns
    -------
    datetime
        Current UTC timestamp.
    """
    return datetime.now(UTC)
