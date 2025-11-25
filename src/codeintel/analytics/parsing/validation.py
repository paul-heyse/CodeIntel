"""Validation reporters shared across analytics domains."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Generic, List, TypeVar

from codeintel.models.rows import (
    FunctionValidationRow,
    GraphValidationRow,
    function_validation_row_to_tuple,
    graph_validation_row_to_tuple,
)
from codeintel.storage.gateway import StorageGateway

RowT = TypeVar("RowT")


@dataclass
class BaseValidationReporter(Generic[RowT]):
    """Collect validation rows and flush them to DuckDB."""

    repo: str
    commit: str
    rows: List[RowT] = field(default_factory=list)
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
        kind: str,
        message: str,
    ) -> None:
        self.total += 1
        if kind == "parse_failed":
            self.parse_failed += 1
        elif kind == "span_not_found":
            self.span_not_found += 1
        elif kind == "unknown_function":
            self.unknown_functions += 1

        row: FunctionValidationRow = {
            "repo": self.repo,
            "commit": self.commit,
            "function_goid_h128": function_goid_h128,
            "kind": kind,
            "message": message,
            "created_at": gateway_timestamp(),
        }
        self.rows.append(row)

    def flush(self, gateway: StorageGateway) -> None:
        if not self.rows:
            return
        tuples = [function_validation_row_to_tuple(r) for r in self.rows]
        con = gateway.con
        con.executemany(
            """
            INSERT INTO analytics.function_validation (
                repo, commit, function_goid_h128, kind, message, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
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
        entity_id: str,
        kind: str,
        message: str,
    ) -> None:
        self.total += 1
        row: GraphValidationRow = {
            "repo": self.repo,
            "commit": self.commit,
            "graph_name": graph_name,
            "entity_id": entity_id,
            "kind": kind,
            "message": message,
            "created_at": gateway_timestamp(),
        }
        self.rows.append(row)

    def flush(self, gateway: StorageGateway) -> None:
        if not self.rows:
            return
        tuples = [graph_validation_row_to_tuple(r) for r in self.rows]
        con = gateway.con
        con.executemany(
            """
            INSERT INTO analytics.graph_validation (
                repo, commit, graph_name, entity_id, kind, message, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            tuples,
        )
        self.rows.clear()


def gateway_timestamp() -> datetime:
    """Return a timezone-aware timestamp for validation rows."""
    return datetime.now(timezone.utc)
