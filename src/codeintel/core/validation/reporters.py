"""Validation reporters for structured finding collection.

This module provides reporter classes that collect validation findings
in a structured format suitable for persistence and analysis.

Column definitions and row collection for validation findings are provided
for both function-level and graph-level validation.

For persistence, use the ``to_rows()`` method with Hamilton materializers.

Classes
-------
BaseValidationReporter
    Base class for validation reporters with common fields.
FunctionValidationReporter
    Collects function-level validation findings.
GraphValidationReporter
    Collects graph-level validation findings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TypeVar, cast

import msgspec

from codeintel.core.helpers.payload import encode_payload
from codeintel.core.schemas.generated_rows.analytics import (
    AnalyticsFunctionValidationRow as FunctionValidationRow,
)
from codeintel.core.schemas.generated_rows.analytics import (
    AnalyticsGraphValidationRow as GraphValidationRow,
)
from codeintel.core.schemas.row_serialization import row_serializer_for_table_key
from codeintel.core.serialization.msgspec import to_builtins
from codeintel.core.time import utc_now

RowT = TypeVar("RowT")

FUNCTION_VALIDATION_TABLE_KEY = "analytics.function_validation"
GRAPH_VALIDATION_TABLE_KEY = "analytics.graph_validation"
FUNCTION_VALIDATION_COLS: tuple[str, ...] = (
    "repo",
    "commit",
    "function_goid_h128",
    "rel_path",
    "qualname",
    "issue",
    "detail",
    "created_at",
)
GRAPH_VALIDATION_COLS: tuple[str, ...] = (
    "repo",
    "commit",
    "graph_name",
    "entity_id",
    "issue",
    "severity",
    "rel_path",
    "detail",
    "metadata",
    "created_at",
)


class FunctionValidationFinding(msgspec.Struct, frozen=True):
    """Structured payload for function validation findings."""

    repo: str
    commit: str
    function_goid_h128: int
    rel_path: str
    qualname: str
    issue: str
    detail: str
    created_at: datetime

    def to_row(self) -> FunctionValidationRow:
        """Convert finding into a row mapping for persistence.

        Returns
        -------
        FunctionValidationRow
            Row mapping ready for persistence.
        """
        return cast("FunctionValidationRow", to_builtins(self))


class GraphValidationFinding(msgspec.Struct, frozen=True):
    """Structured payload for graph validation findings."""

    repo: str
    commit: str
    graph_name: str
    entity_id: str
    issue: str
    severity: str | None
    rel_path: str | None
    detail: str
    metadata: bytes | None
    created_at: datetime

    def to_row(self) -> GraphValidationRow:
        """Convert finding into a row mapping for persistence.

        Returns
        -------
        GraphValidationRow
            Row mapping ready for persistence.
        """
        return cast("GraphValidationRow", to_builtins(self))


@dataclass
class BaseValidationReporter[RowT]:
    """Collect validation rows for persistence via policy backend.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit hash.

    Attributes
    ----------
    rows
        List of accumulated validation rows.
    total
        Total count of recorded findings.
    """

    repo: str
    commit: str
    rows: list[RowT] = field(default_factory=list)
    total: int = 0


@dataclass
class FunctionValidationReporter(BaseValidationReporter[FunctionValidationRow]):
    """Validation reporter for function-level parsing/span issues.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit hash.

    Attributes
    ----------
    parse_failed
        Count of parse failures.
    span_not_found
        Count of span lookup failures.
    unknown_functions
        Count of unknown function references.

    Examples
    --------
    >>> reporter = FunctionValidationReporter(repo="org/repo", commit="abc123")
    >>> reporter.record(
    ...     function_goid_h128=12345,
    ...     rel_path="src/main.py",
    ...     qualname="main",
    ...     issue="parse_failed",
    ...     detail="Syntax error on line 10",
    ... )
    >>> rows = reporter.to_rows()
    """

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
        """Record a validation finding for a function GOID.

        Parameters
        ----------
        function_goid_h128
            Function GOID hash.
        rel_path
            Relative path to the source file.
        qualname
            Fully qualified function name.
        issue
            Issue identifier/code.
        detail
            Human-readable description of the issue.
        """
        self.total += 1
        if issue == "parse_failed":
            self.parse_failed += 1
        elif issue == "span_not_found":
            self.span_not_found += 1
        elif issue == "unknown_function":
            self.unknown_functions += 1

        finding = FunctionValidationFinding(
            repo=self.repo,
            commit=self.commit,
            function_goid_h128=function_goid_h128,
            rel_path=rel_path,
            qualname=qualname,
            issue=issue,
            detail=detail,
            created_at=gateway_timestamp(),
        )
        self.rows.append(finding.to_row())

    def to_rows(self) -> tuple[tuple[object, ...], ...]:
        """Return accumulated rows as tuples without writing.

        Use this method with Hamilton materializers for persistence.

        Returns
        -------
        tuple[tuple[object, ...], ...]
            Accumulated validation rows ready for materialization.
        """
        serializer = row_serializer_for_table_key(FUNCTION_VALIDATION_TABLE_KEY)
        return tuple(serializer(r) for r in self.rows)


@dataclass
class GraphValidationReporter(BaseValidationReporter[GraphValidationRow]):
    """Validation reporter for graph-level issues.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit hash.

    Examples
    --------
    >>> reporter = GraphValidationReporter(repo="org/repo", commit="abc123")
    >>> reporter.record(
    ...     graph_name="import_graph",
    ...     entity_id="module.py",
    ...     issue="orphan-module",
    ...     detail="Module has no imports or exports",
    ... )
    >>> rows = reporter.to_rows()
    """

    def record(
        self,
        *,
        graph_name: str,
        issue: str,
        detail: str,
        entity_id: str | None = None,
        extras: Mapping[str, object | None] | None = None,
    ) -> None:
        """Record a graph validation finding.

        Parameters
        ----------
        graph_name
            Name of the graph being validated.
        issue
            Issue identifier/code.
        detail
            Human-readable description.
        entity_id
            Identifier of the entity with the finding.
        extras
            Optional additional metadata (severity, rel_path, metadata).
        """
        self.total += 1
        severity = cast("str | None", extras.get("severity") if extras is not None else None)
        rel_path = cast("str | None", extras.get("rel_path") if extras is not None else None)
        metadata = _encode_metadata(extras.get("metadata") if extras is not None else None)
        entity = entity_id or graph_name
        finding = GraphValidationFinding(
            repo=self.repo,
            commit=self.commit,
            graph_name=graph_name,
            entity_id=entity,
            issue=issue,
            severity=severity,
            rel_path=rel_path,
            detail=detail,
            metadata=metadata,
            created_at=gateway_timestamp(),
        )
        self.rows.append(finding.to_row())

    def to_rows(self) -> tuple[tuple[object, ...], ...]:
        """Return accumulated rows as tuples without writing.

        Use this method with Hamilton materializers for persistence.

        Returns
        -------
        tuple[tuple[object, ...], ...]
            Accumulated validation rows ready for materialization.
        """
        serializer = row_serializer_for_table_key(GRAPH_VALIDATION_TABLE_KEY)
        return tuple(serializer(r) for r in self.rows)


def _encode_metadata(value: object | None) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return encode_payload(value)
    if isinstance(value, (str, int, float, bool, Mapping, Sequence)):
        return encode_payload(value)
    return encode_payload(str(value))


def gateway_timestamp() -> datetime:
    """Return a timezone-aware timestamp for validation rows.

    Returns
    -------
    datetime
        Current UTC timestamp.
    """
    return utc_now()


__all__ = [
    "FUNCTION_VALIDATION_COLS",
    "GRAPH_VALIDATION_COLS",
    "BaseValidationReporter",
    "FunctionValidationFinding",
    "FunctionValidationReporter",
    "GraphValidationFinding",
    "GraphValidationReporter",
    "gateway_timestamp",
]
