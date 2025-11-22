"""Shared typed protocols and TypedDicts to reduce use of ``Any``."""

from __future__ import annotations

from typing import Protocol, TypedDict


class PytestCallEntry(TypedDict, total=False):
    """Subset of pytest-json-report call data."""

    duration: float


class PytestTestEntry(TypedDict, total=False):
    """Shape of a pytest-json-report test object."""

    nodeid: str
    keywords: dict[str, bool]
    outcome: str
    status: str
    call: PytestCallEntry


class PyreflyError(TypedDict, total=False):
    """Shape of a pyrefly JSON diagnostic."""

    path: str
    line: int | None
    column: int | None
    message: str
    code: str | None


class ScipOccurrence(TypedDict, total=False):
    """Occurrence entry within a SCIP JSON document."""

    symbol: str
    symbol_roles: int


class ScipDocument(TypedDict, total=False):
    """SCIP JSON document emitted by scip-python."""

    relative_path: str
    occurrences: list[ScipOccurrence]


class HasModelDump(Protocol):
    """Protocol for Pydantic models used in MCP responses."""

    def model_dump(self) -> dict[str, object]:
        """Return a dictionary representation."""
        ...
