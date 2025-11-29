"""Shared typed protocols to reduce use of ``Any`` in serving layer."""

from __future__ import annotations

from typing import Protocol

from codeintel.core.types import (
    PytestCallEntry,
    PytestTestEntry,
    ScipDocument,
    ScipOccurrence,
    ScipRange,
)


class HasModelDump(Protocol):
    """Protocol for Pydantic models used in MCP responses."""

    def model_dump(self) -> dict[str, object]:
        """Return a dictionary representation."""
        ...


__all__ = [
    "HasModelDump",
    "PytestCallEntry",
    "PytestTestEntry",
    "ScipDocument",
    "ScipOccurrence",
    "ScipRange",
]
