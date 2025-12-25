"""Shared upsert specification for storage writers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlglot import exp


@dataclass(frozen=True, slots=True)
class UpsertSpec:
    """Conflict handling specification for upsert operations."""

    conflict_columns: Sequence[str]
    update_columns: Sequence[str] | None = None
    update_condition: exp.Expression | None = None


__all__ = ["UpsertSpec"]
