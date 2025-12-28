"""Backend-neutral query specs for semantic serving."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.serving.semantic.models import FilterSpec

if TYPE_CHECKING:
    from codeintel.core.schemas.primitives import ColumnType


@dataclass(frozen=True, slots=True)
class SemanticQuerySpec:
    """Normalized, backend-neutral semantic query specification."""

    view_id: str
    table_key: str
    allowed_columns: frozenset[str]
    columns: list[str]
    filters: list[FilterSpec]
    order_by: list[str]
    limit: int
    offset: int
    column_types: dict[str, ColumnType] | None = None


__all__ = ["SemanticQuerySpec"]
