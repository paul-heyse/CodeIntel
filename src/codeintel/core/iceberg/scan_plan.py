"""Scan planning payloads for Iceberg reads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pyiceberg.expressions import BooleanExpression


@dataclass(frozen=True, slots=True)
class IcebergScanPlan:
    """Scan planning configuration for Iceberg reads."""

    table_key: str
    ref: str | None = None
    snapshot_id: int | None = None
    selected_fields: Sequence[str] = ()
    row_filter: BooleanExpression | None = None
    case_sensitive: bool = True
    batch_size: int | None = None
    limit: int | None = None
    io_options: Mapping[str, str] | None = None
    pushdown_coverage: float | None = None


__all__ = ["IcebergScanPlan"]
