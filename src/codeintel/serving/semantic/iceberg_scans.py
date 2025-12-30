"""Iceberg scan helpers for serving engines."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyiceberg.expressions import (
    AlwaysTrue,
    And,
    BooleanExpression,
    EqualTo,
    GreaterThan,
    GreaterThanOrEqual,
    In,
    LessThan,
    LessThanOrEqual,
    NotEqualTo,
    StartsWith,
)

from codeintel.core.config.settings import IcebergSettings
from codeintel.core.iceberg.catalog import IcebergCatalogProvider
from codeintel.core.iceberg.scan_plan import IcebergScanPlan
from codeintel.serving.semantic.filter_ops import allowed_ops_for_column_type
from codeintel.serving.semantic.models import FilterSpec, FilterValue

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pyiceberg.table import DataScan, Table

    from codeintel.core.schemas.primitives import ColumnType
    from codeintel.serving.db.pointer import ServingSnapshotPointer

LOG = logging.getLogger(__name__)


class IcebergScanError(RuntimeError):
    """Raised when Iceberg scan setup fails."""


@dataclass(frozen=True, slots=True)
class IcebergFilterResult:
    """Result of translating filters into Iceberg expressions."""

    row_filter: BooleanExpression | None
    coverage: float | None
    supported: int
    total: int


@dataclass(frozen=True, slots=True)
class IcebergScanResult:
    """Resolved Iceberg scan configuration for serving reads."""

    scan: DataScan
    plan: IcebergScanPlan
    snapshot_id: int | None


@dataclass(frozen=True, slots=True)
class IcebergScanRequest:
    """Inputs required to build an Iceberg scan."""

    table_key: str
    columns: Sequence[str]
    filters: Sequence[FilterSpec]
    order_by: Sequence[str]
    column_types: Mapping[str, ColumnType] | None
    pointer: ServingSnapshotPointer
    settings: IcebergSettings
    batch_size: int | None = None


def iceberg_table_exists(*, settings: IcebergSettings, table_key: str) -> bool:
    """Return True when an Iceberg table exists for the table key.

    Returns
    -------
    bool
        True when the table exists and Iceberg reads are enabled.
    """
    if not settings.read_enabled:
        return False
    provider = IcebergCatalogProvider(settings)
    try:
        return provider.table_exists(table_key)
    except (RuntimeError, ValueError, KeyError, OSError) as exc:  # pragma: no cover
        LOG.warning("Iceberg table existence check failed: %s", exc)
        return False


def resolve_iceberg_ref(
    *, pointer: ServingSnapshotPointer, settings: IcebergSettings
) -> str | None:
    """Resolve the Iceberg snapshot ref for serving reads.

    Returns
    -------
    str | None
        Resolved Iceberg ref for the serving snapshot, or None when disabled.
    """
    if settings.read_ref:
        return settings.read_ref
    if not settings.read_enabled:
        return None
    if pointer.run_id:
        return f"run/{pointer.run_id}"
    if pointer.commit:
        return f"commit/{pointer.commit}"
    return "main"


def required_scan_fields(
    *,
    columns: Sequence[str],
    filters: Sequence[FilterSpec],
    order_by: Sequence[str],
) -> tuple[str, ...]:
    """Return the minimum column set needed to execute a query.

    Returns
    -------
    tuple[str, ...]
        Minimal set of columns needed for filters, ordering, and projection.
    """
    required: list[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        if value and value not in seen:
            seen.add(value)
            required.append(value)

    for col in columns:
        _add(col)
    for filt in filters:
        _add(filt.column)
    for item in order_by:
        _add(item[1:] if item.startswith("-") else item)
    return tuple(required)


def iceberg_row_filter_from_filters(
    *,
    filters: Sequence[FilterSpec],
    column_types: Mapping[str, ColumnType] | None,
) -> IcebergFilterResult:
    """Translate filter specs into an Iceberg row filter.

    Returns
    -------
    IcebergFilterResult
        Filter translation result with coverage and expression metadata.
    """
    if not filters:
        return IcebergFilterResult(row_filter=None, coverage=None, supported=0, total=0)
    expressions: list[BooleanExpression] = []
    supported = 0
    for filt in filters:
        expr = _iceberg_expression(
            filt=filt,
            column_types=column_types,
        )
        if expr is None:
            continue
        supported += 1
        expressions.append(expr)
    row_filter = _combine_expressions(expressions)
    coverage = supported / len(filters) if filters else None
    return IcebergFilterResult(
        row_filter=row_filter,
        coverage=coverage,
        supported=supported,
        total=len(filters),
    )


def iceberg_scan_for_query(*, request: IcebergScanRequest) -> IcebergScanResult:
    """Build an Iceberg scan for a serving query.

    Returns
    -------
    IcebergScanResult
        Resolved scan configuration and plan for the query.

    Raises
    ------
    IcebergScanError
        If the Iceberg table cannot be loaded.
    """
    provider = IcebergCatalogProvider(request.settings)
    try:
        table = provider.load_table(request.table_key)
    except (RuntimeError, ValueError, KeyError, OSError) as exc:  # pragma: no cover
        msg = f"Iceberg table load failed for {request.table_key}"
        raise IcebergScanError(msg) from exc

    ref = resolve_iceberg_ref(pointer=request.pointer, settings=request.settings)
    snapshot_id = _resolve_snapshot_id(table, ref=ref)
    filter_result = iceberg_row_filter_from_filters(
        filters=request.filters,
        column_types=request.column_types,
    )
    selected_fields = _selected_fields(
        required_scan_fields(
            columns=request.columns,
            filters=request.filters,
            order_by=request.order_by,
        )
    )
    options = dict(request.settings.io_options)
    plan = IcebergScanPlan(
        table_key=request.table_key,
        ref=ref,
        snapshot_id=snapshot_id,
        selected_fields=selected_fields,
        row_filter=filter_result.row_filter,
        case_sensitive=True,
        batch_size=request.batch_size,
        io_options=options or None,
        pushdown_coverage=filter_result.coverage,
    )

    row_filter = filter_result.row_filter or AlwaysTrue()
    if options:
        scan = table.scan(
            row_filter=row_filter,
            selected_fields=selected_fields,
            snapshot_id=snapshot_id,
            case_sensitive=True,
            options=options,
        )
    else:
        scan = table.scan(
            row_filter=row_filter,
            selected_fields=selected_fields,
            snapshot_id=snapshot_id,
            case_sensitive=True,
        )
    return IcebergScanResult(scan=scan, plan=plan, snapshot_id=snapshot_id)


def resolve_iceberg_snapshot_id(
    *,
    table_key: str,
    pointer: ServingSnapshotPointer,
    settings: IcebergSettings,
) -> int | None:
    """Resolve the snapshot id for a table and serving ref.

    Returns
    -------
    int | None
        Resolved snapshot id when available; otherwise None.
    """
    provider = IcebergCatalogProvider(settings)
    try:
        table = provider.load_table(table_key)
    except (RuntimeError, ValueError, KeyError, OSError) as exc:  # pragma: no cover
        LOG.warning("Iceberg snapshot lookup failed for %s: %s", table_key, exc)
        return None
    ref = resolve_iceberg_ref(pointer=pointer, settings=settings)
    return _resolve_snapshot_id(table, ref=ref)


def _resolve_snapshot_id(table: Table, *, ref: str | None) -> int | None:
    if ref:
        snapshot = table.snapshot_by_name(ref)
        if snapshot is not None:
            return snapshot.snapshot_id
        LOG.warning("Iceberg ref missing; falling back to current snapshot: %s", ref)
    current = table.current_snapshot()
    if current is None:
        return None
    return current.snapshot_id


def _selected_fields(fields: Sequence[str]) -> tuple[str, ...]:
    if not fields:
        return ("*",)
    if "*" in fields:
        return ("*",)
    return tuple(fields)


def _combine_expressions(
    expressions: Sequence[BooleanExpression],
) -> BooleanExpression | None:
    if not expressions:
        return None
    combined = expressions[0]
    for expr in expressions[1:]:
        combined = And(combined, expr)
    return combined


def _iceberg_expression(
    *,
    filt: FilterSpec,
    column_types: Mapping[str, ColumnType] | None,
) -> BooleanExpression | None:
    column_type = column_types.get(filt.column) if column_types is not None else None
    allowed_ops = allowed_ops_for_column_type(column_type)
    if filt.op not in allowed_ops:
        return None
    return _expression_for_filter(filt)


def _expression_for_filter(filt: FilterSpec) -> BooleanExpression | None:
    if filt.op == "contains":
        return None
    if filt.op == "startswith":
        return _startswith_expression(filt.column, filt.value)
    if filt.op == "in":
        return _in_expression(filt.column, filt.value)
    return _comparison_expression(filt.column, filt.op, filt.value)


def _comparison_expression(
    column: str,
    op: str,
    value: FilterValue,
) -> BooleanExpression | None:
    result: BooleanExpression | None = None
    if not isinstance(value, list):
        if isinstance(value, bool):
            result = _comparison_for_bool(column, op, value=value)
        elif isinstance(value, int) and not isinstance(value, bool):
            result = _comparison_for_int(column, op, value)
        elif isinstance(value, float):
            result = _comparison_for_float(column, op, value)
        elif isinstance(value, str):
            result = _comparison_for_str(column, op, value)
    return result


def _comparison_for_bool(
    term: str,
    op: str,
    *,
    value: bool,
) -> BooleanExpression | None:
    op_map: dict[str, Callable[[str, bool], BooleanExpression]] = {
        "eq": EqualTo,
        "ne": NotEqualTo,
        "lt": LessThan,
        "lte": LessThanOrEqual,
        "gt": GreaterThan,
        "gte": GreaterThanOrEqual,
    }
    handler = op_map.get(op)
    if handler is None:
        return None
    return handler(term, value)


def _comparison_for_int(
    term: str,
    op: str,
    value: int,
) -> BooleanExpression | None:
    op_map: dict[str, Callable[[str, int], BooleanExpression]] = {
        "eq": EqualTo,
        "ne": NotEqualTo,
        "lt": LessThan,
        "lte": LessThanOrEqual,
        "gt": GreaterThan,
        "gte": GreaterThanOrEqual,
    }
    handler = op_map.get(op)
    if handler is None:
        return None
    return handler(term, value)


def _comparison_for_float(
    term: str,
    op: str,
    value: float,
) -> BooleanExpression | None:
    op_map: dict[str, Callable[[str, float], BooleanExpression]] = {
        "eq": EqualTo,
        "ne": NotEqualTo,
        "lt": LessThan,
        "lte": LessThanOrEqual,
        "gt": GreaterThan,
        "gte": GreaterThanOrEqual,
    }
    handler = op_map.get(op)
    if handler is None:
        return None
    return handler(term, value)


def _comparison_for_str(
    term: str,
    op: str,
    value: str,
) -> BooleanExpression | None:
    op_map: dict[str, Callable[[str, str], BooleanExpression]] = {
        "eq": EqualTo,
        "ne": NotEqualTo,
        "lt": LessThan,
        "lte": LessThanOrEqual,
        "gt": GreaterThan,
        "gte": GreaterThanOrEqual,
    }
    handler = op_map.get(op)
    if handler is None:
        return None
    return handler(term, value)


def _in_expression(column: str, value: FilterValue) -> BooleanExpression | None:
    result: BooleanExpression | None = None
    if isinstance(value, list) and value:
        bool_values = [item for item in value if isinstance(item, bool)]
        if len(bool_values) == len(value):
            result = In(column, bool_values)
        else:
            int_values = [
                item for item in value if isinstance(item, int) and not isinstance(item, bool)
            ]
            if len(int_values) == len(value):
                result = In(column, int_values)
            else:
                float_values = [item for item in value if isinstance(item, float)]
                if len(float_values) == len(value):
                    result = In(column, float_values)
                else:
                    str_values = [item for item in value if isinstance(item, str)]
                    if len(str_values) == len(value):
                        result = In(column, str_values)
    return result


def _startswith_expression(column: str, value: FilterValue) -> BooleanExpression | None:
    if not isinstance(value, str):
        return None
    return StartsWith(column, value)


__all__ = [
    "IcebergFilterResult",
    "IcebergScanError",
    "IcebergScanRequest",
    "IcebergScanResult",
    "iceberg_row_filter_from_filters",
    "iceberg_scan_for_query",
    "iceberg_table_exists",
    "required_scan_fields",
    "resolve_iceberg_ref",
    "resolve_iceberg_snapshot_id",
]
