"""Finalize gate helpers for Arrow table contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.build.schemas.service import get_schema_service
from codeintel.build.tabular.arrow_ops import AlignmentReport, align_table_to_contract
from codeintel.build.tabular.compute_helpers import call_compute, require_array
from codeintel.build.tabular.conversion import reader_to_table
from codeintel.build.tabular.dedupe_ops import dedupe_table_for_table
from codeintel.build.tabular.kernels import SortKey, stable_sort_indices
from codeintel.core.columnar.rows import empty_table_for_table

FinalizeMode = Literal["strict", "tolerant"]

_STRICT_ERROR_STAGES = frozenset({"schema", "invariant"})


@dataclass(frozen=True, slots=True)
class FinalizeDedupe:
    """Dedupe configuration for finalize gate."""

    prefer_columns: Sequence[str] = ()
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class FinalizeInvariant:
    """Invariant specification for finalize gate."""

    kind: Literal["list_alignment", "struct_required"]
    column: str
    related: tuple[str, ...]

    @classmethod
    def list_alignment(cls, column: str, related: Sequence[str]) -> FinalizeInvariant:
        """Require aligned list lengths for list columns.

        Parameters
        ----------
        column
            Primary list column.
        related
            Aligned list columns to match with the primary.

        Returns
        -------
        FinalizeInvariant
            List alignment invariant.
        """
        return cls(kind="list_alignment", column=column, related=tuple(related))

    @classmethod
    def struct_required(cls, column: str, fields: Sequence[str]) -> FinalizeInvariant:
        """Require non-null struct fields.

        Parameters
        ----------
        column
            Struct column name.
        fields
            Field names that must be non-null.

        Returns
        -------
        FinalizeInvariant
            Struct field invariant.
        """
        return cls(kind="struct_required", column=column, related=tuple(fields))


@dataclass(frozen=True, slots=True)
class FinalizeSpec:
    """Specification for finalize gate execution."""

    table_key: str
    mode: FinalizeMode
    required_non_null: Sequence[str] = ()
    invariants: Sequence[FinalizeInvariant] = ()
    key_fields: Sequence[str] = ()
    dedupe: FinalizeDedupe | None = None
    emit_artifacts: bool = False
    target_name: str | None = None


@dataclass(frozen=True, slots=True)
class FinalizeResult:
    """Finalize gate output."""

    good: pa.Table
    errors: pa.Table
    alignment: pa.Table
    stats: pa.Table


@dataclass(frozen=True, slots=True)
class FinalizeContext:
    """Context for finalize table evaluation."""

    table: pa.Table
    row_id: pa.Array


@dataclass(frozen=True, slots=True)
class ErrorSpec:
    """Error metadata for finalize error tables."""

    error_code: str
    stage: str
    column: str
    detail: str
    key_fields: Sequence[str] = ()


@dataclass(frozen=True, slots=True)
class JoinPrecheckReport:
    """Captured join precheck errors for persistence."""

    table_key: str | None
    target_name: str | None
    join_keys: tuple[str, ...]
    errors: pa.Table


_JOIN_PRECHECK_REPORTS: list[JoinPrecheckReport] = []


def record_join_precheck_errors(
    result: FinalizeResult,
    *,
    table_key: str | None,
    target_name: str | None,
    join_keys: Sequence[str],
) -> None:
    """Capture join precheck errors for persistence."""
    if result.errors.num_rows == 0:
        return
    report = JoinPrecheckReport(
        table_key=table_key,
        target_name=target_name,
        join_keys=tuple(join_keys),
        errors=result.errors,
    )
    _JOIN_PRECHECK_REPORTS.append(report)


def drain_join_precheck_reports() -> tuple[JoinPrecheckReport, ...]:
    """Return and clear stored join precheck diagnostics."""
    reports = tuple(_JOIN_PRECHECK_REPORTS)
    _JOIN_PRECHECK_REPORTS.clear()
    return reports


def finalize_table(
    table: pa.Table,
    *,
    spec: FinalizeSpec,
) -> FinalizeResult:
    """Finalize a table against its contract and invariants.

    Parameters
    ----------
    table
        Table to finalize.
    spec
        Finalize specification.

    Returns
    -------
    FinalizeResult
        Finalized result containing good rows, error rows, and artifacts.

    Raises
    ------
    ValueError
        If strict mode detects errors.
    """
    if table.num_rows == 0:
        return _empty_finalize_result(spec)

    aligned, report = _align_table(table, spec)
    context = FinalizeContext(table=aligned, row_id=pa.arange(0, aligned.num_rows))

    error_tables, masks = _collect_error_tables(context, spec)
    bad_mask = _combine_masks(masks)
    good = _filter_good_rows(aligned, bad_mask)
    good_context = _filtered_context(context, bad_mask)
    error_tables.append(_alignment_error_table(report, aligned, spec.key_fields))
    error_tables.append(_dedupe_error_table(good_context, spec))
    good = _apply_dedupe(good, spec)

    errors = _concat_errors(error_tables, aligned, key_fields=spec.key_fields)
    alignment, stats = _build_artifacts(report, errors, spec)

    if spec.mode == "strict":
        strict_errors = _strict_error_table(errors)
        if strict_errors.num_rows:
            msg = f"Finalize strict mode: {strict_errors.num_rows} invalid rows"
            raise ValueError(msg)

    return FinalizeResult(good=good, errors=errors, alignment=alignment, stats=stats)


def _empty_finalize_result(spec: FinalizeSpec) -> FinalizeResult:
    empty = empty_table_for_table(spec.table_key)
    return FinalizeResult(
        good=empty,
        errors=_empty_error_table(empty, key_fields=spec.key_fields),
        alignment=_empty_alignment_table(),
        stats=_empty_stats_table(),
    )


def _align_table(
    table: pa.Table,
    spec: FinalizeSpec,
) -> tuple[pa.Table, AlignmentReport | None]:
    report: AlignmentReport | None = None

    def _reporter(alignment_report: AlignmentReport) -> None:
        nonlocal report
        report = alignment_report

    aligned = align_table_to_contract(
        spec.table_key,
        table,
        reporter=_reporter,
        target_name=spec.target_name,
    )
    return aligned, report


def _collect_error_tables(
    context: FinalizeContext,
    spec: FinalizeSpec,
) -> tuple[list[pa.Table], list[pa.Array | pa.ChunkedArray]]:
    error_tables: list[pa.Table] = []
    masks: list[pa.Array | pa.ChunkedArray] = []

    required_mask = _required_non_null_mask(context.table, spec.required_non_null)
    if required_mask is not None:
        masks.append(required_mask)
        error_tables.append(
            _error_table_from_mask(
                context,
                mask=required_mask,
                spec=ErrorSpec(
                    error_code="NULL_REQUIRED_FIELD",
                    stage="schema",
                    column="required_non_null",
                    detail="missing required value",
                    key_fields=spec.key_fields,
                ),
            )
        )

    for invariant in spec.invariants:
        mask = _invariant_mask(context.table, invariant)
        if mask is None:
            continue
        masks.append(mask)
        error_tables.append(
            _error_table_from_mask(
                context,
                mask=mask,
                spec=ErrorSpec(
                    error_code=_invariant_error_code(invariant),
                    stage="invariant",
                    column=invariant.column,
                    detail=_invariant_detail(invariant),
                    key_fields=spec.key_fields,
                ),
            )
        )

    return error_tables, masks


def _filtered_context(
    context: FinalizeContext,
    bad_mask: pa.Array | pa.ChunkedArray | None,
) -> FinalizeContext:
    if bad_mask is None:
        return context
    good_mask = _invert(bad_mask)
    good_table = context.table.filter(good_mask)
    good_row_id = _filter_array(context.row_id, good_mask)
    return FinalizeContext(table=good_table, row_id=good_row_id)


def _filter_good_rows(
    table: pa.Table,
    bad_mask: pa.Array | pa.ChunkedArray | None,
) -> pa.Table:
    if bad_mask is None:
        return table
    good_mask = _invert(bad_mask)
    return table.filter(good_mask)


def _apply_dedupe(table: pa.Table, spec: FinalizeSpec) -> pa.Table:
    dedupe = spec.dedupe or FinalizeDedupe()
    if not dedupe.enabled:
        return table
    return dedupe_table_for_table(
        spec.table_key,
        table,
        prefer_columns=dedupe.prefer_columns,
    )


def _build_artifacts(
    report: AlignmentReport | None,
    errors: pa.Table,
    spec: FinalizeSpec,
) -> tuple[pa.Table, pa.Table]:
    if spec.emit_artifacts:
        return _alignment_table_from_report(report), _stats_table(errors)
    return _empty_alignment_table(), _empty_stats_table()


def _required_non_null_mask(
    table: pa.Table,
    required: Sequence[str],
) -> pa.Array | pa.ChunkedArray | None:
    if not required:
        return None
    mask: pa.Array | pa.ChunkedArray | None = None
    for name in required:
        if name not in table.column_names:
            continue
        valid = _fill_null_false(_is_valid(table[name]))
        invalid = _invert(valid)
        mask = invalid if mask is None else _or(mask, invalid)
    return mask


def _invariant_mask(
    table: pa.Table,
    invariant: FinalizeInvariant,
) -> pa.Array | pa.ChunkedArray | None:
    if invariant.column not in table.column_names:
        return None
    if invariant.kind == "list_alignment":
        base = _list_value_length(table[invariant.column])
        mismatch: pa.Array | pa.ChunkedArray | None = None
        for name in invariant.related:
            if name not in table.column_names:
                continue
            other = _list_value_length(table[name])
            equal = _fill_null_false(_equal(base, other))
            diff = _invert(equal)
            mismatch = diff if mismatch is None else _or(mismatch, diff)
        return mismatch
    if invariant.kind == "struct_required":
        struct_col = table[invariant.column]
        missing: pa.Array | pa.ChunkedArray | None = None
        for field_name in invariant.related:
            field_values = _struct_field(struct_col, field_name)
            valid = _fill_null_false(_is_valid(field_values))
            invalid = _invert(valid)
            missing = invalid if missing is None else _or(missing, invalid)
        return missing
    return None


def _invariant_error_code(invariant: FinalizeInvariant) -> str:
    if invariant.kind == "list_alignment":
        return "MISALIGNED_LIST_COLUMNS"
    if invariant.kind == "struct_required":
        return "NULL_REQUIRED_STRUCT_FIELD"
    return "INVARIANT_FAILED"


def _invariant_detail(invariant: FinalizeInvariant) -> str:
    if invariant.kind == "list_alignment":
        return "aligned list lengths differ"
    if invariant.kind == "struct_required":
        return "missing struct field"
    return "invariant failed"


def _combine_masks(
    masks: Sequence[pa.Array | pa.ChunkedArray],
) -> pa.Array | pa.ChunkedArray | None:
    combined: pa.Array | pa.ChunkedArray | None = None
    for mask in masks:
        combined = mask if combined is None else _or(combined, mask)
    return combined


def _fill_null_false(mask: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    return _fill_null(mask, fill_value=False)


def _error_table_from_mask(
    context: FinalizeContext,
    *,
    mask: pa.Array | pa.ChunkedArray,
    spec: ErrorSpec,
) -> pa.Table:
    indices = _indices_nonzero(mask)
    if len(indices) == 0:
        return _empty_error_table(context.table, spec.key_fields)
    columns = _error_columns(
        row_id=context.row_id,
        indices=indices,
        spec=spec,
    )
    for name in spec.key_fields:
        if name in context.table.column_names:
            columns[name] = pc.take(context.table[name], indices)
    return pa.table(columns)


def _error_table_from_indices(
    table: pa.Table,
    *,
    row_id: pa.Array | pa.ChunkedArray,
    indices: pa.Array | pa.ChunkedArray,
    spec: ErrorSpec,
) -> pa.Table:
    if len(indices) == 0:
        return _empty_error_table(table, spec.key_fields)
    columns = _error_columns(
        row_id=row_id,
        indices=indices,
        spec=spec,
    )
    for name in spec.key_fields:
        if name in table.column_names:
            columns[name] = pc.take(table[name], indices)
    return pa.table(columns)


def _error_columns(
    *,
    row_id: pa.Array | pa.ChunkedArray,
    indices: pa.Array | pa.ChunkedArray,
    spec: ErrorSpec,
) -> dict[str, pa.Array | pa.ChunkedArray]:
    count = len(indices)
    return {
        "row_id": pc.take(row_id, indices),
        "error_code": pa.array([spec.error_code] * count, type=pa.string()),
        "stage": pa.array([spec.stage] * count, type=pa.string()),
        "column": pa.array([spec.column] * count, type=pa.string()),
        "detail": pa.array([spec.detail] * count, type=pa.string()),
    }


def _empty_error_table(table: pa.Table, key_fields: Sequence[str]) -> pa.Table:
    fields = [
        pa.field("row_id", pa.int64()),
        pa.field("error_code", pa.string()),
        pa.field("stage", pa.string()),
        pa.field("column", pa.string()),
        pa.field("detail", pa.string()),
    ]
    fields.extend(
        [
            table.schema.field(name)
            for name in key_fields
            if name in table.column_names
        ]
    )
    return pa.Table.from_batches([], schema=pa.schema(fields))


def _concat_errors(
    errors: Sequence[pa.Table],
    table: pa.Table,
    *,
    key_fields: Sequence[str],
) -> pa.Table:
    non_empty = [err for err in errors if err.num_rows > 0]
    if not non_empty:
        return _empty_error_table(table, key_fields=key_fields)
    return pa.concat_tables(non_empty, promote_options="default")


def _alignment_error_table(
    report: AlignmentReport | None,
    table: pa.Table,
    key_fields: Sequence[str],
) -> pa.Table:
    if report is None:
        return _empty_error_table(table, key_fields=key_fields)
    issues: list[tuple[str, str, str]] = []
    issues.extend(("MISSING_COLUMN", name, "missing column") for name in report.missing_columns)
    issues.extend(("EXTRA_COLUMN", name, "extra column") for name in report.extra_columns)
    issues.extend(
        ("COERCED_COLUMN", name, "column type coerced") for name in report.coerced_columns
    )
    if not issues:
        return _empty_error_table(table, key_fields=key_fields)
    count = len(issues)
    row_id = pa.nulls(count, type=pa.int64())
    error_code = pa.array([issue[0] for issue in issues], type=pa.string())
    column = pa.array([issue[1] for issue in issues], type=pa.string())
    detail = pa.array([issue[2] for issue in issues], type=pa.string())
    columns: dict[str, pa.Array | pa.ChunkedArray] = {
        "row_id": row_id,
        "error_code": error_code,
        "stage": pa.array(["schema_alignment"] * count, type=pa.string()),
        "column": column,
        "detail": detail,
    }
    for name in key_fields:
        if name in table.column_names:
            columns[name] = pa.nulls(count, type=table.schema.field(name).type)
    return pa.table(columns)


def _dedupe_error_table(context: FinalizeContext, spec: FinalizeSpec) -> pa.Table:
    dedupe = spec.dedupe or FinalizeDedupe()
    if not dedupe.enabled:
        return _empty_error_table(context.table, key_fields=spec.key_fields)
    key_columns = _primary_key_columns(spec.table_key)
    available = set(context.table.column_names)
    present_keys = [name for name in key_columns if name in available]
    if not present_keys:
        return _empty_error_table(context.table, key_fields=spec.key_fields)
    sort_keys = _dedupe_sort_keys(
        present_keys,
        [name for name in dedupe.prefer_columns if name in available],
    )
    try:
        indexed = context.table.append_column("_row_id", context.row_id)
        sorted_indexed = indexed.take(
            stable_sort_indices(indexed, sort_keys=sort_keys, null_placement="at_end")
        )
        sort_rank = pa.array(range(sorted_indexed.num_rows), type=pa.int64())
        sorted_indexed = sorted_indexed.append_column("_sort_rank", sort_rank)
        grouped = sorted_indexed.group_by(present_keys).aggregate([("_sort_rank", "min")])
        min_column = "_sort_rank_min"
        if min_column not in grouped.column_names:
            return _empty_error_table(context.table, key_fields=spec.key_fields)
        min_ranks = grouped[min_column]
        keep_mask = _fill_null_false(_is_in(sort_rank, min_ranks))
        duplicate_mask = _invert(keep_mask)
        duplicate_indices = _indices_nonzero(duplicate_mask)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError, ValueError):
        return _empty_error_table(context.table, key_fields=spec.key_fields)
    dedupe_key_fields = spec.key_fields or present_keys
    return _error_table_from_indices(
        sorted_indexed,
        row_id=sorted_indexed["_row_id"],
        indices=duplicate_indices,
        spec=ErrorSpec(
            error_code="DUPLICATE_PRIMARY_KEY",
            stage="dedupe",
            column="primary_key",
            detail="duplicate primary key row dropped",
            key_fields=dedupe_key_fields,
        ),
    )


def _alignment_table_from_report(report: AlignmentReport | None) -> pa.Table:
    if report is None:
        return _empty_alignment_table()
    return pa.table(
        {
            "table_key": pa.array([report.table_key], type=pa.string()),
            "target_name": pa.array([report.target_name], type=pa.string()),
            "row_count": pa.array([report.row_count], type=pa.int64()),
            "missing_columns": pa.array(
                [list(report.missing_columns)],
                type=pa.list_(pa.string()),
            ),
            "extra_columns": pa.array(
                [list(report.extra_columns)],
                type=pa.list_(pa.string()),
            ),
            "coerced_columns": pa.array(
                [list(report.coerced_columns)],
                type=pa.list_(pa.string()),
            ),
        }
    )


def _empty_alignment_table() -> pa.Table:
    schema = pa.schema(
        [
            pa.field("table_key", pa.string()),
            pa.field("target_name", pa.string()),
            pa.field("row_count", pa.int64()),
            pa.field("missing_columns", pa.list_(pa.string())),
            pa.field("extra_columns", pa.list_(pa.string())),
            pa.field("coerced_columns", pa.list_(pa.string())),
        ]
    )
    return pa.Table.from_batches([], schema=schema)


def _stats_table(errors: pa.Table) -> pa.Table:
    if errors.num_rows == 0:
        return _empty_stats_table()
    grouped = errors.group_by(["error_code"]).aggregate([("row_id", "count")])
    return grouped.rename_columns(["error_code", "count"])


def _empty_stats_table() -> pa.Table:
    schema = pa.schema(
        [
            pa.field("error_code", pa.string()),
            pa.field("count", pa.int64()),
        ]
    )
    return pa.Table.from_batches([], schema=schema)


def _primary_key_columns(table_key: str) -> tuple[str, ...]:
    try:
        schema = get_schema_service().get_table_schema(table_key)
    except (KeyError, RuntimeError, TypeError):
        return ()
    if schema is None or not schema.primary_key:
        return ()
    return tuple(schema.primary_key)


def _dedupe_sort_keys(
    key_columns: Sequence[str],
    prefer_columns: Sequence[str],
) -> list[SortKey]:
    sort_keys: list[SortKey] = [(name, "ascending") for name in key_columns]
    sort_keys.extend((name, "descending") for name in prefer_columns if name not in key_columns)
    return sort_keys


def _strict_error_table(errors: pa.Table) -> pa.Table:
    if errors.num_rows == 0 or "stage" not in errors.column_names:
        return errors
    allowed = pa.array(sorted(_STRICT_ERROR_STAGES), type=pa.string())
    mask = _fill_null_false(_is_in(errors["stage"], allowed))
    return errors.filter(mask)


def _compute_array(name: str, args: Sequence[object]) -> pa.Array | pa.ChunkedArray:
    return require_array(call_compute(name, list(args)), name=name)


def _filter_array(
    values: pa.Array | pa.ChunkedArray,
    mask: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    return _compute_array("filter", [values, mask])


def _is_valid(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    return _compute_array("is_valid", [values])


def _is_in(
    values: pa.Array | pa.ChunkedArray,
    value_set: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    return _compute_array("is_in", [values, value_set])


def _list_value_length(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    return _compute_array("list_value_length", [values])


def _struct_field(
    values: pa.Array | pa.ChunkedArray,
    field_name: str,
) -> pa.Array | pa.ChunkedArray:
    return _compute_array("struct_field", [values, field_name])


def _equal(
    left: pa.Array | pa.ChunkedArray,
    right: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    return _compute_array("equal", [left, right])


def _invert(mask: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    return _compute_array("invert", [mask])


def _or(
    left: pa.Array | pa.ChunkedArray,
    right: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    return _compute_array("or", [left, right])


def _indices_nonzero(mask: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    return _compute_array("indices_nonzero", [mask])


def _fill_null(
    mask: pa.Array | pa.ChunkedArray,
    *,
    fill_value: bool,
) -> pa.Array | pa.ChunkedArray:
    return _compute_array("fill_null", [mask, fill_value])


def finalize_join_keys(
    table: pa.Table,
    *,
    required_non_null: Sequence[str],
    key_fields: Sequence[str] = (),
    stage: str = "schema",
) -> FinalizeResult:
    """Finalize a table by validating required join keys without a contract schema.

    Parameters
    ----------
    table
        Table to finalize.
    required_non_null
        Join key columns that must be non-null.
    key_fields
        Columns to copy into the error table for context.
    stage
        Error stage label for join-key failures.

    Returns
    -------
    FinalizeResult
        Finalize result with filtered good rows and error rows.
    """
    if table.num_rows == 0:
        empty_errors = _empty_error_table(table, key_fields=key_fields)
        return FinalizeResult(
            good=table,
            errors=empty_errors,
            alignment=_empty_alignment_table(),
            stats=_empty_stats_table(),
        )
    context = FinalizeContext(table=table, row_id=pa.arange(0, table.num_rows))
    required_mask = _required_non_null_mask(table, required_non_null)
    error_tables: list[pa.Table] = []
    if required_mask is not None:
        error_tables.append(
            _error_table_from_mask(
                context,
                mask=required_mask,
                spec=ErrorSpec(
                    error_code="NULL_REQUIRED_FIELD",
                    stage=stage,
                    column="required_non_null",
                    detail="missing required value",
                    key_fields=key_fields,
                ),
            )
        )
    bad_mask = required_mask
    good = _filter_good_rows(table, bad_mask)
    errors = _concat_errors(error_tables, table, key_fields=key_fields)
    stats = _stats_table(errors)
    return FinalizeResult(
        good=good,
        errors=errors,
        alignment=_empty_alignment_table(),
        stats=stats,
    )


def finalize_reader(
    reader: pa.RecordBatchReader,
    *,
    spec: FinalizeSpec,
) -> FinalizeResult:
    """Finalize an Arrow reader against its contract and invariants.

    Parameters
    ----------
    reader
        RecordBatchReader to finalize.
    spec
        Finalize specification.

    Returns
    -------
    FinalizeResult
        Finalized result containing good rows, error rows, and artifacts.
    """
    table = reader_to_table(reader)
    return finalize_table(table, spec=spec)


__all__ = [
    "FinalizeDedupe",
    "FinalizeInvariant",
    "FinalizeResult",
    "FinalizeSpec",
    "finalize_join_keys",
    "finalize_reader",
    "finalize_table",
]
