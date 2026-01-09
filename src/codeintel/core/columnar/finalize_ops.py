"""Finalize gate helpers for Arrow table contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.core.columnar.compute_helpers import call_compute, require_array
from codeintel.core.columnar.dedupe_ops import DedupeDeterminism, dedupe_table_for_table
from codeintel.core.columnar.schema_alignment import (
    align_table_to_contract,
    extras_policy_from_schema,
)
from codeintel.core.columnar.table_utils import empty_table_for_table
from codeintel.core.columnar.type_normalization import (
    binary_view_cast_type,
    string_view_cast_type,
)
from codeintel.core.schemas.service import get_schema_service
from codeintel.core.validation.schema_constraints import (
    is_list_like,
    list_alignment_specs_for_table_key,
)

FinalizeMode = Literal["strict", "tolerant"]
NullListPolicy = Literal["error", "empty"]


@dataclass(frozen=True, slots=True)
class FinalizeDedupe:
    """Dedupe configuration for finalize gate."""

    prefer_columns: Sequence[str] = ()
    enabled: bool = True
    determinism: DedupeDeterminism = "best_effort"
    tie_breaker_columns: Sequence[str] = ()


@dataclass(frozen=True, slots=True)
class FinalizeInvariant:
    """Invariant specification for finalize gate."""

    kind: Literal["list_alignment", "struct_required"]
    column: str
    related: tuple[str, ...]

    @classmethod
    def list_alignment(cls, column: str, related: Sequence[str]) -> FinalizeInvariant:
        """Require aligned list lengths for list columns.

        Returns
        -------
        FinalizeInvariant
            List alignment invariant.
        """
        return cls(kind="list_alignment", column=column, related=tuple(related))

    @classmethod
    def struct_required(cls, column: str, fields: Sequence[str]) -> FinalizeInvariant:
        """Require non-null struct fields.

        Returns
        -------
        FinalizeInvariant
            Struct field invariant.
        """
        return cls(kind="struct_required", column=column, related=tuple(fields))


@dataclass(frozen=True, slots=True)
class FinalizeListPolicy:
    """List-specific null handling policy."""

    column: str
    null_policy: NullListPolicy = "error"


@dataclass(frozen=True, slots=True)
class FinalizeSpec:
    """Specification for finalize gate execution."""

    table_key: str
    mode: FinalizeMode
    required_non_null: Sequence[str] = ()
    invariants: Sequence[FinalizeInvariant] = ()
    list_policies: Sequence[FinalizeListPolicy] = ()
    key_fields: Sequence[str] = ()
    context_fields: Sequence[str] = ()
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
    context_columns: Mapping[str, pa.Array | pa.ChunkedArray] = field(default_factory=dict)
    context_fields: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ErrorSpec:
    """Error metadata for finalize error tables."""

    error_code: str
    stage: str
    column: str
    detail: str
    key_fields: Sequence[str] = ()
    context_fields: Sequence[str] = ()


@dataclass(frozen=True, slots=True)
class AlignmentReport:
    """Alignment report for finalize artifacts."""

    table_key: str
    target_name: str | None
    row_count: int
    missing_columns: tuple[str, ...]
    extra_columns: tuple[str, ...]
    coerced_columns: tuple[str, ...]


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

    context_fields = _resolve_context_fields(spec)
    context_columns = _context_column_map(table, context_fields)
    aligned, report = _align_table(table, spec, context_fields=context_fields)
    if aligned is None:
        return _missing_contract_result(table, spec, context_columns)

    resolved_context_fields = _filter_context_fields(
        aligned,
        context_columns=context_columns,
        fields=context_fields,
    )
    context = FinalizeContext(
        table=aligned,
        row_id=pa.arange(0, aligned.num_rows),
        context_columns=context_columns,
        context_fields=resolved_context_fields,
    )
    invariants = _resolve_invariants(spec)
    error_tables, masks = _collect_error_tables(context, spec, invariants=invariants)
    bad_mask = _combine_masks(masks)
    good = _filter_good_rows(aligned, bad_mask)
    good = _apply_dedupe(good, spec)

    errors = _concat_errors(
        error_tables,
        aligned,
        key_fields=spec.key_fields,
        context_fields=context.context_fields,
        context_columns=context.context_columns,
    )
    alignment, stats = _build_artifacts(report, errors, spec)

    if spec.mode == "strict" and errors.num_rows:
        msg = f"Finalize strict mode: {errors.num_rows} invalid rows"
        raise ValueError(msg)

    return FinalizeResult(good=good, errors=errors, alignment=alignment, stats=stats)


def _empty_finalize_result(spec: FinalizeSpec) -> FinalizeResult:
    empty = empty_table_for_table(spec.table_key)
    context_fields = _filter_context_fields(
        empty,
        context_columns={},
        fields=_resolve_context_fields(spec),
    )
    return FinalizeResult(
        good=empty,
        errors=_empty_error_table(
            empty,
            key_fields=spec.key_fields,
            context_fields=context_fields,
            context_columns={},
        ),
        alignment=_empty_alignment_table(),
        stats=_empty_stats_table(),
    )


def _missing_contract_result(
    table: pa.Table,
    spec: FinalizeSpec,
    context_columns: Mapping[str, pa.Array | pa.ChunkedArray],
) -> FinalizeResult:
    errors = _error_table_for_missing_contract(table, spec, context_columns=context_columns)
    alignment = _empty_alignment_table()
    stats = _stats_table(errors)
    if spec.mode == "strict":
        msg = f"Missing contract schema for {spec.table_key}"
        raise ValueError(msg)
    return FinalizeResult(good=table, errors=errors, alignment=alignment, stats=stats)


def _error_table_for_missing_contract(
    table: pa.Table,
    spec: FinalizeSpec,
    *,
    context_columns: Mapping[str, pa.Array | pa.ChunkedArray],
) -> pa.Table:
    if table.num_rows == 0:
        return _empty_error_table(
            table,
            key_fields=spec.key_fields,
            context_fields=_filter_context_fields(
                table,
                context_columns=context_columns,
                fields=_resolve_context_fields(spec),
            ),
            context_columns=context_columns,
        )
    row_id = pa.arange(0, table.num_rows)
    indices = pa.array(range(table.num_rows), type=pa.int64())
    columns = _error_columns(
        row_id=row_id,
        indices=indices,
        spec=ErrorSpec(
            error_code="MISSING_CONTRACT_SCHEMA",
            stage="alignment",
            column="table_key",
            detail=spec.table_key,
            key_fields=spec.key_fields,
            context_fields=spec.context_fields,
        ),
    )
    _add_context_columns(
        columns,
        context=context_columns,
        table=table,
        fields=_filter_context_fields(
            table,
            context_columns=context_columns,
            fields=_resolve_context_fields(spec),
        ),
        indices=indices,
    )
    return pa.table(columns)


def _align_table(
    table: pa.Table,
    spec: FinalizeSpec,
    *,
    context_fields: Sequence[str],
) -> tuple[pa.Table | None, AlignmentReport | None]:
    schema_service = get_schema_service()
    contract_schema = schema_service.get_arrow_schema(spec.table_key)
    if contract_schema is None:
        return None, None
    table_for_alignment = _drop_context_columns(table, contract_schema, context_fields)
    report = _alignment_report(
        contract_schema=contract_schema,
        incoming_schema=table_for_alignment.schema,
        table_key=spec.table_key,
        target_name=spec.target_name,
        row_count=table.num_rows,
    )
    aligned = align_table_to_contract(
        table_for_alignment,
        contract_schema,
        extras_policy=extras_policy_from_schema(contract_schema),
    )
    return aligned, report


def _alignment_report(
    *,
    contract_schema: pa.Schema,
    incoming_schema: pa.Schema,
    table_key: str,
    target_name: str | None,
    row_count: int,
) -> AlignmentReport:
    contract_fields = {field.name: field.type for field in contract_schema}
    incoming_fields = {field.name: field.type for field in incoming_schema}
    missing = tuple(name for name in contract_fields if name not in incoming_fields)
    extra = tuple(name for name in incoming_fields if name not in contract_fields)
    coerced: list[str] = []
    for name, contract_type in contract_fields.items():
        incoming_type = incoming_fields.get(name)
        if incoming_type is None:
            continue
        normalized_incoming = _normalize_type(incoming_type)
        normalized_contract = _normalize_type(contract_type)
        if not normalized_incoming.equals(normalized_contract):
            coerced.append(name)
    return AlignmentReport(
        table_key=table_key,
        target_name=target_name,
        row_count=row_count,
        missing_columns=missing,
        extra_columns=extra,
        coerced_columns=tuple(coerced),
    )


def _normalize_type(data_type: pa.DataType) -> pa.DataType:
    normalized = string_view_cast_type(data_type)
    return binary_view_cast_type(normalized)


def _resolve_context_fields(spec: FinalizeSpec) -> tuple[str, ...]:
    return _dedupe_field_names((*spec.key_fields, *spec.context_fields))


def _filter_context_fields(
    table: pa.Table,
    *,
    context_columns: Mapping[str, pa.Array | pa.ChunkedArray],
    fields: Sequence[str],
) -> tuple[str, ...]:
    resolved: list[str] = []
    for name in _dedupe_field_names(fields):
        if name in table.column_names or name in context_columns:
            resolved.append(name)
    return tuple(resolved)


def _dedupe_field_names(fields: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for name in fields:
        if not name or name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return tuple(ordered)


def _context_column_map(
    table: pa.Table,
    fields: Sequence[str],
) -> dict[str, pa.Array | pa.ChunkedArray]:
    return {name: table[name] for name in fields if name in table.column_names}


def _add_context_columns(
    columns: dict[str, pa.Array | pa.ChunkedArray],
    *,
    context: Mapping[str, pa.Array | pa.ChunkedArray],
    table: pa.Table,
    fields: Sequence[str],
    indices: pa.Array | pa.ChunkedArray,
) -> None:
    for name in fields:
        if name in table.column_names:
            columns[name] = pc.take(table[name], indices)
        elif name in context:
            columns[name] = pc.take(context[name], indices)


def _drop_context_columns(
    table: pa.Table,
    contract_schema: pa.Schema,
    context_fields: Sequence[str],
) -> pa.Table:
    drop_names = [
        name
        for name in context_fields
        if name in table.column_names and name not in contract_schema.names
    ]
    if not drop_names:
        return table
    try:
        return table.drop(drop_names)
    except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError, ValueError):
        return table


def _resolve_invariants(spec: FinalizeSpec) -> tuple[FinalizeInvariant, ...]:
    invariants = list(spec.invariants)
    seen = {(inv.kind, inv.column, inv.related) for inv in invariants}
    for alignment in list_alignment_specs_for_table_key(spec.table_key):
        invariant = FinalizeInvariant.list_alignment(alignment.column, alignment.related)
        key = (invariant.kind, invariant.column, invariant.related)
        if key in seen:
            continue
        invariants.append(invariant)
        seen.add(key)
    return tuple(invariants)


def _list_policy_errors(
    context: FinalizeContext,
    spec: FinalizeSpec,
) -> tuple[list[pa.Table], list[pa.Array | pa.ChunkedArray]]:
    error_tables: list[pa.Table] = []
    masks: list[pa.Array | pa.ChunkedArray] = []
    for policy in spec.list_policies:
        if policy.null_policy == "empty":
            continue
        if policy.column not in context.table.column_names:
            continue
        values = context.table[policy.column]
        if not is_list_like(values.type):
            continue
        null_mask = _list_null_mask(values)
        if null_mask is None:
            continue
        masks.append(null_mask)
        error_tables.append(
            _error_table_from_mask(
                context,
                mask=null_mask,
                spec=ErrorSpec(
                    error_code="NULL_REQUIRED_LIST",
                    stage="invariant",
                    column=policy.column,
                    detail="null list not allowed",
                    key_fields=spec.key_fields,
                    context_fields=spec.context_fields,
                ),
            )
        )
    return error_tables, masks


def _list_null_mask(
    values: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray | None:
    try:
        nulls = _is_null(values)
    except (TypeError, pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError):
        return None
    return _fill_null_false(nulls)


def _collect_error_tables(
    context: FinalizeContext,
    spec: FinalizeSpec,
    *,
    invariants: Sequence[FinalizeInvariant],
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
                    stage="invariant",
                    column="required_non_null",
                    detail="missing required value",
                    key_fields=spec.key_fields,
                    context_fields=spec.context_fields,
                ),
            )
        )

    list_policy_tables, list_policy_masks = _list_policy_errors(context, spec)
    error_tables.extend(list_policy_tables)
    masks.extend(list_policy_masks)

    for invariant in invariants:
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
                    context_fields=spec.context_fields,
                ),
            )
        )

    return error_tables, masks


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
        determinism=dedupe.determinism,
        tie_breaker_columns=dedupe.tie_breaker_columns,
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
        return _empty_error_table(
            context.table,
            key_fields=spec.key_fields,
            context_fields=context.context_fields,
            context_columns=context.context_columns,
        )
    columns = _error_columns(
        row_id=context.row_id,
        indices=indices,
        spec=spec,
    )
    _add_context_columns(
        columns,
        context=context.context_columns,
        table=context.table,
        fields=context.context_fields,
        indices=indices,
    )
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


def _empty_error_table(
    table: pa.Table,
    *,
    key_fields: Sequence[str],
    context_fields: Sequence[str],
    context_columns: Mapping[str, pa.Array | pa.ChunkedArray],
) -> pa.Table:
    fields = [
        pa.field("row_id", pa.int64()),
        pa.field("error_code", pa.string()),
        pa.field("stage", pa.string()),
        pa.field("column", pa.string()),
        pa.field("detail", pa.string()),
    ]
    for name in _dedupe_field_names((*key_fields, *context_fields)):
        if name in table.column_names:
            fields.append(table.schema.field(name))
        elif name in context_columns:
            fields.append(pa.field(name, context_columns[name].type))
    return pa.Table.from_batches([], schema=pa.schema(fields))


def _concat_errors(
    errors: Sequence[pa.Table],
    table: pa.Table,
    *,
    key_fields: Sequence[str],
    context_fields: Sequence[str],
    context_columns: Mapping[str, pa.Array | pa.ChunkedArray],
) -> pa.Table:
    non_empty = [err for err in errors if err.num_rows > 0]
    if not non_empty:
        return _empty_error_table(
            table,
            key_fields=key_fields,
            context_fields=context_fields,
            context_columns=context_columns,
        )
    return pa.concat_tables(non_empty, promote_options="default")


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


def _compute_array(name: str, args: Sequence[object]) -> pa.Array | pa.ChunkedArray:
    return require_array(call_compute(name, list(args)), name=name)


def _is_valid(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    return _compute_array("is_valid", [values])


def _is_null(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    return _compute_array("is_null", [values])


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


__all__ = [
    "FinalizeDedupe",
    "FinalizeInvariant",
    "FinalizeListPolicy",
    "FinalizeResult",
    "FinalizeSpec",
    "NullListPolicy",
    "finalize_table",
]
