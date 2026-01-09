"""Finalize gate helpers for Arrow table contracts."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.core.columnar.compute_helpers import call_compute, require_array
from codeintel.core.columnar.conversion import reader_to_table, record_batch_reader_from_iterable
from codeintel.core.columnar.dedupe_ops import (
    DedupeDeterminism,
    DedupeLegacy,
    DedupeSpec,
    DedupeStrategy,
    DedupeTier,
    dedupe_table_for_table,
    normalize_dedupe_tier,
)
from codeintel.core.columnar.kernels import (
    SortKey,
    list_value_length,
    stable_sort_table,
    struct_field,
)
from codeintel.core.columnar.nested_ops import (
    deep_cast_table_to_contract,
    unify_schemas_with_contract_first,
)
from codeintel.core.columnar.queryspec import PROVENANCE_FIELDS
from codeintel.core.columnar.readers import empty_reader_from_schema
from codeintel.core.columnar.schema_alignment import (
    align_table_to_contract,
    extras_policy_from_schema,
)
from codeintel.core.columnar.schema_ops import (
    DEFAULT_SCHEMA_PROMOTE_OPTIONS,
    SchemaPromoteOptions,
)
from codeintel.core.columnar.table_utils import empty_table_for_table
from codeintel.core.columnar.type_normalization import (
    binary_view_cast_type,
    string_view_cast_type,
)
from codeintel.core.schemas.primitives import (
    FinalizeDedupeSpec,
    FinalizeInvariantSpec,
    FinalizeListPolicySpec,
    FinalizePolicy,
    TableSchema,
    resolve_stable_sort_keys,
)
from codeintel.core.schemas.service import get_schema_service
from codeintel.core.validation.schema_constraints import (
    is_list_like,
    list_alignment_specs_for_table_key,
)

FinalizeMode = Literal["strict", "tolerant"]
NullListPolicy = Literal["error", "empty"]

ERROR_CODE_NULL_REQUIRED_LIST = "NULL_REQUIRED_LIST"
ERROR_CODE_MISALIGNED_LIST_COLUMNS = "MISALIGNED_LIST_COLUMNS"
_PROVENANCE_CONTEXT_FIELDS: tuple[str, ...] = tuple(
    output_name for output_name, _source_name in PROVENANCE_FIELDS
)


@dataclass(frozen=True, slots=True)
class FinalizeDedupe:
    """Dedupe configuration for finalize gate."""

    prefer_columns: Sequence[str] = ()
    enabled: bool = True
    determinism: DedupeDeterminism = "best_effort"
    tie_breaker_columns: Sequence[str] = ()
    keys: Sequence[str] | None = None
    tie_breakers: Sequence[SortKey] | None = None
    tier: DedupeTier | None = None
    strategy: DedupeStrategy | None = None


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
    schema_promote_options: SchemaPromoteOptions = DEFAULT_SCHEMA_PROMOTE_OPTIONS
    required_non_null: Sequence[str] = ()
    invariants: Sequence[FinalizeInvariant] = ()
    list_policies: Sequence[FinalizeListPolicy] = ()
    key_fields: Sequence[str] = ()
    context_fields: Sequence[str] = ()
    dedupe: FinalizeDedupe | None = None
    order_by: Sequence[SortKey] = ()
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
class _AlignedContext:
    aligned: pa.Table
    report: AlignmentReport | None
    context_columns: Mapping[str, pa.Array | pa.ChunkedArray]
    context_fields: tuple[str, ...]


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
    """Return and clear stored join precheck diagnostics.

    Returns
    -------
    tuple[JoinPrecheckReport, ...]
        Collected join precheck reports.
    """
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
    resolved_spec = _resolve_finalize_spec(spec)
    if table.num_rows == 0:
        return _empty_finalize_result(resolved_spec)

    aligned_context, failure = _prepare_alignment(table, resolved_spec)
    if failure is not None:
        return failure
    if aligned_context is None:
        return _missing_contract_result(table, resolved_spec, context_columns={})
    context = FinalizeContext(
        table=aligned_context.aligned,
        row_id=pa.arange(0, aligned_context.aligned.num_rows),
        context_columns=aligned_context.context_columns,
        context_fields=aligned_context.context_fields,
    )
    invariants = _resolve_invariants(resolved_spec)
    error_tables, masks = _collect_error_tables(context, resolved_spec, invariants=invariants)
    bad_mask = _combine_masks(masks)
    good = _filter_good_rows(context.table, bad_mask)
    dedupe_spec = _dedupe_spec_from_finalize(resolved_spec)
    good = _apply_dedupe(good, resolved_spec, dedupe_spec=dedupe_spec)
    good = _apply_order_by(good, resolved_spec, dedupe_spec=dedupe_spec)

    errors = _concat_errors(
        error_tables,
        context.table,
        key_fields=resolved_spec.key_fields,
        context_fields=context.context_fields,
        context_columns=context.context_columns,
    )
    alignment, stats = _build_artifacts(aligned_context.report, errors, resolved_spec)

    if resolved_spec.mode == "strict" and errors.num_rows:
        msg = f"Finalize strict mode: {errors.num_rows} invalid rows"
        raise ValueError(msg)

    return FinalizeResult(good=good, errors=errors, alignment=alignment, stats=stats)


def finalize_join_keys(
    table: pa.Table,
    *,
    required_non_null: Sequence[str],
    key_fields: Sequence[str] = (),
    context_fields: Sequence[str] = (),
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
    context_fields
        Additional context columns to copy into error rows.
    stage
        Error stage label for join-key failures.

    Returns
    -------
    FinalizeResult
        Finalize result with filtered good rows and error rows.
    """
    if table.num_rows == 0:
        empty_errors = _empty_error_table(
            table,
            key_fields=key_fields,
            context_fields=context_fields,
            context_columns={},
        )
        return FinalizeResult(
            good=table,
            errors=empty_errors,
            alignment=_empty_alignment_table(),
            stats=_empty_stats_table(),
        )
    context_columns = _context_column_map(table, context_fields)
    resolved_context_fields = _filter_context_fields(
        table,
        context_columns=context_columns,
        fields=(*key_fields, *context_fields),
    )
    context = FinalizeContext(
        table=table,
        row_id=pa.arange(0, table.num_rows),
        context_columns=context_columns,
        context_fields=resolved_context_fields,
    )
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
                    context_fields=resolved_context_fields,
                ),
            )
        )
    good = _filter_good_rows(table, required_mask)
    errors = _concat_errors(
        error_tables,
        table,
        key_fields=key_fields,
        context_fields=resolved_context_fields,
        context_columns=context_columns,
    )
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


def finalize_reader_batches(
    reader: pa.RecordBatchReader,
    *,
    spec: FinalizeSpec,
    finalize_hook: Callable[[FinalizeResult], None] | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> pa.RecordBatchReader:
    """Finalize an Arrow reader per batch, returning a new reader.

    Parameters
    ----------
    reader
        RecordBatchReader to finalize.
    spec
        Finalize specification.
    finalize_hook
        Optional callback invoked with finalize artifacts per batch.
    cancel_check
        Optional cancellation hook invoked between batches.

    Returns
    -------
    pyarrow.RecordBatchReader
        Reader over finalized batches.
    """

    def _iter_batches() -> Iterable[pa.RecordBatch]:
        for batch in reader:
            if cancel_check is not None:
                cancel_check()
            if batch.num_rows == 0:
                continue
            table = pa.Table.from_batches([batch], schema=batch.schema)
            result = finalize_table(table, spec=spec)
            if finalize_hook is not None:
                finalize_hook(result)
            yield from result.good.to_batches(max_chunksize=batch.num_rows)

    finalized = record_batch_reader_from_iterable(_iter_batches(), empty_policy="none")
    if finalized is None:
        return empty_reader_from_schema(reader.schema)
    return finalized


def _prepare_alignment(
    table: pa.Table,
    spec: FinalizeSpec,
) -> tuple[_AlignedContext | None, FinalizeResult | None]:
    context_fields = _resolve_context_fields(spec)
    context_columns = _context_column_map(table, context_fields)
    aligned, report, cast_error = _align_table(table, spec, context_fields=context_fields)
    if aligned is None:
        if cast_error is not None:
            return None, _cast_failure_result(
                table,
                spec,
                context_columns=context_columns,
                report=report,
                detail=cast_error,
            )
        return None, _missing_contract_result(
            table,
            spec,
            context_columns=context_columns,
        )
    resolved_context_fields = _filter_context_fields(
        aligned,
        context_columns=context_columns,
        fields=context_fields,
    )
    return (
        _AlignedContext(
            aligned=aligned,
            report=report,
            context_columns=context_columns,
            context_fields=resolved_context_fields,
        ),
        None,
    )


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


def _cast_failure_result(
    table: pa.Table,
    spec: FinalizeSpec,
    *,
    context_columns: Mapping[str, pa.Array | pa.ChunkedArray],
    report: AlignmentReport | None,
    detail: str,
) -> FinalizeResult:
    errors = _error_table_for_cast_failure(
        table,
        spec,
        context_columns=context_columns,
        detail=detail,
    )
    alignment, stats = _build_artifacts(report, errors, spec)
    if spec.mode == "strict":
        msg = f"Finalize strict mode: nested cast failed ({detail})"
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


def _error_table_for_cast_failure(
    table: pa.Table,
    spec: FinalizeSpec,
    *,
    context_columns: Mapping[str, pa.Array | pa.ChunkedArray],
    detail: str,
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
            error_code="NESTED_CAST_FAILED",
            stage="alignment",
            column="schema",
            detail=detail,
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
) -> tuple[pa.Table | None, AlignmentReport | None, str | None]:
    schema_service = get_schema_service()
    contract_schema = schema_service.get_arrow_schema(spec.table_key)
    if contract_schema is None:
        return None, None, None
    normalized_contract = _normalize_view_schema(contract_schema)
    table_for_alignment = _drop_context_columns(table, normalized_contract, context_fields)
    report = _alignment_report(
        contract_schema=normalized_contract,
        incoming_schema=table_for_alignment.schema,
        table_key=spec.table_key,
        target_name=spec.target_name,
        row_count=table.num_rows,
    )
    try:
        unify_schemas_with_contract_first(
            normalized_contract,
            [_normalize_view_schema(table_for_alignment.schema)],
            promote=spec.schema_promote_options,
        )
    except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError, ValueError) as exc:
        return None, report, str(exc)
    aligned = align_table_to_contract(
        table_for_alignment,
        normalized_contract,
        extras_policy=extras_policy_from_schema(normalized_contract),
        schema_promote_options=spec.schema_promote_options,
    )
    cast_schema = _contract_cast_schema(
        contract_schema=normalized_contract,
        aligned_schema=aligned.schema,
    )
    try:
        aligned_cast = deep_cast_table_to_contract(aligned, cast_schema)
    except (
        pa.ArrowInvalid,
        pa.ArrowNotImplementedError,
        pa.ArrowTypeError,
        TypeError,
        ValueError,
    ) as exc:
        return None, report, str(exc)
    return aligned_cast, report, None


def _alignment_report(
    *,
    contract_schema: pa.Schema,
    incoming_schema: pa.Schema,
    table_key: str,
    target_name: str | None,
    row_count: int,
) -> AlignmentReport:
    contract_fields = {schema_field.name: schema_field.type for schema_field in contract_schema}
    incoming_fields = {schema_field.name: schema_field.type for schema_field in incoming_schema}
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


def _normalize_view_schema(schema: pa.Schema) -> pa.Schema:
    normalized_fields = [
        schema_field.with_type(normalized)
        if (normalized := _normalize_type(schema_field.type)) != schema_field.type
        else schema_field
        for schema_field in schema
    ]
    if not any(
        normalized_field.type != schema_field.type
        for normalized_field, schema_field in zip(normalized_fields, schema, strict=True)
    ):
        return schema
    return pa.schema(normalized_fields, metadata=schema.metadata)


def _contract_cast_schema(
    *,
    contract_schema: pa.Schema,
    aligned_schema: pa.Schema,
) -> pa.Schema:
    contract_by_name = {schema_field.name: schema_field for schema_field in contract_schema}
    fields: list[pa.Field] = []
    for schema_field in aligned_schema:
        contract_field = contract_by_name.get(schema_field.name)
        fields.append(contract_field or schema_field)
    return pa.schema(fields, metadata=contract_schema.metadata)


def _resolve_context_fields(spec: FinalizeSpec) -> tuple[str, ...]:
    return _dedupe_field_names(
        (*spec.key_fields, *spec.context_fields, *_PROVENANCE_CONTEXT_FIELDS)
    )


def _filter_context_fields(
    table: pa.Table,
    *,
    context_columns: Mapping[str, pa.Array | pa.ChunkedArray],
    fields: Sequence[str],
) -> tuple[str, ...]:
    return tuple(
        name
        for name in _dedupe_field_names(fields)
        if name in table.column_names or name in context_columns
    )


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


def _resolve_finalize_spec(spec: FinalizeSpec) -> FinalizeSpec:
    policy = _finalize_policy_for_table(spec.table_key)
    if policy is None:
        return spec
    required_non_null = _dedupe_field_names((*policy.required_non_null, *spec.required_non_null))
    list_policies = _merge_list_policies(spec.list_policies, policy.list_policies)
    invariants = _merge_invariants(spec.invariants, policy.invariants)
    dedupe = spec.dedupe or _dedupe_from_policy(policy.dedupe)
    order_by = spec.order_by
    if not order_by and policy.canonical_sort_keys is not None:
        order_by = tuple((name, "ascending") for name in policy.canonical_sort_keys)
    if (
        required_non_null == spec.required_non_null
        and list_policies == spec.list_policies
        and invariants == spec.invariants
        and dedupe == spec.dedupe
        and order_by == spec.order_by
    ):
        return spec
    return replace(
        spec,
        required_non_null=required_non_null,
        list_policies=list_policies,
        invariants=invariants,
        dedupe=dedupe,
        order_by=order_by,
    )


def _finalize_policy_for_table(table_key: str) -> FinalizePolicy | None:
    schema = get_schema_service().get_table_schema(table_key)
    if schema is None:
        return None
    return schema.finalize_policy


def _dedupe_from_policy(policy: FinalizeDedupeSpec | None) -> FinalizeDedupe | None:
    if policy is None:
        return None
    return FinalizeDedupe(
        enabled=policy.enabled,
        keys=policy.keys,
        prefer_columns=policy.prefer_columns,
        tie_breakers=policy.tie_breakers,
        tier=policy.tier,
        strategy=policy.strategy,
    )


def _merge_list_policies(
    overrides: Sequence[FinalizeListPolicy],
    defaults: Sequence[FinalizeListPolicySpec],
) -> tuple[FinalizeListPolicy, ...]:
    merged: dict[str, FinalizeListPolicy] = {
        policy.column: FinalizeListPolicy(
            column=policy.column,
            null_policy=policy.null_policy,
        )
        for policy in defaults
    }
    for policy in overrides:
        merged[policy.column] = policy
    return tuple(merged.values())


def _merge_invariants(
    overrides: Sequence[FinalizeInvariant],
    defaults: Sequence[FinalizeInvariantSpec],
) -> tuple[FinalizeInvariant, ...]:
    invariants = list(overrides)
    seen = {(inv.kind, inv.column, inv.related) for inv in invariants}
    for policy in defaults:
        invariant = FinalizeInvariant(
            kind=policy.kind,
            column=policy.column,
            related=policy.related,
        )
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
                    error_code=ERROR_CODE_NULL_REQUIRED_LIST,
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


def _dedupe_spec_from_finalize(spec: FinalizeSpec) -> DedupeSpec | None:
    dedupe = spec.dedupe
    if dedupe is None:
        return None
    if (
        dedupe.keys is None
        and dedupe.tie_breakers is None
        and dedupe.tier is None
        and dedupe.strategy is None
    ):
        return None
    return DedupeSpec(
        keys=dedupe.keys or (),
        prefer_columns=dedupe.prefer_columns,
        tie_breakers=dedupe.tie_breakers or (),
        tier=dedupe.tier or "stable_set",
        strategy=dedupe.strategy or "order_independent",
    )


def _apply_dedupe(
    table: pa.Table,
    spec: FinalizeSpec,
    *,
    dedupe_spec: DedupeSpec | None,
) -> pa.Table:
    dedupe = spec.dedupe or FinalizeDedupe()
    if not dedupe.enabled:
        return table
    if dedupe_spec is not None:
        return dedupe_table_for_table(spec.table_key, table, spec=dedupe_spec)
    legacy = DedupeLegacy(
        prefer_columns=dedupe.prefer_columns,
        determinism=dedupe.determinism,
        tie_breaker_columns=dedupe.tie_breaker_columns,
    )
    return dedupe_table_for_table(
        spec.table_key,
        table,
        legacy=legacy,
    )


def _default_order_by(
    table: pa.Table,
    *,
    spec: FinalizeSpec,
    dedupe_spec: DedupeSpec,
) -> Sequence[SortKey]:
    schema = get_schema_service().get_table_schema(spec.table_key)
    canonical_keys = _canonical_sort_keys_for_table(schema)
    if canonical_keys is not None:
        if not canonical_keys:
            return ()
        order_by = [(name, "ascending") for name in canonical_keys]
        return _extend_with_tie_breakers(order_by, dedupe_spec.tie_breakers)
    if dedupe_spec.keys:
        order_by = [(name, "ascending") for name in dedupe_spec.keys]
        return _extend_with_tie_breakers(order_by, dedupe_spec.tie_breakers)
    if schema is None or not schema.primary_key:
        return ()
    order_by = [(name, "ascending") for name in schema.primary_key if name in table.column_names]
    return _extend_with_tie_breakers(order_by, dedupe_spec.tie_breakers)


def _canonical_sort_keys_for_table(
    schema: TableSchema | None,
) -> tuple[str, ...] | None:
    if schema is None:
        return None
    finalize_policy = getattr(schema, "finalize_policy", None)
    if (
        isinstance(finalize_policy, FinalizePolicy)
        and finalize_policy.canonical_sort_keys is not None
    ):
        return finalize_policy.canonical_sort_keys
    return resolve_stable_sort_keys(schema)


def _extend_with_tie_breakers(
    order_by: Sequence[SortKey],
    tie_breakers: Sequence[SortKey],
) -> list[SortKey]:
    if not tie_breakers:
        return list(order_by)
    seen = {name for name, _order in order_by}
    merged = list(order_by)
    for name, order in tie_breakers:
        if name in seen:
            continue
        merged.append((name, order))
        seen.add(name)
    return merged


def _apply_order_by(
    table: pa.Table,
    spec: FinalizeSpec,
    *,
    dedupe_spec: DedupeSpec | None,
) -> pa.Table:
    if spec.order_by:
        return stable_sort_table(table, sort_keys=spec.order_by)
    dedupe = spec.dedupe or FinalizeDedupe()
    if not dedupe.enabled:
        return table
    if dedupe_spec is None:
        if dedupe.determinism == "best_effort":
            return table
        fallback = _default_order_by(
            table,
            spec=spec,
            dedupe_spec=DedupeSpec(),
        )
        return stable_sort_table(table, sort_keys=fallback) if fallback else table
    tier = normalize_dedupe_tier(dedupe_spec.tier)
    if tier != "canonical":
        return table
    fallback = _default_order_by(table, spec=spec, dedupe_spec=dedupe_spec)
    if not fallback:
        msg = "Canonical finalize requires stable order_by keys."
        raise ValueError(msg)
    return stable_sort_table(table, sort_keys=fallback)


def _build_artifacts(
    report: AlignmentReport | None,
    errors: pa.Table,
    spec: FinalizeSpec,
) -> tuple[pa.Table, pa.Table]:
    if spec.emit_artifacts or spec.mode == "tolerant":
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
        base = list_value_length(table[invariant.column])
        mismatch: pa.Array | pa.ChunkedArray | None = None
        for name in invariant.related:
            if name not in table.column_names:
                continue
            other = list_value_length(table[name])
            equal = _fill_null_false(_equal(base, other))
            diff = _invert(equal)
            mismatch = diff if mismatch is None else _or(mismatch, diff)
        return mismatch
    if invariant.kind == "struct_required":
        struct_col = table[invariant.column]
        missing: pa.Array | pa.ChunkedArray | None = None
        for field_name in invariant.related:
            field_values = struct_field(struct_col, field_name)
            valid = _fill_null_false(_is_valid(field_values))
            invalid = _invert(valid)
            missing = invalid if missing is None else _or(missing, invalid)
        return missing
    return None


def _invariant_error_code(invariant: FinalizeInvariant) -> str:
    if invariant.kind == "list_alignment":
        return ERROR_CODE_MISALIGNED_LIST_COLUMNS
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
    "AlignmentReport",
    "FinalizeDedupe",
    "FinalizeInvariant",
    "FinalizeListPolicy",
    "FinalizeResult",
    "FinalizeSpec",
    "JoinPrecheckReport",
    "NullListPolicy",
    "drain_join_precheck_reports",
    "finalize_join_keys",
    "finalize_reader",
    "finalize_reader_batches",
    "finalize_table",
    "record_join_precheck_errors",
]
