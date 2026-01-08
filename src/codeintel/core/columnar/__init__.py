"""Columnar streaming protocols and adapters."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.core.columnar.acero_ops import build_exec_plan
    from codeintel.core.columnar.compute import (
        count_distinct,
        count_non_positive,
        count_true,
        orphan_ref_count,
    )
    from codeintel.core.columnar.compute_config import (
        DEFAULT_CAST_SAFE,
        DEFAULT_SCALAR_AGG,
        DEFAULT_SCALAR_AGG_ALLOW_NULL,
        DEFAULT_TAKE,
    )
    from codeintel.core.columnar.compute_helpers import call_compute, require_array, require_scalar
    from codeintel.core.columnar.dedupe_ops import dedupe_table_for_table
    from codeintel.core.columnar.explode_ops import (
        ExplodeResult,
        ExplodeSpec,
        explode_edges,
        explode_list_struct,
    )
    from codeintel.core.columnar.expr_vocab import E, ExprVocab
    from codeintel.core.columnar.finalize_ops import (
        FinalizeDedupe,
        FinalizeInvariant,
        FinalizeResult,
        FinalizeSpec,
        finalize_table,
    )
    from codeintel.core.columnar.groupby import group_by_aggregate
    from codeintel.core.columnar.ipc_ops import (
        ArrowIpcStreamError,
        iter_ipc_stream,
        read_ipc_stream,
        write_ipc_stream,
    )
    from codeintel.core.columnar.iter import iter_array_values, iter_batches, iter_rows
    from codeintel.core.columnar.kernels import (
        case_when,
        coalesce,
        hash_struct_ordinal,
        stable_sort_indices,
    )
    from codeintel.core.columnar.masks import (
        and_mask,
        fill_null_false,
        filter_valid,
        invert_mask,
        is_valid_mask,
    )
    from codeintel.core.columnar.nested_ops import (
        deep_cast_array,
        deep_cast_table_to_contract,
        make_extras_kv_map,
        make_extras_struct,
        unify_schemas_with_contract_first,
    )
    from codeintel.core.columnar.normalization import (
        normalize_array,
        normalize_array_for_compute,
        normalize_table,
        normalize_table_for_compute,
    )
    from codeintel.core.columnar.plan_ops import HashJoinSpec, Plan
    from codeintel.core.columnar.rows import (
        ColumnarRowBuffer,
        ColumnarRows,
        columnar_buffer_for_table_key,
        columnar_row_count,
    )
    from codeintel.core.columnar.schema_alignment import (
        align_reader_to_contract,
        extras_policy_from_schema,
    )
    from codeintel.core.columnar.schema_ops import concat_tables_unified, unify_schemas
    from codeintel.core.columnar.set_ops import is_in_mask, value_set
    from codeintel.core.columnar.stream import (
        ColumnarStream,
        ColumnarStreamAdapter,
        LazyFrameStream,
        RecordBatchReaderStream,
        coerce_arrow_reader,
        coerce_arrow_table,
    )

    _TYPE_CHECKING_EXPORTS = (
        DEFAULT_CAST_SAFE,
        DEFAULT_SCALAR_AGG,
        DEFAULT_SCALAR_AGG_ALLOW_NULL,
        DEFAULT_TAKE,
        count_distinct,
        count_non_positive,
        count_true,
        orphan_ref_count,
        call_compute,
        require_array,
        require_scalar,
        dedupe_table_for_table,
        build_exec_plan,
        E,
        ExprVocab,
        ExplodeResult,
        ExplodeSpec,
        explode_edges,
        explode_list_struct,
        FinalizeDedupe,
        FinalizeInvariant,
        FinalizeResult,
        FinalizeSpec,
        finalize_table,
        group_by_aggregate,
        ArrowIpcStreamError,
        iter_ipc_stream,
        read_ipc_stream,
        write_ipc_stream,
        iter_array_values,
        iter_batches,
        iter_rows,
        case_when,
        coalesce,
        hash_struct_ordinal,
        stable_sort_indices,
        and_mask,
        fill_null_false,
        filter_valid,
        invert_mask,
        is_valid_mask,
        normalize_array,
        normalize_array_for_compute,
        normalize_table,
        normalize_table_for_compute,
        deep_cast_array,
        deep_cast_table_to_contract,
        make_extras_kv_map,
        make_extras_struct,
        unify_schemas_with_contract_first,
        HashJoinSpec,
        Plan,
        ColumnarRowBuffer,
        ColumnarRows,
        columnar_buffer_for_table_key,
        columnar_row_count,
        align_reader_to_contract,
        extras_policy_from_schema,
        concat_tables_unified,
        unify_schemas,
        ColumnarStream,
        ColumnarStreamAdapter,
        LazyFrameStream,
        RecordBatchReaderStream,
        coerce_arrow_reader,
        coerce_arrow_table,
        is_in_mask,
        value_set,
    )

_EXPORTS: dict[str, tuple[str, str]] = {
    "DEFAULT_CAST_SAFE": ("codeintel.core.columnar.compute_config", "DEFAULT_CAST_SAFE"),
    "DEFAULT_SCALAR_AGG": ("codeintel.core.columnar.compute_config", "DEFAULT_SCALAR_AGG"),
    "DEFAULT_SCALAR_AGG_ALLOW_NULL": (
        "codeintel.core.columnar.compute_config",
        "DEFAULT_SCALAR_AGG_ALLOW_NULL",
    ),
    "DEFAULT_TAKE": ("codeintel.core.columnar.compute_config", "DEFAULT_TAKE"),
    "count_distinct": ("codeintel.core.columnar.compute", "count_distinct"),
    "count_non_positive": ("codeintel.core.columnar.compute", "count_non_positive"),
    "count_true": ("codeintel.core.columnar.compute", "count_true"),
    "orphan_ref_count": ("codeintel.core.columnar.compute", "orphan_ref_count"),
    "call_compute": ("codeintel.core.columnar.compute_helpers", "call_compute"),
    "require_array": ("codeintel.core.columnar.compute_helpers", "require_array"),
    "require_scalar": ("codeintel.core.columnar.compute_helpers", "require_scalar"),
    "dedupe_table_for_table": ("codeintel.core.columnar.dedupe_ops", "dedupe_table_for_table"),
    "build_exec_plan": ("codeintel.core.columnar.acero_ops", "build_exec_plan"),
    "E": ("codeintel.core.columnar.expr_vocab", "E"),
    "ExprVocab": ("codeintel.core.columnar.expr_vocab", "ExprVocab"),
    "ExplodeResult": ("codeintel.core.columnar.explode_ops", "ExplodeResult"),
    "ExplodeSpec": ("codeintel.core.columnar.explode_ops", "ExplodeSpec"),
    "explode_edges": ("codeintel.core.columnar.explode_ops", "explode_edges"),
    "explode_list_struct": ("codeintel.core.columnar.explode_ops", "explode_list_struct"),
    "FinalizeDedupe": ("codeintel.core.columnar.finalize_ops", "FinalizeDedupe"),
    "FinalizeInvariant": ("codeintel.core.columnar.finalize_ops", "FinalizeInvariant"),
    "FinalizeResult": ("codeintel.core.columnar.finalize_ops", "FinalizeResult"),
    "FinalizeSpec": ("codeintel.core.columnar.finalize_ops", "FinalizeSpec"),
    "finalize_table": ("codeintel.core.columnar.finalize_ops", "finalize_table"),
    "group_by_aggregate": ("codeintel.core.columnar.groupby", "group_by_aggregate"),
    "ArrowIpcStreamError": ("codeintel.core.columnar.ipc_ops", "ArrowIpcStreamError"),
    "iter_ipc_stream": ("codeintel.core.columnar.ipc_ops", "iter_ipc_stream"),
    "read_ipc_stream": ("codeintel.core.columnar.ipc_ops", "read_ipc_stream"),
    "write_ipc_stream": ("codeintel.core.columnar.ipc_ops", "write_ipc_stream"),
    "iter_array_values": ("codeintel.core.columnar.iter", "iter_array_values"),
    "iter_batches": ("codeintel.core.columnar.iter", "iter_batches"),
    "iter_rows": ("codeintel.core.columnar.iter", "iter_rows"),
    "case_when": ("codeintel.core.columnar.kernels", "case_when"),
    "coalesce": ("codeintel.core.columnar.kernels", "coalesce"),
    "hash_struct_ordinal": ("codeintel.core.columnar.kernels", "hash_struct_ordinal"),
    "stable_sort_indices": ("codeintel.core.columnar.kernels", "stable_sort_indices"),
    "and_mask": ("codeintel.core.columnar.masks", "and_mask"),
    "fill_null_false": ("codeintel.core.columnar.masks", "fill_null_false"),
    "filter_valid": ("codeintel.core.columnar.masks", "filter_valid"),
    "invert_mask": ("codeintel.core.columnar.masks", "invert_mask"),
    "is_valid_mask": ("codeintel.core.columnar.masks", "is_valid_mask"),
    "normalize_array": ("codeintel.core.columnar.normalization", "normalize_array"),
    "normalize_array_for_compute": (
        "codeintel.core.columnar.normalization",
        "normalize_array_for_compute",
    ),
    "normalize_table": ("codeintel.core.columnar.normalization", "normalize_table"),
    "normalize_table_for_compute": (
        "codeintel.core.columnar.normalization",
        "normalize_table_for_compute",
    ),
    "deep_cast_array": ("codeintel.core.columnar.nested_ops", "deep_cast_array"),
    "deep_cast_table_to_contract": (
        "codeintel.core.columnar.nested_ops",
        "deep_cast_table_to_contract",
    ),
    "make_extras_kv_map": ("codeintel.core.columnar.nested_ops", "make_extras_kv_map"),
    "make_extras_struct": ("codeintel.core.columnar.nested_ops", "make_extras_struct"),
    "unify_schemas_with_contract_first": (
        "codeintel.core.columnar.nested_ops",
        "unify_schemas_with_contract_first",
    ),
    "HashJoinSpec": ("codeintel.core.columnar.acero_ops", "HashJoinSpec"),
    "Plan": ("codeintel.core.columnar.acero_ops", "Plan"),
    "ColumnarRowBuffer": ("codeintel.core.columnar.rows", "ColumnarRowBuffer"),
    "ColumnarRows": ("codeintel.core.columnar.rows", "ColumnarRows"),
    "columnar_buffer_for_table_key": (
        "codeintel.core.columnar.rows",
        "columnar_buffer_for_table_key",
    ),
    "columnar_row_count": ("codeintel.core.columnar.rows", "columnar_row_count"),
    "align_reader_to_contract": (
        "codeintel.core.columnar.schema_alignment",
        "align_reader_to_contract",
    ),
    "extras_policy_from_schema": (
        "codeintel.core.columnar.schema_alignment",
        "extras_policy_from_schema",
    ),
    "concat_tables_unified": ("codeintel.core.columnar.schema_ops", "concat_tables_unified"),
    "unify_schemas": ("codeintel.core.columnar.schema_ops", "unify_schemas"),
    "ColumnarStream": ("codeintel.core.columnar.stream", "ColumnarStream"),
    "ColumnarStreamAdapter": ("codeintel.core.columnar.stream", "ColumnarStreamAdapter"),
    "LazyFrameStream": ("codeintel.core.columnar.stream", "LazyFrameStream"),
    "RecordBatchReaderStream": ("codeintel.core.columnar.stream", "RecordBatchReaderStream"),
    "coerce_arrow_reader": ("codeintel.core.columnar.stream", "coerce_arrow_reader"),
    "coerce_arrow_table": ("codeintel.core.columnar.stream", "coerce_arrow_table"),
    "is_in_mask": ("codeintel.core.columnar.set_ops", "is_in_mask"),
    "value_set": ("codeintel.core.columnar.set_ops", "value_set"),
}


def __getattr__(name: str) -> object:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = (
    "DEFAULT_CAST_SAFE",
    "DEFAULT_SCALAR_AGG",
    "DEFAULT_SCALAR_AGG_ALLOW_NULL",
    "DEFAULT_TAKE",
    "ArrowIpcStreamError",
    "ColumnarRowBuffer",
    "ColumnarRows",
    "ColumnarStream",
    "ColumnarStreamAdapter",
    "E",
    "ExplodeResult",
    "ExplodeSpec",
    "ExprVocab",
    "FinalizeDedupe",
    "FinalizeInvariant",
    "FinalizeResult",
    "FinalizeSpec",
    "HashJoinSpec",
    "LazyFrameStream",
    "Plan",
    "RecordBatchReaderStream",
    "align_reader_to_contract",
    "and_mask",
    "build_exec_plan",
    "call_compute",
    "case_when",
    "coalesce",
    "coerce_arrow_reader",
    "coerce_arrow_table",
    "columnar_buffer_for_table_key",
    "columnar_row_count",
    "concat_tables_unified",
    "count_distinct",
    "count_non_positive",
    "count_true",
    "dedupe_table_for_table",
    "deep_cast_array",
    "deep_cast_table_to_contract",
    "explode_edges",
    "explode_list_struct",
    "extras_policy_from_schema",
    "fill_null_false",
    "filter_valid",
    "finalize_table",
    "group_by_aggregate",
    "hash_struct_ordinal",
    "invert_mask",
    "is_in_mask",
    "is_valid_mask",
    "iter_array_values",
    "iter_batches",
    "iter_ipc_stream",
    "iter_rows",
    "make_extras_kv_map",
    "make_extras_struct",
    "normalize_array",
    "normalize_array_for_compute",
    "normalize_table",
    "normalize_table_for_compute",
    "orphan_ref_count",
    "read_ipc_stream",
    "require_array",
    "require_scalar",
    "stable_sort_indices",
    "unify_schemas",
    "unify_schemas_with_contract_first",
    "value_set",
    "write_ipc_stream",
)
