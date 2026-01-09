"""Arrow compute kernel helpers for columnar pipelines."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.core.columnar.compute_helpers import call_compute, cast_options, require_array
from codeintel.core.columnar.dedupe_ops import (
    DedupeDeterminism,
    DedupeLegacy,
    DedupeSpec,
    DedupeStrategy,
    DedupeTier,
    DedupeTierNormalized,
)
from codeintel.core.columnar.dedupe_ops import (
    dedupe_keep_first_after_sort as _dedupe_keep_first_after_sort,
)
from codeintel.core.columnar.dedupe_ops import (
    dedupe_table_for_table as _dedupe_table_for_table,
)
from codeintel.core.columnar.dedupe_ops import (
    normalize_dedupe_tier as _normalize_dedupe_tier,
)
from codeintel.core.columnar.dedupe_ops import (
    stable_dedupe_with_ties as _stable_dedupe_with_ties,
)
from codeintel.core.columnar.explode_ops import (
    ExplodeResult,
    ExplodeSpec,
)
from codeintel.core.columnar.explode_ops import (
    explode_list_struct as _explode_list_struct,
)
from codeintel.core.columnar.kernel_shared import (
    SortKey,
    _make_struct,
    hash_struct_ordinal,
    stable_sort_indices,
    stable_sort_table,
)
from codeintel.core.columnar.plan_kernels import explode_edges_for_join as _explode_edges_for_join

if TYPE_CHECKING:
    from codeintel.core.schemas.service import SchemaService


def explode_edges(
    table: pa.Table,
    *,
    spec: ExplodeSpec,
    allowed_columns: Sequence[str] = (),
    table_key: str | None = None,
    schema_service: SchemaService | None = None,
) -> ExplodeResult:
    """Explode list payloads into edge rows.

    Returns
    -------
    ExplodeResult
        Explode output with good rows and errors.
    """
    return _explode_edges_for_join(
        table,
        spec=spec,
        allowed_columns=allowed_columns,
        table_key=table_key,
        schema_service=schema_service,
    )


def explode_edges_with_aligned_lists(
    table: pa.Table,
    *,
    spec: ExplodeSpec,
    allowed_columns: Sequence[str] = (),
    table_key: str | None = None,
    schema_service: SchemaService | None = None,
) -> ExplodeResult:
    """Explode list payloads with aligned list validation.

    Returns
    -------
    ExplodeResult
        Explode output with aligned list validation results.
    """
    return _explode_edges_for_join(
        table,
        spec=spec,
        allowed_columns=allowed_columns,
        table_key=table_key,
        schema_service=schema_service,
    )


def explode_list_struct(
    table: pa.Table,
    *,
    list_col: str,
    parent_cols: Sequence[str],
    struct_fields: Sequence[str] | dict[str, str],
) -> pa.Table:
    """Explode a list<struct> column into a row-per-element table.

    Returns
    -------
    pyarrow.Table
        Table with one row per list element.
    """
    if isinstance(struct_fields, dict):
        mapping = struct_fields
    else:
        mapping = {name: name for name in struct_fields}
    return _explode_list_struct(
        table,
        list_col=list_col,
        parent_cols=parent_cols,
        struct_fields=mapping,
    )


def group_by_aggregate(
    table: pa.Table,
    *,
    keys: Sequence[str],
    aggregations: Sequence[tuple[str, str]],
) -> pa.Table:
    """Group by keys and aggregate columns.

    Parameters
    ----------
    table
        Arrow table to aggregate.
    keys
        Column names to group by.
    aggregations
        Sequence of (column, aggregation) tuples.

    Returns
    -------
    pyarrow.Table
        Aggregated Arrow table.
    """
    return table.group_by(list(keys)).aggregate(list(aggregations))


def dedupe_keep_first_after_sort(
    table: pa.Table,
    *,
    key_columns: Sequence[str],
) -> pa.Table:
    """Return a table with the first row kept per key after sorting.

    Returns
    -------
    pyarrow.Table
        Deduped table with one row per key.
    """
    return _dedupe_keep_first_after_sort(table, key_columns=key_columns)


def stable_dedupe_with_ties(
    table: pa.Table,
    *,
    key_columns: Sequence[str],
    order_by: Sequence[SortKey] = (),
    tie_breakers: Sequence[SortKey] = (),
    require_tie_breakers: bool = False,
) -> pa.Table:
    """Return a deduped table with deterministic tie handling.

    Returns
    -------
    pyarrow.Table
        Deduped table with tie handling applied.
    """
    return _stable_dedupe_with_ties(
        table,
        key_columns=key_columns,
        order_by=order_by,
        tie_breakers=tie_breakers,
        require_tie_breakers=require_tie_breakers,
    )


def dedupe_table_for_table(
    table_key: str,
    table: pa.Table,
    *,
    spec: DedupeSpec | None = None,
    legacy: DedupeLegacy | None = None,
) -> pa.Table:
    """Return a table with duplicate primary-key rows removed.

    Returns
    -------
    pyarrow.Table
        Deduped table.
    """
    return _dedupe_table_for_table(table_key, table, spec=spec, legacy=legacy)


def normalize_dedupe_tier(tier: DedupeTier | None) -> DedupeTierNormalized:
    """Return a normalized dedupe tier for policy enforcement.

    Returns
    -------
    DedupeTierNormalized
        Normalized dedupe tier.
    """
    return _normalize_dedupe_tier(tier)


def coalesce(
    values: Iterable[pa.Array | pa.ChunkedArray | pa.Scalar],
) -> pa.Array | pa.ChunkedArray:
    """Return the first non-null value across inputs.

    Parameters
    ----------
    values
        Arrays or scalars to coalesce.

    Returns
    -------
    pyarrow.Array | pyarrow.ChunkedArray
        Coalesced array.

    Raises
    ------
    ValueError
        If no inputs are provided.
    """
    seq = list(values)
    if not seq:
        msg = "coalesce requires at least one input"
        raise ValueError(msg)
    return require_array(call_compute("coalesce", seq), name="coalesce")


def case_when(
    cases: Sequence[tuple[pa.Array | pa.ChunkedArray, object]],
    *,
    else_: object,
) -> pa.Array | pa.ChunkedArray:
    """Return a case-when array based on boolean masks.

    Parameters
    ----------
    cases
        Sequence of (condition, value) pairs.
    else_
        Default value for rows that do not match any condition.

    Returns
    -------
    pyarrow.Array | pyarrow.ChunkedArray
        Resulting array.

    Raises
    ------
    ValueError
        If no cases are provided.
    """
    if not cases:
        msg = "case_when requires at least one case"
        raise ValueError(msg)
    masks = [mask for mask, _ in cases]
    values = [value for _, value in cases]
    cond_struct = _make_struct(
        masks,
        field_names=[f"cond_{idx}" for idx in range(len(masks))],
    )
    args = [cond_struct, *values, else_]
    return require_array(call_compute("case_when", args), name="case_when")


def hash_struct_goid(
    table: pa.Table,
    *,
    columns: Sequence[str],
) -> pa.Array | pa.ChunkedArray:
    """Hash columns into a deterministic GOID using Arrow kernels.

    Parameters
    ----------
    table
        Source table containing the columns to hash.
    columns
        Column names to hash together.

    Returns
    -------
    pyarrow.Array | pyarrow.ChunkedArray
        Decimal128 values derived from the hash kernel.

    Raises
    ------
    RuntimeError
        If the hash kernel is unavailable.
    ValueError
        If no columns are provided.
    """
    if not columns:
        msg = "hash_struct_goid requires at least one column"
        raise ValueError(msg)
    try:
        pc.get_function("hash")
    except (AttributeError, KeyError):
        msg = "Arrow hash kernel is unavailable; upgrade pyarrow to enable it."
        raise RuntimeError(msg) from None
    struct_values = _make_struct(
        [table[column] for column in columns],
        field_names=list(columns),
    )
    hashed = require_array(call_compute("hash", [struct_values]), name="hash")
    return pc.cast(hashed, pa.decimal128(38, 0))


def list_value_length(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    """Return list value lengths for list-like arrays.

    Returns
    -------
    pyarrow.Array | pyarrow.ChunkedArray
        Length of each list element.
    """
    return require_array(
        call_compute("list_value_length", [values]),
        name="list_value_length",
    )


def list_parent_indices(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    """Return parent indices for list elements.

    Returns
    -------
    pyarrow.Array | pyarrow.ChunkedArray
        Parent row indices for each list element.
    """
    return require_array(
        call_compute("list_parent_indices", [values]),
        name="list_parent_indices",
    )


def list_flatten(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    """Flatten list elements into a single array.

    Returns
    -------
    pyarrow.Array | pyarrow.ChunkedArray
        Flattened list values.
    """
    return require_array(
        call_compute("list_flatten", [values]),
        name="list_flatten",
    )


def list_element(
    values: pa.Array | pa.ChunkedArray,
    *,
    index: int,
) -> pa.Array | pa.ChunkedArray:
    """Return list elements at the provided index.

    Returns
    -------
    pyarrow.Array | pyarrow.ChunkedArray
        Array of list elements at the index.
    """
    return require_array(
        call_compute("list_element", [values, index]),
        name="list_element",
    )


def list_slice(
    values: pa.Array | pa.ChunkedArray,
    *,
    start: int | None = None,
    stop: int | None = None,
) -> pa.Array | pa.ChunkedArray:
    """Return list slices for each list element.

    Returns
    -------
    pyarrow.Array | pyarrow.ChunkedArray
        Array of list slices.
    """
    args: list[object] = [values]
    if start is not None:
        args.append(start)
    if stop is not None:
        args.append(stop)
    return require_array(
        call_compute("list_slice", args),
        name="list_slice",
    )


def struct_field(
    values: pa.Array | pa.ChunkedArray,
    field_name: str,
) -> pa.Array | pa.ChunkedArray:
    """Return a struct field array for the provided field name.

    Returns
    -------
    pyarrow.Array | pyarrow.ChunkedArray
        Struct field values.

    Raises
    ------
    TypeError
        If the values are not struct arrays.
    """
    if not pa.types.is_struct(values.type):
        msg = f"struct_field expects struct values but got {values.type}"
        raise TypeError(msg)
    return require_array(
        call_compute("struct_field", [values, field_name]),
        name="struct_field",
    )


def regex_match(
    values: pa.Array | pa.ChunkedArray,
    *,
    pattern: str,
    ignore_case: bool = False,
) -> pa.Array | pa.ChunkedArray:
    """Return a boolean mask for regex matches.

    Returns
    -------
    pyarrow.Array | pyarrow.ChunkedArray
        Boolean match mask.
    """
    options = _match_regex_options(pattern=pattern, ignore_case=ignore_case)
    if options is None:
        result = call_compute("match_substring_regex", [values, pattern])
    else:
        result = call_compute("match_substring_regex", [values], options=options)
    return require_array(result, name="match_substring_regex")


def regex_replace(
    values: pa.Array | pa.ChunkedArray,
    *,
    pattern: str,
    replacement: str,
    ignore_case: bool = False,
) -> pa.Array | pa.ChunkedArray:
    """Return an array with regex replacements applied.

    Returns
    -------
    pyarrow.Array | pyarrow.ChunkedArray
        Array with substitutions applied.
    """
    options = _replace_regex_options(
        pattern=pattern,
        replacement=replacement,
        ignore_case=ignore_case,
    )
    if options is None:
        result = call_compute("replace_substring_regex", [values, pattern, replacement])
    else:
        result = call_compute("replace_substring_regex", [values], options=options)
    return require_array(result, name="replace_substring_regex")


def safe_cast(
    values: pa.Array | pa.ChunkedArray,
    *,
    target_type: pa.DataType,
) -> pa.Array | pa.ChunkedArray:
    """Return safely cast values to a target type.

    Returns
    -------
    pyarrow.Array | pyarrow.ChunkedArray
        Casted values.
    """
    options = cast_options(target_type, safe=True)
    return require_array(call_compute("cast", [values], options=options), name="cast")


def safe_divide(
    numerator: pa.Array | pa.ChunkedArray,
    denominator: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    """Divide arrays while returning nulls on divide-by-zero.

    Returns
    -------
    pyarrow.Array | pyarrow.ChunkedArray
        Safe division results.
    """
    zero_mask = require_array(
        call_compute(
            "equal",
            [denominator, pa.scalar(0, type=denominator.type)],
        ),
        name="equal",
    )
    null_denominator = pa.scalar(None, type=denominator.type)
    safe_denominator = require_array(
        call_compute("if_else", [zero_mask, null_denominator, denominator]),
        name="if_else",
    )
    return require_array(
        call_compute("divide", [numerator, safe_denominator]),
        name="divide",
    )


def indices_nonzero(
    mask: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    """Return indices where mask is non-zero/true.

    Returns
    -------
    pyarrow.Array | pyarrow.ChunkedArray
        Indices for true values.
    """
    return require_array(call_compute("indices_nonzero", [mask]), name="indices_nonzero")


def replace_with_mask(
    values: pa.Array | pa.ChunkedArray,
    *,
    mask: pa.Array | pa.ChunkedArray,
    replacement: object,
) -> pa.Array | pa.ChunkedArray:
    """Replace values selected by a mask with a replacement.

    Returns
    -------
    pyarrow.Array | pyarrow.ChunkedArray
        Array with masked replacements applied.
    """
    return require_array(
        call_compute("replace_with_mask", [values, mask, replacement]),
        name="replace_with_mask",
    )


def _match_regex_options(
    *,
    pattern: str,
    ignore_case: bool,
) -> pc.FunctionOptions | None:
    options_type = getattr(pc, "MatchSubstringRegexOptions", None)
    if not callable(options_type):
        return None
    try:
        return options_type(pattern=pattern, ignore_case=ignore_case)
    except TypeError:
        return options_type(pattern=pattern)


def _replace_regex_options(
    *,
    pattern: str,
    replacement: str,
    ignore_case: bool,
) -> pc.FunctionOptions | None:
    options_type = getattr(pc, "ReplaceSubstringRegexOptions", None)
    if not callable(options_type):
        return None
    try:
        return options_type(
            pattern=pattern,
            replacement=replacement,
            ignore_case=ignore_case,
        )
    except TypeError:
        return options_type(pattern=pattern, replacement=replacement)


__all__ = [
    "DedupeDeterminism",
    "DedupeLegacy",
    "DedupeSpec",
    "DedupeStrategy",
    "DedupeTier",
    "DedupeTierNormalized",
    "case_when",
    "coalesce",
    "dedupe_keep_first_after_sort",
    "dedupe_table_for_table",
    "explode_edges",
    "explode_edges_with_aligned_lists",
    "explode_list_struct",
    "group_by_aggregate",
    "hash_struct_goid",
    "hash_struct_ordinal",
    "indices_nonzero",
    "list_element",
    "list_flatten",
    "list_parent_indices",
    "list_slice",
    "list_value_length",
    "normalize_dedupe_tier",
    "regex_match",
    "regex_replace",
    "replace_with_mask",
    "safe_cast",
    "safe_divide",
    "stable_dedupe_with_ties",
    "stable_sort_indices",
    "stable_sort_table",
    "struct_field",
]
