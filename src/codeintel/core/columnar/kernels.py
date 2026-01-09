"""Arrow compute kernel helpers for columnar pipelines."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.core.columnar.compute_helpers import (
    call_compute,
    cast_options,
    require_array,
    sort_options,
)
from codeintel.core.columnar.explode_ops import (
    ExplodeResult,
    ExplodeSpec,
)
from codeintel.core.columnar.explode_ops import (
    explode_edges as _explode_edges,
)
from codeintel.core.columnar.explode_ops import (
    explode_edges_with_aligned_lists as _explode_edges_with_aligned_lists,
)
from codeintel.core.columnar.explode_ops import (
    explode_list_struct as _explode_list_struct,
)

SortKey = tuple[str, Literal["ascending", "descending"]]


def explode_edges(
    table: pa.Table,
    *,
    spec: ExplodeSpec,
) -> ExplodeResult:
    """Explode list payloads into edge rows.

    Returns
    -------
    ExplodeResult
        Explode output with good rows and errors.
    """
    return _explode_edges(table, spec=spec)


def explode_edges_with_aligned_lists(
    table: pa.Table,
    *,
    spec: ExplodeSpec,
) -> ExplodeResult:
    """Explode list payloads with aligned list validation.

    Returns
    -------
    ExplodeResult
        Explode output with aligned list validation results.
    """
    return _explode_edges_with_aligned_lists(table, spec=spec)


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


def stable_sort_indices(
    table: pa.Table,
    *,
    sort_keys: Sequence[SortKey],
    null_placement: Literal["at_end", "at_start"] = "at_end",
) -> pa.Array | pa.ChunkedArray:
    """Return stable sort indices for a table.

    Parameters
    ----------
    table
        Table to sort.
    sort_keys
        Sequence of (column, order) pairs.
    null_placement
        Null placement policy.

    Returns
    -------
    pyarrow.Array | pyarrow.ChunkedArray
        Indices that define a stable sort order.

    Raises
    ------
    TypeError
        If Arrow does not return an array.
    """
    options = sort_options(sort_keys, null_placement=null_placement)
    result = call_compute("sort_indices", [table], options=options)
    if isinstance(result, (pa.Array, pa.ChunkedArray)):
        return result
    msg = "Arrow compute sort_indices did not return an array."
    raise TypeError(msg)


def stable_sort_table(
    table: pa.Table,
    *,
    sort_keys: Sequence[SortKey],
    null_placement: Literal["at_end", "at_start"] = "at_end",
) -> pa.Table:
    """Return a table sorted using stable sort indices.

    Parameters
    ----------
    table
        Table to sort.
    sort_keys
        Sequence of (column, order) pairs.
    null_placement
        Null placement policy.

    Returns
    -------
    pyarrow.Table
        Table sorted by the provided keys.
    """
    if table.num_rows <= 1 or not sort_keys:
        return table
    indices = stable_sort_indices(
        table,
        sort_keys=sort_keys,
        null_placement=null_placement,
    )
    return table.take(indices)


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


def hash_struct_ordinal(
    table: pa.Table,
    *,
    columns: Sequence[str],
    modulus: int,
) -> pa.Array | pa.ChunkedArray:
    """Hash columns into a deterministic ordinal when kernels are available.

    Parameters
    ----------
    table
        Source table containing the columns to hash.
    columns
        Column names to hash together.
    modulus
        Modulus applied to the hash result.

    Returns
    -------
    pyarrow.Array | pyarrow.ChunkedArray
        Ordinal values derived from the hash.

    Raises
    ------
    RuntimeError
        If the hash kernel is unavailable.
    ValueError
        If the modulus is invalid.
    """
    if modulus <= 0:
        msg = "hash_struct_ordinal requires a positive modulus"
        raise ValueError(msg)
    if not columns:
        msg = "hash_struct_ordinal requires at least one column"
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
    hashed_u64 = pc.cast(hashed, pa.uint64())
    modded = require_array(
        call_compute("mod", [hashed_u64, pa.scalar(modulus, type=pa.uint64())]),
        name="mod",
    )
    return pc.cast(modded, pa.int64())


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


def _make_struct(
    values: Sequence[pa.Array | pa.ChunkedArray],
    *,
    field_names: Sequence[str],
) -> pa.Array | pa.ChunkedArray:
    options_factory = getattr(pc, "MakeStructOptions", None)
    options = options_factory(field_names=list(field_names)) if callable(options_factory) else None
    result = call_compute("make_struct", list(values), options=options)
    return require_array(result, name="make_struct")


__all__ = [
    "case_when",
    "coalesce",
    "explode_edges",
    "explode_edges_with_aligned_lists",
    "explode_list_struct",
    "hash_struct_goid",
    "hash_struct_ordinal",
    "indices_nonzero",
    "list_element",
    "list_flatten",
    "list_parent_indices",
    "list_slice",
    "list_value_length",
    "regex_match",
    "regex_replace",
    "replace_with_mask",
    "safe_cast",
    "safe_divide",
    "stable_sort_indices",
    "stable_sort_table",
    "struct_field",
]
