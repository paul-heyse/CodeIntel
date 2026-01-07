"""Common PyArrow compute masks for columnar filtering."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.build.tabular import array_ops as _array_ops
from codeintel.build.tabular.compute_helpers import safe_filter

ensure_array = _array_ops.ensure_array
index_in = _array_ops.index_in
normalize_string_view_array = _array_ops.normalize_string_view_array
value_set_array = _array_ops.value_set_array

if TYPE_CHECKING:
    from pyarrow.compute import Expression as ComputeExpression
else:
    ComputeExpression = object

_EXPR_TYPE = getattr(pc, "Expression", None)


def and_kleene(
    left: pa.Array | pa.ChunkedArray,
    right: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    """Combine two boolean masks using Kleene AND semantics.

    Returns
    -------
    pa.Array | pa.ChunkedArray
        Combined boolean mask.
    """
    return pc.call_function("and_kleene", [left, right])


def or_kleene(
    left: pa.Array | pa.ChunkedArray,
    right: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    """Combine two boolean masks using Kleene OR semantics.

    Returns
    -------
    pa.Array | pa.ChunkedArray
        Combined boolean mask.
    """
    return pc.call_function("or_kleene", [left, right])


def _coerce_scalar_like(
    left: pa.Array | pa.ChunkedArray,
    right: pa.Array | pa.ChunkedArray | pa.Scalar,
) -> pa.Array | pa.ChunkedArray | pa.Scalar:
    if isinstance(right, (pa.Array, pa.ChunkedArray)):
        return normalize_string_view_array(right)
    if not isinstance(right, pa.Scalar):
        return right
    if right.type == left.type:
        return right
    return pa.scalar(right.as_py(), type=left.type)


def equal_mask(
    left: pa.Array | pa.ChunkedArray,
    right: pa.Array | pa.ChunkedArray | pa.Scalar,
) -> pa.Array | pa.ChunkedArray:
    """Return a mask for equality between two values/arrays.

    Returns
    -------
    pa.Array | pa.ChunkedArray
        Boolean mask of equality comparisons.
    """
    left_norm = normalize_string_view_array(left)
    return pc.call_function("equal", [left_norm, _coerce_scalar_like(left_norm, right)])


def not_equal_mask(
    left: pa.Array | pa.ChunkedArray,
    right: pa.Array | pa.ChunkedArray | pa.Scalar,
) -> pa.Array | pa.ChunkedArray:
    """Return a mask for inequality between two values/arrays.

    Returns
    -------
    pa.Array | pa.ChunkedArray
        Boolean mask of inequality comparisons.
    """
    left_norm = normalize_string_view_array(left)
    return pc.call_function("not_equal", [left_norm, _coerce_scalar_like(left_norm, right)])


def bit_wise_and(
    left: pa.Array | pa.ChunkedArray,
    right: pa.Array | pa.ChunkedArray | pa.Scalar,
) -> pa.Array | pa.ChunkedArray:
    """Return a bitwise AND between two values/arrays.

    Returns
    -------
    pa.Array | pa.ChunkedArray
        Resulting array from bitwise AND.
    """
    return pc.call_function("bit_wise_and", [left, right])


def is_valid_mask(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    """Return a mask for non-null values.

    Returns
    -------
    pa.Array | pa.ChunkedArray
        Boolean mask of valid entries.
    """
    return pc.call_function("is_valid", [values])


def is_null_mask(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    """Return a mask for null values.

    Returns
    -------
    pa.Array | pa.ChunkedArray
        Boolean mask of null entries.
    """
    return pc.call_function("is_null", [values])


def invert_mask(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    """Invert a boolean mask.

    Returns
    -------
    pa.Array | pa.ChunkedArray
        Inverted boolean mask.
    """
    return pc.call_function("invert", [values])


def is_in_mask(
    values: pa.Array | pa.ChunkedArray,
    *,
    value_set: Sequence[object] | pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    """Return a mask for membership in a value set.

    Returns
    -------
    pa.Array | pa.ChunkedArray
        Boolean mask for membership in the value set.
    """
    normalized = normalize_string_view_array(values)
    resolved = value_set_array(value_set, like=normalized)
    options = pc.SetLookupOptions(value_set=resolved)
    return pc.call_function("is_in", [normalized], options=options)


def index_in_values(
    values: pa.Array | pa.ChunkedArray,
    *,
    value_set: Sequence[object] | pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    """Return index positions of values in a lookup set.

    Returns
    -------
    pa.Array | pa.ChunkedArray
        Index positions per input value.
    """
    return index_in(values, value_set=value_set)


def non_empty_string_mask(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    """Return a mask for non-empty string entries.

    Returns
    -------
    pa.Array | pa.ChunkedArray
        Boolean mask for non-empty strings.
    """
    is_valid = is_valid_mask(values)
    lengths = pc.call_function("utf8_length", [values])
    non_empty = pc.call_function("greater", [lengths, pc.scalar(0)])
    return and_kleene(is_valid, non_empty)


def _field_expr(field: str | ComputeExpression) -> ComputeExpression:
    if isinstance(field, str):
        return pc.field(field)
    return field


def _scalar_expr(value: object) -> ComputeExpression:
    if _EXPR_TYPE is not None and isinstance(value, _EXPR_TYPE):
        return value
    if isinstance(value, pa.Scalar):
        as_py = getattr(value, "as_py", None)
        scalar_value = as_py() if callable(as_py) else value
        return pc.scalar(scalar_value)
    return pc.scalar(value)


def equal_expr(field: str | ComputeExpression, value: object) -> ComputeExpression:
    """Return an equality expression for Arrow filters.

    Returns
    -------
    pyarrow.compute.Expression
        Equality expression for the given field and value.
    """
    return pc.call_function("equal", [_field_expr(field), _scalar_expr(value)])


def not_equal_expr(field: str | ComputeExpression, value: object) -> ComputeExpression:
    """Return an inequality expression for Arrow filters.

    Returns
    -------
    pyarrow.compute.Expression
        Inequality expression for the given field and value.
    """
    return pc.call_function("not_equal", [_field_expr(field), _scalar_expr(value)])


def is_valid_expr(field: str | ComputeExpression) -> ComputeExpression:
    """Return an is-valid expression for Arrow filters.

    Returns
    -------
    pyarrow.compute.Expression
        Validity expression for the given field.
    """
    return pc.call_function("is_valid", [_field_expr(field)])


def is_in_expr(
    field: str | ComputeExpression,
    *,
    value_set: Sequence[object] | pa.Array | pa.ChunkedArray,
) -> ComputeExpression:
    """Return an is-in expression for Arrow filters.

    Returns
    -------
    pyarrow.compute.Expression
        Membership expression for the given field.
    """
    resolved = value_set_array(value_set)
    options = pc.SetLookupOptions(value_set=resolved)
    return pc.call_function("is_in", [_field_expr(field)], options=options)


def non_empty_string_expr(field: str | ComputeExpression) -> ComputeExpression:
    """Return an expression for non-empty string entries.

    Returns
    -------
    pyarrow.compute.Expression
        Expression for non-empty string entries.
    """
    field_expr = _field_expr(field)
    lengths = pc.call_function("utf8_length", [field_expr])
    non_empty = pc.call_function("greater", [lengths, pc.scalar(0)])
    is_valid = pc.call_function("is_valid", [field_expr])
    return pc.call_function("and_kleene", [is_valid, non_empty])


def language_is_python_or_null(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    """Return a mask for Python language markers or NULLs.

    Returns
    -------
    pa.Array | pa.ChunkedArray
        Boolean mask for Python or NULL values.
    """
    is_null = pc.call_function("is_null", [values])
    is_python = equal_mask(values, pc.scalar("python"))
    return or_kleene(is_null, is_python)


def kind_is_function_or_method(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    """Return a mask for function or method kinds.

    Returns
    -------
    pa.Array | pa.ChunkedArray
        Boolean mask for function/method kinds.
    """
    is_function = equal_mask(values, pc.scalar("function"))
    is_method = equal_mask(values, pc.scalar("method"))
    return or_kleene(is_function, is_method)


def node_type_is_function(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    """Return a mask for Python function AST node types.

    Returns
    -------
    pa.Array | pa.ChunkedArray
        Boolean mask for function/async function node types.
    """
    is_function = equal_mask(values, pc.scalar("FunctionDef"))
    is_async = equal_mask(values, pc.scalar("AsyncFunctionDef"))
    return or_kleene(is_function, is_async)


@dataclass(frozen=True, slots=True)
class FilterExprContext:
    """Context for snapshot-aligned table filtering."""

    repo: str | None = None
    commit: str | None = None

    def apply(self, table: pa.Table) -> pa.Table:
        """Apply repo/commit filters when available.

        Returns
        -------
        pyarrow.Table
            Filtered table when snapshot columns are present.
        """
        mask: pa.Array | pa.ChunkedArray | None = None
        if self.repo is not None and "repo" in table.column_names:
            mask = equal_mask(table["repo"], pa.scalar(self.repo))
        if self.commit is not None and "commit" in table.column_names:
            commit_mask = equal_mask(table["commit"], pa.scalar(self.commit))
            mask = commit_mask if mask is None else and_kleene(mask, commit_mask)
        if mask is None:
            return table
        return safe_filter(table, mask)


__all__ = [
    "FilterExprContext",
    "and_kleene",
    "bit_wise_and",
    "ensure_array",
    "equal_expr",
    "equal_mask",
    "index_in_values",
    "invert_mask",
    "is_in_expr",
    "is_in_mask",
    "is_null_mask",
    "is_valid_expr",
    "is_valid_mask",
    "kind_is_function_or_method",
    "language_is_python_or_null",
    "node_type_is_function",
    "non_empty_string_expr",
    "non_empty_string_mask",
    "not_equal_expr",
    "not_equal_mask",
    "or_kleene",
    "value_set_array",
]
