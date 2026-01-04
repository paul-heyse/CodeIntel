"""Common PyArrow compute masks for columnar filtering."""

from __future__ import annotations

from collections.abc import Sequence

import pyarrow as pa
import pyarrow.compute as pc


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
    return pc.call_function("equal", [left, right])


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
    return pc.call_function("not_equal", [left, right])


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
    options = pc.SetLookupOptions(value_set=value_set)
    return pc.call_function("is_in", [values], options=options)


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


def language_is_python_or_null(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    """Return a mask for Python language markers or NULLs.

    Returns
    -------
    pa.Array | pa.ChunkedArray
        Boolean mask for Python or NULL values.
    """
    is_null = pc.call_function("is_null", [values])
    is_python = pc.call_function("equal", [values, pc.scalar("python")])
    return or_kleene(is_null, is_python)


def kind_is_function_or_method(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    """Return a mask for function or method kinds.

    Returns
    -------
    pa.Array | pa.ChunkedArray
        Boolean mask for function/method kinds.
    """
    is_function = pc.call_function("equal", [values, pc.scalar("function")])
    is_method = pc.call_function("equal", [values, pc.scalar("method")])
    return or_kleene(is_function, is_method)


def node_type_is_function(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    """Return a mask for Python function AST node types.

    Returns
    -------
    pa.Array | pa.ChunkedArray
        Boolean mask for function/async function node types.
    """
    is_function = pc.call_function("equal", [values, pc.scalar("FunctionDef")])
    is_async = pc.call_function("equal", [values, pc.scalar("AsyncFunctionDef")])
    return or_kleene(is_function, is_async)


__all__ = [
    "and_kleene",
    "bit_wise_and",
    "equal_mask",
    "invert_mask",
    "is_in_mask",
    "is_null_mask",
    "is_valid_mask",
    "kind_is_function_or_method",
    "language_is_python_or_null",
    "node_type_is_function",
    "non_empty_string_mask",
    "not_equal_mask",
    "or_kleene",
]
