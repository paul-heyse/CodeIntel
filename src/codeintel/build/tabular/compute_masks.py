"""Common PyArrow compute masks for columnar filtering."""

from __future__ import annotations

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


def is_valid_mask(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    """Return a mask for non-null values.

    Returns
    -------
    pa.Array | pa.ChunkedArray
        Boolean mask of valid entries.
    """
    return pc.call_function("is_valid", [values])


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


__all__ = [
    "and_kleene",
    "is_valid_mask",
    "kind_is_function_or_method",
    "language_is_python_or_null",
    "non_empty_string_mask",
    "or_kleene",
]
