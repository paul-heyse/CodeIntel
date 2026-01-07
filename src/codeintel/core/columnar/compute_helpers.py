"""Arrow compute helper wrappers for core use."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc


def call_compute(
    name: str,
    args: list[object],
    *,
    options: pc.FunctionOptions | None = None,
) -> object | None:
    """Call an Arrow compute kernel and return the raw result.

    Parameters
    ----------
    name
        Compute kernel name.
    args
        Positional arguments passed to the kernel.
    options
        Optional compute options for the kernel.

    Returns
    -------
    result : object | None
        Raw compute result, or None when the kernel fails.
    """
    try:
        return pc.call_function(name, args, options=options)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError, ValueError):
        return None


def require_array(result: object | None, *, name: str) -> pa.Array | pa.ChunkedArray:
    """Return result as an array or raise when types do not match.

    Parameters
    ----------
    result
        Compute result to validate.
    name
        Kernel name used for error reporting.

    Returns
    -------
    array : pyarrow.Array | pyarrow.ChunkedArray
        The array result.

    Raises
    ------
    TypeError
        Raised when the compute result is not an array.
    """
    if isinstance(result, (pa.Array, pa.ChunkedArray)):
        return result
    msg = f"Arrow compute {name} did not return an array."
    raise TypeError(msg)


def require_scalar(result: object | None, *, name: str) -> pa.Scalar:
    """Return result as a scalar or raise when types do not match.

    Parameters
    ----------
    result
        Compute result to validate.
    name
        Kernel name used for error reporting.

    Returns
    -------
    scalar : pyarrow.Scalar
        The scalar result.

    Raises
    ------
    TypeError
        Raised when the compute result is not a scalar.
    """
    if isinstance(result, pa.Scalar):
        return result
    msg = f"Arrow compute {name} did not return a scalar."
    raise TypeError(msg)


__all__ = [
    "call_compute",
    "require_array",
    "require_scalar",
]
