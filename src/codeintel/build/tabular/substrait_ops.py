"""Optional Substrait execution helpers for build pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.core.columnar.plan_ops import register_external_plan_runner

try:
    import pyarrow.substrait as _substrait
except ImportError:
    _substrait = None

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    from pyarrow import substrait

    from codeintel.core.columnar.plan_ops import ExternalPlanSpec
    from codeintel.core.columnar.streaming import DatasetScanOptions


def substrait_available() -> bool:
    """Return True when pyarrow.substrait is available.

    Returns
    -------
    bool
        True when Substrait support is available.
    """
    return _substrait is not None


def require_substrait() -> substrait:
    """Return the Substrait module or raise when unavailable.

    Returns
    -------
    pyarrow.substrait
        Substrait module handle.

    Raises
    ------
    RuntimeError
        Raised when Substrait support is unavailable.
    """
    if _substrait is None:
        msg = "pyarrow.substrait is unavailable; install pyarrow with Substrait support."
        raise RuntimeError(msg)
    return _substrait


def run_substrait_plan(plan: bytes | bytearray | memoryview) -> pa.RecordBatchReader:
    """Execute a Substrait plan and return a RecordBatchReader.

    Parameters
    ----------
    plan
        Serialized Substrait plan payload.

    Returns
    -------
    pyarrow.RecordBatchReader
        Record batch reader for the plan results.
    """
    substrait = require_substrait()
    result = substrait.run_query(bytes(plan))
    return _reader_from_result(result)


def substrait_plan_runner(
    *,
    spec: ExternalPlanSpec,
    dataset: ds.Dataset | None,
    filter_expr: ds.Expression | None,
    columns: Sequence[str] | Mapping[str, pc.Expression] | None,
    scan_options: DatasetScanOptions | None,
    use_threads: bool | None,
) -> pa.RecordBatchReader:
    """Execute a Substrait plan via ExternalPlanSpec."""
    _ = (dataset, filter_expr, columns, scan_options, use_threads)
    payload = spec.payload
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return run_substrait_plan(payload)
    msg = "Substrait plan payload must be bytes-like."
    raise TypeError(msg)


def register_substrait_plan_runner(name: str = "substrait") -> None:
    """Register the Substrait external plan runner."""
    register_external_plan_runner(name, substrait_plan_runner)


def _reader_from_result(result: object) -> pa.RecordBatchReader:
    if isinstance(result, pa.RecordBatchReader):
        return result
    if isinstance(result, pa.Table):
        return pa.RecordBatchReader.from_batches(result.schema, result.to_batches())
    msg = f"Unexpected Substrait result type: {type(result)}"
    raise TypeError(msg)


__all__ = [
    "register_substrait_plan_runner",
    "require_substrait",
    "run_substrait_plan",
    "substrait_available",
    "substrait_plan_runner",
]
