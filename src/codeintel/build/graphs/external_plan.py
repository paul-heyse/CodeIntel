"""External plan runner for rustworkx-backed graph computations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import pyarrow as pa

from codeintel.core.columnar.conversion import table_to_reader
from codeintel.core.columnar.plan_ops import (
    ExternalPlanRequest,
    ExternalPlanSpec,
    register_external_plan_runner,
    run_external_plan,
)
from codeintel.core.columnar.rows import ColumnarRowBuffer, columnar_batch_collector_for_table_key

RustworkxPlanBuilder = Callable[
    ...,
    pa.Table | pa.RecordBatchReader | ColumnarRowBuffer,
]


@dataclass(frozen=True, slots=True)
class RustworkxPlanPayload:
    """Payload for rustworkx external plan execution."""

    builder: RustworkxPlanBuilder
    args: tuple[object, ...] = ()
    kwargs: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)


def rustworkx_plan_runner(
    *,
    request: ExternalPlanRequest,
) -> pa.RecordBatchReader:
    """Execute rustworkx plans via ExternalPlanSpec payloads.

    Parameters
    ----------
    request
        External plan request containing rustworkx payload metadata.

    Returns
    -------
    pyarrow.RecordBatchReader
        Record batch reader for plan results.

    Raises
    ------
    TypeError
        Raised when the payload is not a RustworkxPlanPayload.
    """
    _ = (
        request.dataset,
        request.filter_expr,
        request.columns,
        request.scan_options,
        request.use_threads,
    )
    payload = request.spec.payload
    if not isinstance(payload, RustworkxPlanPayload):
        msg = "Rustworkx plan payload must be RustworkxPlanPayload."
        raise TypeError(msg)
    return _reader_from_result(payload.builder(*payload.args, **dict(payload.kwargs)))


def register_rustworkx_plan_runner(name: str = "rustworkx") -> None:
    """Register the rustworkx external plan runner.

    Parameters
    ----------
    name
        Engine name used for rustworkx external plans.
    """
    register_external_plan_runner(name, rustworkx_plan_runner)


def run_rustworkx_external_plan(
    *,
    builder: RustworkxPlanBuilder,
    args: tuple[object, ...] = (),
    kwargs: Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
    use_threads: bool | None = None,
) -> pa.RecordBatchReader:
    """Execute a rustworkx external plan and return a reader.

    Parameters
    ----------
    builder
        Callable that builds a rustworkx-backed result.
    args
        Positional arguments forwarded to the builder.
    kwargs
        Keyword arguments forwarded to the builder.
    metadata
        Metadata attached to the external plan request.
    use_threads
        Whether to enable threaded execution for the plan.

    Returns
    -------
    pyarrow.RecordBatchReader
        Record batch reader for the plan results.
    """
    payload = RustworkxPlanPayload(
        builder=builder,
        args=args,
        kwargs=kwargs or {},
        metadata=metadata or {},
    )
    request = ExternalPlanRequest(
        spec=ExternalPlanSpec(
            engine="rustworkx",
            payload=payload,
            metadata=payload.metadata,
        ),
        dataset=None,
        filter_expr=None,
        columns=None,
        scan_options=None,
        use_threads=use_threads,
    )
    return run_external_plan(request)


def _reader_from_result(
    result: pa.Table | pa.RecordBatchReader | ColumnarRowBuffer,
) -> pa.RecordBatchReader:
    """Coerce rustworkx outputs into a record batch reader.

    Parameters
    ----------
    result
        Rustworkx plan output.

    Returns
    -------
    pyarrow.RecordBatchReader
        Reader view of the rustworkx output.

    Raises
    ------
    TypeError
        Raised when the plan output is not supported.
    """
    if isinstance(result, pa.RecordBatchReader):
        return result
    if isinstance(result, ColumnarRowBuffer):
        collector = columnar_batch_collector_for_table_key(result.table_key)
        collector.extend(result)
        return collector.to_reader()
    if isinstance(result, pa.Table):
        return table_to_reader(result, batch_size=None)
    msg = f"Unexpected rustworkx plan result type: {type(result)}"
    raise TypeError(msg)


__all__ = [
    "RustworkxPlanBuilder",
    "RustworkxPlanPayload",
    "register_rustworkx_plan_runner",
    "run_rustworkx_external_plan",
    "rustworkx_plan_runner",
]
