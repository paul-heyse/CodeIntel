"""Shared JSONL encoding utilities for serving exports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.core.columnar.arrowdsl import ExecutionContext, ExecutionPlan, run_pipeline
from codeintel.core.exports.serialization import coerce_export_row
from codeintel.storage.query_results import records_from_arrow_batch

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping

    from pyarrow import RecordBatch, RecordBatchReader

    from codeintel.core.columnar.finalize_ops import FinalizeResult, FinalizeSpec


try:
    import orjson

    _ORJSON = orjson
except ImportError:
    _ORJSON = None

try:
    import msgspec

    _MSG_ENCODER: msgspec.json.Encoder | None = msgspec.json.Encoder()
except ImportError:
    _MSG_ENCODER = None


@dataclass(frozen=True, slots=True)
class NdjsonBatchOptions:
    """Options for NDJSON batch encoding helpers."""

    cancel_check: Callable[[], None] | None = None
    batch_hook: Callable[[RecordBatch], None] | None = None
    finalize_spec: FinalizeSpec | None = None
    finalize_hook: Callable[[FinalizeResult], None] | None = None
    execution_ctx: ExecutionContext | None = None


def encode_ndjson_line(row: Mapping[str, object]) -> bytes:
    """Encode a single row as a UTF-8 JSONL line.

    Returns
    -------
    bytes
        Serialized JSONL line with a trailing newline.
    """
    payload_row = coerce_export_row(row)
    if _ORJSON is not None:
        return _ORJSON.dumps(payload_row, option=_ORJSON.OPT_APPEND_NEWLINE)
    if _MSG_ENCODER is not None:
        return _MSG_ENCODER.encode(payload_row) + b"\n"
    payload = json.dumps(
        payload_row,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return payload.encode("utf-8") + b"\n"


def iter_ndjson_bytes(rows: Iterable[Mapping[str, object]]) -> Iterator[bytes]:
    """Yield rows as UTF-8 JSONL byte lines.

    Yields
    ------
    bytes
        Serialized JSONL line with a trailing newline.
    """
    for row in rows:
        yield encode_ndjson_line(row)


def iter_ndjson_bytes_from_batches(
    batches: Iterable[RecordBatch],
    *,
    options: NdjsonBatchOptions | None = None,
) -> Iterator[bytes]:
    """Yield Arrow record batches as UTF-8 JSONL byte chunks.

    Parameters
    ----------
    batches
        Record batch iterable to serialize as newline-delimited JSON.
    options
        Optional batch encoding options for cancellation, hooks, and finalize settings.

    Yields
    ------
    bytes
        UTF-8 JSONL chunks for each record batch.

    Notes
    -----
    This function uses shared row coercion to ensure consistent export encoding.
    """
    resolved = options or NdjsonBatchOptions()
    cancel_check = resolved.cancel_check
    batch_hook = resolved.batch_hook
    finalize_spec = resolved.finalize_spec
    finalize_hook = resolved.finalize_hook
    execution_ctx = resolved.execution_ctx
    for batch in batches:
        if cancel_check is not None:
            cancel_check()
        if finalize_spec is None:
            if batch_hook is not None:
                batch_hook(batch)
            payload = _batch_to_ndjson_bytes(batch, columns=batch.schema.names)
            if payload:
                yield payload
            continue
        finalized_batches, finalize_result = _finalize_batches(
            batch,
            finalize_spec=finalize_spec,
            execution_ctx=execution_ctx,
        )
        if finalize_hook is not None:
            finalize_hook(finalize_result)
        for finalized in finalized_batches:
            if batch_hook is not None:
                batch_hook(finalized)
            payload = _batch_to_ndjson_bytes(finalized, columns=finalized.schema.names)
            if payload:
                yield payload


def iter_ndjson_bytes_from_reader(
    reader: RecordBatchReader,
    *,
    options: NdjsonBatchOptions | None = None,
) -> Iterator[bytes]:
    """Yield Arrow record batches from a reader as UTF-8 JSONL byte chunks.

    Parameters
    ----------
    reader
        Record batch reader to serialize as newline-delimited JSON.
    options
        Optional batch encoding options for cancellation, hooks, and finalize settings.

    Yields
    ------
    bytes
        UTF-8 JSONL chunks for each record batch.
    """
    yield from iter_ndjson_bytes_from_batches(reader, options=options)


def _batch_to_ndjson_bytes(batch: RecordBatch, *, columns: list[str]) -> bytes:
    rows = records_from_arrow_batch(batch, columns=columns)
    if not rows:
        return b""
    return b"".join(encode_ndjson_line(row) for row in rows)


def _finalize_batches(
    batch: RecordBatch,
    *,
    finalize_spec: FinalizeSpec,
    execution_ctx: ExecutionContext | None,
) -> tuple[list[RecordBatch], FinalizeResult]:
    table = pa.Table.from_batches([batch], schema=batch.schema)
    resolved_ctx = execution_ctx or ExecutionContext()
    result = run_pipeline(
        plan=ExecutionPlan.from_table(table),
        finalize=finalize_spec,
        ctx=resolved_ctx,
    )
    if result.good.num_rows == 0:
        return [], result
    batches = result.good.to_batches(max_chunksize=batch.num_rows)
    if not batches:
        return [], result
    return batches, result


__all__ = [
    "NdjsonBatchOptions",
    "encode_ndjson_line",
    "iter_ndjson_bytes",
    "iter_ndjson_bytes_from_batches",
    "iter_ndjson_bytes_from_reader",
]
