"""Columnar streaming protocols and adapters."""

from __future__ import annotations

from codeintel.core.columnar.stream import (
    ColumnarStream,
    ColumnarStreamAdapter,
    LazyFrameStream,
    RecordBatchReaderStream,
)

__all__ = [
    "ColumnarStream",
    "ColumnarStreamAdapter",
    "LazyFrameStream",
    "RecordBatchReaderStream",
]
