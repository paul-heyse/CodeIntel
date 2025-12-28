"""Columnar streaming protocols and adapters."""

from __future__ import annotations

from codeintel.core.columnar.rows import (
    ColumnarRowBuffer,
    ColumnarRows,
    columnar_buffer_for_table_key,
    columnar_row_count,
)
from codeintel.core.columnar.stream import (
    ColumnarStream,
    ColumnarStreamAdapter,
    LazyFrameStream,
    RecordBatchReaderStream,
)

__all__ = [
    "ColumnarRowBuffer",
    "ColumnarRows",
    "ColumnarStream",
    "ColumnarStreamAdapter",
    "LazyFrameStream",
    "RecordBatchReaderStream",
    "columnar_buffer_for_table_key",
    "columnar_row_count",
]
