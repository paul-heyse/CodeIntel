"""Columnar streaming protocols and adapters."""

from __future__ import annotations

from codeintel.core.columnar.rows import (
    ColumnarRowBuffer,
    ColumnarRows,
    columnar_buffer_for_table_key,
    columnar_row_count,
)
from codeintel.core.columnar.schema_alignment import align_reader_to_contract
from codeintel.core.columnar.stream import (
    ColumnarStream,
    ColumnarStreamAdapter,
    LazyFrameStream,
    RecordBatchReaderStream,
    SupportsArrowCStream,
    SupportsDataFrameInterop,
    coerce_arrow_reader,
    coerce_arrow_table,
)

__all__ = [
    "ColumnarRowBuffer",
    "ColumnarRows",
    "ColumnarStream",
    "ColumnarStreamAdapter",
    "LazyFrameStream",
    "RecordBatchReaderStream",
    "SupportsArrowCStream",
    "SupportsDataFrameInterop",
    "align_reader_to_contract",
    "columnar_buffer_for_table_key",
    "columnar_row_count",
    "coerce_arrow_reader",
    "coerce_arrow_table",
]
