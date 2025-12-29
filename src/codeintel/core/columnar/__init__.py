"""Columnar streaming protocols and adapters."""

from __future__ import annotations

from codeintel.core.columnar.rows import (
    ColumnarRowBuffer,
    ColumnarRows,
    columnar_buffer_for_table_key,
    columnar_row_count,
)
from codeintel.core.columnar.schema_alignment import (
    align_reader_to_contract,
    extras_policy_from_schema,
)
from codeintel.core.columnar.stream import (
    ColumnarStream,
    ColumnarStreamAdapter,
    LazyFrameStream,
    RecordBatchReaderStream,
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
    "align_reader_to_contract",
    "coerce_arrow_reader",
    "coerce_arrow_table",
    "columnar_buffer_for_table_key",
    "columnar_row_count",
    "extras_policy_from_schema",
]
