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
    ColumnarStreamAdapter,
    LazyFrameStream,
    RecordBatchReaderStream,
)
from codeintel.core.columnar.tabular_adapter import (
    ColumnarStream,
    PolarsExecutionOptions,
    TabularFrame,
    TabularInput,
    TabularRelation,
    coerce_arrow_reader,
    coerce_arrow_table,
    collect_batches,
    collect_lazyframe,
    register_ephemeral,
    to_lazyframe,
    to_record_batch_reader,
    to_relation,
    to_table,
)

__all__ = [
    "ColumnarRowBuffer",
    "ColumnarRows",
    "ColumnarStream",
    "ColumnarStreamAdapter",
    "LazyFrameStream",
    "PolarsExecutionOptions",
    "RecordBatchReaderStream",
    "TabularFrame",
    "TabularInput",
    "TabularRelation",
    "align_reader_to_contract",
    "coerce_arrow_reader",
    "coerce_arrow_table",
    "collect_batches",
    "collect_lazyframe",
    "columnar_buffer_for_table_key",
    "columnar_row_count",
    "extras_policy_from_schema",
    "register_ephemeral",
    "to_lazyframe",
    "to_record_batch_reader",
    "to_relation",
    "to_table",
]
