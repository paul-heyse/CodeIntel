"""Columnar streaming protocols and adapters."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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

    _TYPE_CHECKING_EXPORTS = (
        ColumnarRowBuffer,
        ColumnarRows,
        columnar_buffer_for_table_key,
        columnar_row_count,
        align_reader_to_contract,
        extras_policy_from_schema,
        ColumnarStream,
        ColumnarStreamAdapter,
        LazyFrameStream,
        RecordBatchReaderStream,
        coerce_arrow_reader,
        coerce_arrow_table,
    )

_EXPORTS: dict[str, tuple[str, str]] = {
    "ColumnarRowBuffer": ("codeintel.core.columnar.rows", "ColumnarRowBuffer"),
    "ColumnarRows": ("codeintel.core.columnar.rows", "ColumnarRows"),
    "columnar_buffer_for_table_key": (
        "codeintel.core.columnar.rows",
        "columnar_buffer_for_table_key",
    ),
    "columnar_row_count": ("codeintel.core.columnar.rows", "columnar_row_count"),
    "align_reader_to_contract": (
        "codeintel.core.columnar.schema_alignment",
        "align_reader_to_contract",
    ),
    "extras_policy_from_schema": (
        "codeintel.core.columnar.schema_alignment",
        "extras_policy_from_schema",
    ),
    "ColumnarStream": ("codeintel.core.columnar.stream", "ColumnarStream"),
    "ColumnarStreamAdapter": ("codeintel.core.columnar.stream", "ColumnarStreamAdapter"),
    "LazyFrameStream": ("codeintel.core.columnar.stream", "LazyFrameStream"),
    "RecordBatchReaderStream": ("codeintel.core.columnar.stream", "RecordBatchReaderStream"),
    "coerce_arrow_reader": ("codeintel.core.columnar.stream", "coerce_arrow_reader"),
    "coerce_arrow_table": ("codeintel.core.columnar.stream", "coerce_arrow_table"),
}


def __getattr__(name: str) -> object:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = tuple(_EXPORTS)
