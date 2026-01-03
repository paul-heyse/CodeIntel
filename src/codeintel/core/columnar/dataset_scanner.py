"""Arrow dataset scanner utilities.

Deprecated: use ``codeintel.core.columnar.streaming``.
"""

from __future__ import annotations

from codeintel.core.columnar.streaming import (
    empty_reader_from_schema,
    sample_reader,
    scan_dataset_lazyframe,
    scan_dataset_reader,
)

__all__ = [
    "empty_reader_from_schema",
    "sample_reader",
    "scan_dataset_lazyframe",
    "scan_dataset_reader",
]
