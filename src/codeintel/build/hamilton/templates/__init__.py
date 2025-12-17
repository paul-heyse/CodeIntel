"""Template Hamilton modules for the build DAG.

This package contains **template** Hamilton nodes that provide a fallback
implementation for all build targets. Native target modules override these
templates via Hamilton's module override semantics.

The package also provides reusable subDAG pipeline templates:

- ``materialize_template``: Unified materialization helpers (Patterns B/D)
- ``multi_table_pipeline``: For multi-table row materializations (Pattern C)
- ``tool_pipeline``: For tool-invocation artifact targets (Pattern F)

Row Conversion Helpers
----------------------
The ``rows_pipeline`` module also exports row conversion utilities:

- ``row_to_tuple``: Convert a mapping row to a tuple in column order
- ``rows_to_tuples``: Convert a sequence of mapping rows to tuples
"""

from __future__ import annotations

from types import ModuleType

from codeintel.build.hamilton.templates.all_targets import get_template_module
from codeintel.build.hamilton.templates.materialize_template import (
    ComputeResult,
    duckdb_record,
    executor_materialize,
    executor_record,
    row_to_tuple,
    rows_to_save,
    rows_to_tuples,
)
from codeintel.build.hamilton.templates.multi_table_pipeline import (
    create_row_extractor,
    multi_table_record,
)
from codeintel.build.hamilton.templates.tool_pipeline import (
    record as tool_record,
)
from codeintel.build.hamilton.templates.tool_pipeline import (
    tool_output_to_save,
)

__all__ = [
    "ComputeResult",
    "ModuleType",
    "create_row_extractor",
    "duckdb_record",
    "executor_materialize",
    "executor_record",
    "get_template_module",
    "multi_table_record",
    "row_to_tuple",
    "rows_to_save",
    "rows_to_tuples",
    "tool_output_to_save",
    "tool_record",
]
