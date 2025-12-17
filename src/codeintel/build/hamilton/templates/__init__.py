"""Template Hamilton modules for the build DAG.

This package contains **template** Hamilton nodes that provide a fallback
implementation for all build targets. Native target modules override these
templates via Hamilton's module override semantics.

The package also provides reusable subDAG pipeline templates:
- ``rows_pipeline``: For row-oriented DuckDB targets
- ``tool_pipeline``: For tool-invocation artifact targets
"""

from __future__ import annotations

from types import ModuleType

from codeintel.build.hamilton.templates.all_targets import get_template_module
from codeintel.build.hamilton.templates.rows_pipeline import (
    record as rows_record,
)
from codeintel.build.hamilton.templates.rows_pipeline import (
    rows_to_save,
)
from codeintel.build.hamilton.templates.tool_pipeline import (
    record as tool_record,
)
from codeintel.build.hamilton.templates.tool_pipeline import (
    tool_output_to_save,
)

__all__ = [
    "ModuleType",
    "get_template_module",
    "rows_record",
    "rows_to_save",
    "tool_output_to_save",
    "tool_record",
]
