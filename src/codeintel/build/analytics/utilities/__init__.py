"""Common utilities for analytics pipelines.

This package provides shared utility functions and helpers used across
analytics modules, including AST utilities and dataset contract helpers.

Modules
-------
- ast: AST traversal and extraction utilities
- datasets: Dataset contracts and row insertion helpers

Example
-------
```python
from codeintel.build.analytics.utilities.ast import call_name, snippet_from_lines
from codeintel.build.analytics.utilities.datasets import (
    get_analytics_dataset_contract,
    insert_analytics_rows,
)


name = call_name(node)


contract = get_analytics_dataset_contract(gateway, "analytics.function_types")
insert_analytics_rows(gateway, contract, rows)
```
"""

from __future__ import annotations

from codeintel.build.analytics.utilities.ast import (
    CallTarget,
    call_name,
    literal_bool,
    literal_int,
    literal_int_sequence,
    literal_str,
    literal_value,
    resolve_call_target,
    safe_unparse,
    snippet_from_lines,
)
from codeintel.build.analytics.utilities.dataframe import to_records
from codeintel.build.analytics.utilities.datasets import (
    AnalyticsPipelineRequest,
    get_analytics_dataset_contract,
    get_function_ast_features_contract,
    insert_analytics_rows,
    run_analytics_pipeline_to_parquet,
)
from codeintel.build.analytics.utilities.persistence import DeleteScope
from codeintel.build.analytics.utilities.type_coercion import (
    int_or_default,
    optional_bool,
    optional_float,
    optional_int,
    optional_str,
)

__all__ = [
    "AnalyticsPipelineRequest",
    "CallTarget",
    "DeleteScope",
    "call_name",
    "get_analytics_dataset_contract",
    "get_function_ast_features_contract",
    "insert_analytics_rows",
    "int_or_default",
    "literal_bool",
    "literal_int",
    "literal_int_sequence",
    "literal_str",
    "literal_value",
    "optional_bool",
    "optional_float",
    "optional_int",
    "optional_str",
    "resolve_call_target",
    "run_analytics_pipeline_to_parquet",
    "safe_unparse",
    "snippet_from_lines",
    "to_records",
]
