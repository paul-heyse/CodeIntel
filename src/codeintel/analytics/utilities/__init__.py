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
from codeintel.analytics.utilities.ast import call_name, snippet_from_lines
from codeintel.analytics.utilities.datasets import (
    get_analytics_dataset_contract,
    insert_analytics_rows,
)


name = call_name(node)


contract = get_analytics_dataset_contract(gateway, "analytics.function_metrics")
insert_analytics_rows(gateway, contract, rows)
```
"""

from __future__ import annotations

from codeintel.analytics.utilities.ast import (
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
from codeintel.analytics.utilities.datasets import (
    AnalyticsDatasetContract,
    build_analytics_dataset_contracts,
    get_analytics_dataset_contract,
    get_function_ast_features_contract,
    insert_analytics_rows,
)

__all__ = [
    "AnalyticsDatasetContract",
    "CallTarget",
    "build_analytics_dataset_contracts",
    "call_name",
    "get_analytics_dataset_contract",
    "get_function_ast_features_contract",
    "insert_analytics_rows",
    "literal_bool",
    "literal_int",
    "literal_int_sequence",
    "literal_str",
    "literal_value",
    "resolve_call_target",
    "safe_unparse",
    "snippet_from_lines",
]
