"""Row builders for analytics graph metrics tables.

These functions construct typed row dictionaries from computed metrics,
ready for insertion into DuckDB tables.

The row builders are pure functions that transform metric data structures
into row formats matching the target table schemas.

Submodules
----------
graph_metrics
    Row builders for core graph metrics (functions and modules).
graph_metrics_ext
    Row builders for extended graph metrics.
subsystem_metrics
    Row builders for subsystem-level graph metrics.
symbol_metrics
    Row builders for symbol graph metrics.
"""

from __future__ import annotations

from codeintel.analytics.compute.row_builders.graph_metrics import (
    FunctionGraphMetricInputs,
    ModuleGraphMetricInputs,
    build_function_graph_metric_rows,
    build_module_graph_metric_rows,
    component_metadata_from_import_table,
    load_symbol_module_edges,
    merge_component_metadata,
)
from codeintel.analytics.compute.row_builders.graph_metrics_ext import (
    FunctionMetricExtInputs,
    ModuleMetricExtInputs,
    build_function_metric_ext_rows,
    build_module_metric_ext_rows,
)
from codeintel.analytics.compute.row_builders.subsystem_metrics import (
    SubsystemMetricInputs,
    SubsystemMetricRow,
    build_subsystem_graph_rows,
)
from codeintel.analytics.compute.row_builders.symbol_metrics import (
    SymbolFunctionMetricInputs,
    SymbolFunctionRow,
    SymbolModuleMetricInputs,
    SymbolModuleRow,
    build_symbol_function_rows,
    build_symbol_module_rows,
)

__all__ = [
    "FunctionGraphMetricInputs",
    "FunctionMetricExtInputs",
    "ModuleGraphMetricInputs",
    "ModuleMetricExtInputs",
    "SubsystemMetricInputs",
    "SubsystemMetricRow",
    "SymbolFunctionMetricInputs",
    "SymbolFunctionRow",
    "SymbolModuleMetricInputs",
    "SymbolModuleRow",
    "build_function_graph_metric_rows",
    "build_function_metric_ext_rows",
    "build_module_graph_metric_rows",
    "build_module_metric_ext_rows",
    "build_subsystem_graph_rows",
    "build_symbol_function_rows",
    "build_symbol_module_rows",
    "component_metadata_from_import_table",
    "load_symbol_module_edges",
    "merge_component_metadata",
]
