"""Schema-driven row builders for analytics tables."""

from __future__ import annotations

from codeintel.build.analytics.compute.row_builders.context import RowBuildContext
from codeintel.build.analytics.compute.row_builders.core import (
    row_tuple_for_table,
    rows_to_tuples_for_table,
)
from codeintel.build.analytics.compute.row_builders.graph_metrics import (
    FunctionGraphMetricInputs,
    ModuleGraphMetricInputs,
    build_function_graph_metric_rows,
    build_module_graph_metric_rows,
    component_metadata_from_import_rows,
    merge_component_metadata,
)
from codeintel.build.analytics.compute.row_builders.graph_metrics_ext import (
    FunctionMetricExtInputs,
    ModuleMetricExtInputs,
    build_function_metric_ext_rows,
    build_module_metric_ext_rows,
)
from codeintel.build.analytics.compute.row_builders.subsystem_metrics import (
    SubsystemMetricInputs,
    SubsystemMetricRow,
    build_subsystem_graph_rows,
)
from codeintel.build.analytics.compute.row_builders.symbol_metrics import (
    SymbolFunctionMetricInputs,
    SymbolMetricInputs,
    SymbolModuleMetricInputs,
    build_symbol_function_rows,
    build_symbol_module_rows,
)

__all__ = [
    "FunctionGraphMetricInputs",
    "FunctionMetricExtInputs",
    "ModuleGraphMetricInputs",
    "ModuleMetricExtInputs",
    "RowBuildContext",
    "SubsystemMetricInputs",
    "SubsystemMetricRow",
    "SymbolFunctionMetricInputs",
    "SymbolMetricInputs",
    "SymbolModuleMetricInputs",
    "build_function_graph_metric_rows",
    "build_function_metric_ext_rows",
    "build_module_graph_metric_rows",
    "build_module_metric_ext_rows",
    "build_subsystem_graph_rows",
    "build_symbol_function_rows",
    "build_symbol_module_rows",
    "component_metadata_from_import_rows",
    "merge_component_metadata",
    "row_tuple_for_table",
    "rows_to_tuples_for_table",
]
