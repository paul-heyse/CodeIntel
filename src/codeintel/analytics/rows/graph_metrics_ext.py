"""Typed row helpers for extended graph metrics tables."""

from __future__ import annotations

from codeintel.storage.rows import (
    GraphMetricsFunctionsExtRow,
    GraphMetricsModulesExtRow,
    graph_metrics_functions_ext_row_to_tuple,
    graph_metrics_modules_ext_row_to_tuple,
)

FunctionGraphMetricsExtRow = GraphMetricsFunctionsExtRow
ModuleGraphMetricsExtRow = GraphMetricsModulesExtRow


def function_graph_metrics_ext_row_to_tuple(
    row: FunctionGraphMetricsExtRow,
) -> tuple[object, ...]:
    """
    Serialize a function graph metrics ext row into INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values ordered for analytics.graph_metrics_functions_ext.
    """
    return graph_metrics_functions_ext_row_to_tuple(row)


def module_graph_metrics_ext_row_to_tuple(
    row: ModuleGraphMetricsExtRow,
) -> tuple[object, ...]:
    """
    Serialize a module graph metrics ext row into INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values ordered for analytics.graph_metrics_modules_ext.
    """
    return graph_metrics_modules_ext_row_to_tuple(row)


__all__ = [
    "FunctionGraphMetricsExtRow",
    "ModuleGraphMetricsExtRow",
    "function_graph_metrics_ext_row_to_tuple",
    "module_graph_metrics_ext_row_to_tuple",
]
