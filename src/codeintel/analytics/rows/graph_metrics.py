"""Typed row helpers for graph metrics tables."""

from __future__ import annotations

from codeintel.storage.rows import (
    GraphMetricsFunctionsRow,
    GraphMetricsModulesRow,
    graph_metrics_functions_row_to_tuple,
    graph_metrics_modules_row_to_tuple,
)

FunctionGraphMetricsRow = GraphMetricsFunctionsRow
ModuleGraphMetricsRow = GraphMetricsModulesRow


def function_graph_metrics_row_to_tuple(
    row: FunctionGraphMetricsRow,
) -> tuple[object, ...]:
    """
    Serialize a function graph metrics row into INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values ordered for analytics.graph_metrics_functions.
    """
    return graph_metrics_functions_row_to_tuple(row)


def module_graph_metrics_row_to_tuple(row: ModuleGraphMetricsRow) -> tuple[object, ...]:
    """
    Serialize a module graph metrics row into INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values ordered for analytics.graph_metrics_modules.
    """
    return graph_metrics_modules_row_to_tuple(row)


__all__ = [
    "FunctionGraphMetricsRow",
    "ModuleGraphMetricsRow",
    "function_graph_metrics_row_to_tuple",
    "module_graph_metrics_row_to_tuple",
]
