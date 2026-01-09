"""Row builders for symbol graph metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from codeintel.build.analytics.compute.row_builders.context import RowBuildContext
from codeintel.build.analytics.compute.row_builders.core import buffer_for_table
from codeintel.core.columnar.rows import ColumnarRowBuffer
from codeintel.core.data_models.ids import as_int

if TYPE_CHECKING:
    from collections.abc import Mapping

SymbolModuleRow = tuple[Any, ...]
SymbolFunctionRow = tuple[Any, ...]


@dataclass(frozen=True)
class SymbolMetricInputs[TNode]:
    """Generic inputs for symbol graph metrics computation.

    Type Parameters
    ---------------
    TNode
        Node type for graph nodes (e.g., str for modules, int for functions).
    """

    row_context: RowBuildContext
    centrality: Mapping[str, Mapping[TNode, float]]
    structure: Mapping[str, Mapping[TNode, float | int]]
    comp_id: Mapping[TNode, int]
    comp_size: Mapping[TNode, int]


# Convenience type aliases for clearer function signatures
SymbolModuleMetricInputs = SymbolMetricInputs[str]
SymbolFunctionMetricInputs = SymbolMetricInputs[int]


def build_symbol_module_rows(inputs: SymbolModuleMetricInputs) -> ColumnarRowBuffer:
    """Construct rows for analytics.symbol_graph_metrics_modules.

    Returns
    -------
    ColumnarRowBuffer
        Buffer containing rows ready for analytics.symbol_graph_metrics_modules.
    """
    buffer = buffer_for_table("analytics.symbol_graph_metrics_modules")
    for module in inputs.centrality["betweenness"]:
        buffer.append(
            {
                "repo": inputs.row_context.repo,
                "commit": inputs.row_context.commit,
                "module": module,
                "symbol_betweenness": inputs.centrality["betweenness"].get(module, 0.0),
                "symbol_closeness": inputs.centrality["closeness"].get(module, 0.0),
                "symbol_eigenvector": inputs.centrality["eigenvector"].get(module, 0.0),
                "symbol_harmonic": inputs.centrality["harmonic"].get(module, 0.0),
                "symbol_k_core": inputs.structure["core_number"].get(module),
                "symbol_constraint": inputs.structure["constraint"].get(module, 0.0),
                "symbol_effective_size": inputs.structure["effective_size"].get(module, 0.0),
                "symbol_community_id": inputs.structure["community_id"].get(module),
                "symbol_component_id": inputs.comp_id.get(module),
                "symbol_component_size": inputs.comp_size.get(module),
                "created_at": inputs.row_context.created_at,
            }
        )
    return buffer


def build_symbol_function_rows(inputs: SymbolFunctionMetricInputs) -> ColumnarRowBuffer:
    """Construct rows for analytics.symbol_graph_metrics_functions.

    Returns
    -------
    ColumnarRowBuffer
        Buffer containing rows ready for analytics.symbol_graph_metrics_functions.
    """
    buffer = buffer_for_table("analytics.symbol_graph_metrics_functions")
    for node in inputs.centrality["betweenness"]:
        buffer.append(
            {
                "repo": inputs.row_context.repo,
                "commit": inputs.row_context.commit,
                "function_goid_h128": as_int(node),
                "symbol_betweenness": inputs.centrality["betweenness"].get(node, 0.0),
                "symbol_closeness": inputs.centrality["closeness"].get(node, 0.0),
                "symbol_eigenvector": inputs.centrality["eigenvector"].get(node, 0.0),
                "symbol_harmonic": inputs.centrality["harmonic"].get(node, 0.0),
                "symbol_k_core": inputs.structure["core_number"].get(node),
                "symbol_constraint": inputs.structure["constraint"].get(node, 0.0),
                "symbol_effective_size": inputs.structure["effective_size"].get(node, 0.0),
                "symbol_community_id": inputs.structure["community_id"].get(node),
                "symbol_component_id": inputs.comp_id.get(node),
                "symbol_component_size": inputs.comp_size.get(node),
                "created_at": inputs.row_context.created_at,
            }
        )
    return buffer


__all__ = [
    "SymbolFunctionMetricInputs",
    "SymbolFunctionRow",
    "SymbolMetricInputs",
    "SymbolModuleMetricInputs",
    "SymbolModuleRow",
    "build_symbol_function_rows",
    "build_symbol_module_rows",
]
