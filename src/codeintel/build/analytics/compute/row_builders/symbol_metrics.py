"""Row builders for symbol graph metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from codeintel.build.analytics.compute.row_builders.context import RowBuildContext
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


def build_symbol_module_rows(inputs: SymbolModuleMetricInputs) -> list[SymbolModuleRow]:
    """Construct rows for analytics.symbol_graph_metrics_modules.

    Returns
    -------
    list[SymbolModuleRow]
        Rows ready for insertion into analytics.symbol_graph_metrics_modules.
    """
    return [
        (
            inputs.row_context.repo,
            inputs.row_context.commit,
            module,
            inputs.centrality["betweenness"].get(module, 0.0),
            inputs.centrality["closeness"].get(module, 0.0),
            inputs.centrality["eigenvector"].get(module, 0.0),
            inputs.centrality["harmonic"].get(module, 0.0),
            inputs.structure["core_number"].get(module),
            inputs.structure["constraint"].get(module, 0.0),
            inputs.structure["effective_size"].get(module, 0.0),
            inputs.structure["community_id"].get(module),
            inputs.comp_id.get(module),
            inputs.comp_size.get(module),
            inputs.row_context.created_at,
        )
        for module in inputs.centrality["betweenness"]
    ]


def build_symbol_function_rows(inputs: SymbolFunctionMetricInputs) -> list[SymbolFunctionRow]:
    """Construct rows for analytics.symbol_graph_metrics_functions.

    Returns
    -------
    list[SymbolFunctionRow]
        Rows ready for insertion into analytics.symbol_graph_metrics_functions.
    """
    return [
        (
            inputs.row_context.repo,
            inputs.row_context.commit,
            as_int(node),
            inputs.centrality["betweenness"].get(node, 0.0),
            inputs.centrality["closeness"].get(node, 0.0),
            inputs.centrality["eigenvector"].get(node, 0.0),
            inputs.centrality["harmonic"].get(node, 0.0),
            inputs.structure["core_number"].get(node),
            inputs.structure["constraint"].get(node, 0.0),
            inputs.structure["effective_size"].get(node, 0.0),
            inputs.structure["community_id"].get(node),
            inputs.comp_id.get(node),
            inputs.comp_size.get(node),
            inputs.row_context.created_at,
        )
        for node in inputs.centrality["betweenness"]
    ]


__all__ = [
    "SymbolFunctionMetricInputs",
    "SymbolFunctionRow",
    "SymbolMetricInputs",
    "SymbolModuleMetricInputs",
    "SymbolModuleRow",
    "build_symbol_function_rows",
    "build_symbol_module_rows",
]
