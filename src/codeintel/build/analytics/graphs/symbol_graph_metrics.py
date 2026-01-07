"""Symbol-coupling graph metrics for modules and functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.build.analytics.compute.row_builders import (
    build_symbol_function_rows,
    build_symbol_module_rows,
)
from codeintel.build.analytics.graphs.symbol_orchestrator import (
    UndirectedMetricInputs,
    UndirectedMetricsConfig,
    build_undirected_symbol_metric_rows,
)
from codeintel.build.graphs.builders import (
    build_symbol_function_graph as _build_symbol_function_graph,
)
from codeintel.build.graphs.builders import (
    build_symbol_module_graph as _build_symbol_module_graph,
)
from codeintel.build.graphs.rx.algos import GraphInput

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from codeintel.build.graphs.runtime import GraphRuntimeOptions


def build_symbol_module_graph(
    symbol_use_edges: Iterable[Mapping[str, object]],
    module_by_path: Mapping[str, str],
) -> GraphInput:
    """Build an undirected weighted symbol-module graph from use edges.

    Returns
    -------
    GraphInput
        Undirected graph linking modules by symbol coupling.
    """
    return _build_symbol_module_graph(symbol_use_edges, module_by_path)


def build_symbol_function_graph(
    symbol_use_edges: Iterable[Mapping[str, object]],
) -> GraphInput:
    """Build an undirected weighted symbol-function graph from use edges.

    Returns
    -------
    GraphInput
        Undirected graph linking functions by symbol coupling.
    """
    return _build_symbol_function_graph(symbol_use_edges)


def _parse_int_node(node: object) -> int | None:
    parsed: int | None = None
    if isinstance(node, bool):
        parsed = int(node)
    elif isinstance(node, int):
        parsed = node
    elif isinstance(node, float):
        parsed = int(node) if node.is_integer() else None
    elif isinstance(node, str):
        value = node.strip()
        if value:
            try:
                parsed = int(value)
            except ValueError:
                parsed = None
    return parsed


# Configuration for module-level metrics
_MODULE_CONFIG: UndirectedMetricsConfig[str] = UndirectedMetricsConfig(
    table_key="analytics.symbol_graph_metrics_modules",
    graph_name="symbol_module_graph",
    filter_node=lambda node, known: str(node) in known,
    build_rows=build_symbol_module_rows,
)


def _filter_function_node(node: object, known: set[int]) -> bool:
    """Check if a function node should be included in the graph.

    Returns
    -------
    bool
        True if the node is in the set of known functions.
    """
    parsed = _parse_int_node(node)
    return parsed is not None and parsed in known


# Configuration for function-level metrics
_FUNCTION_CONFIG: UndirectedMetricsConfig[int] = UndirectedMetricsConfig(
    table_key="analytics.symbol_graph_metrics_functions",
    graph_name="symbol_function_graph",
    filter_node=_filter_function_node,
    build_rows=build_symbol_function_rows,
)


def build_symbol_graph_metrics_module_rows(
    *,
    repo: str,
    commit: str,
    graph: GraphInput,
    known_modules: set[str] | None = None,
    runtime: GraphRuntimeOptions | None = None,
) -> list[tuple[object, ...]]:
    """Build analytics.symbol_graph_metrics_modules rows from module symbol coupling.

    Returns
    -------
    list[tuple[object, ...]]
        Row tuples for analytics.symbol_graph_metrics_modules.
    """
    return build_undirected_symbol_metric_rows(
        inputs=UndirectedMetricInputs(
            repo=repo,
            commit=commit,
            graph=graph,
            known_nodes=known_modules,
            runtime=runtime,
        ),
        config=_MODULE_CONFIG,
    )


def build_symbol_graph_metrics_function_rows(
    *,
    repo: str,
    commit: str,
    graph: GraphInput,
    known_functions: set[int] | None = None,
    runtime: GraphRuntimeOptions | None = None,
) -> list[tuple[object, ...]]:
    """Build analytics.symbol_graph_metrics_functions rows from function symbol coupling.

    Returns
    -------
    list[tuple[object, ...]]
        Row tuples for analytics.symbol_graph_metrics_functions.
    """
    return build_undirected_symbol_metric_rows(
        inputs=UndirectedMetricInputs(
            repo=repo,
            commit=commit,
            graph=graph,
            known_nodes=known_functions,
            runtime=runtime,
        ),
        config=_FUNCTION_CONFIG,
    )
