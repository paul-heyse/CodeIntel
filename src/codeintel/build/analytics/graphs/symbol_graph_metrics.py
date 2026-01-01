"""Symbol-coupling graph metrics for modules and functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.build.analytics.compute.row_builders import (
    build_symbol_function_rows,
    build_symbol_module_rows,
)
from codeintel.build.analytics.graphs.symbol_orchestrator import (
    UndirectedMetricsConfig,
    build_undirected_symbol_metric_rows,
)
from codeintel.storage.repositories.functions import FunctionRepository
from codeintel.storage.repositories.modules import ModuleRepository

if TYPE_CHECKING:
    from codeintel.build.graphs.runtime import GraphRuntime, GraphRuntimeOptions
    from codeintel.storage.gateway import StorageGateway


def _get_known_modules(gateway: StorageGateway, repo: str, commit: str) -> set[str]:
    """Load known modules from the database.

    Returns
    -------
    set[str]
        Set of known module names.
    """
    module_repo = ModuleRepository(gateway=gateway, repo=repo, commit=commit)
    return set(module_repo.list_modules())


def _get_known_functions(gateway: StorageGateway, repo: str, commit: str) -> set[int]:
    """Load known function GOIDs from the database.

    Returns
    -------
    set[int]
        Set of known function GOIDs.
    """
    function_repo = FunctionRepository(gateway=gateway, repo=repo, commit=commit)
    return set(function_repo.list_function_goids())


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
    get_graph=lambda rt: rt.ensure_symbol_module_graph(),
    get_known_nodes=_get_known_modules,
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
    get_graph=lambda rt: rt.ensure_symbol_function_graph(),
    get_known_nodes=_get_known_functions,
    filter_node=_filter_function_node,
    build_rows=build_symbol_function_rows,
)


def build_symbol_graph_metrics_module_rows(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    runtime: GraphRuntime | GraphRuntimeOptions | None = None,
) -> list[tuple[object, ...]]:
    """Build analytics.symbol_graph_metrics_modules rows from module symbol coupling.

    Returns
    -------
    list[tuple[object, ...]]
        Row tuples for analytics.symbol_graph_metrics_modules.
    """
    return build_undirected_symbol_metric_rows(
        gateway,
        repo=repo,
        commit=commit,
        config=_MODULE_CONFIG,
        runtime=runtime,
    )


def build_symbol_graph_metrics_function_rows(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    runtime: GraphRuntime | GraphRuntimeOptions | None = None,
) -> list[tuple[object, ...]]:
    """Build analytics.symbol_graph_metrics_functions rows from function symbol coupling.

    Returns
    -------
    list[tuple[object, ...]]
        Row tuples for analytics.symbol_graph_metrics_functions.
    """
    return build_undirected_symbol_metric_rows(
        gateway,
        repo=repo,
        commit=commit,
        config=_FUNCTION_CONFIG,
        runtime=runtime,
    )
