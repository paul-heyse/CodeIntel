"""Symbol-coupling graph metrics for modules and functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.analytics.compute.row_builders import (
    build_symbol_function_rows,
    build_symbol_module_rows,
)
from codeintel.analytics.graphs.symbol_orchestrator import (
    UndirectedMetricsConfig,
    compute_undirected_symbol_metrics,
)
from codeintel.analytics.runtime import GraphRuntime, GraphRuntimeOptions
from codeintel.storage.repositories.functions import FunctionRepository
from codeintel.storage.repositories.modules import ModuleRepository

if TYPE_CHECKING:
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
    try:
        return int(node) in known  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


# Configuration for function-level metrics
_FUNCTION_CONFIG: UndirectedMetricsConfig[int] = UndirectedMetricsConfig(
    table_key="analytics.symbol_graph_metrics_functions",
    graph_name="symbol_function_graph",
    get_graph=lambda rt: rt.ensure_symbol_function_graph(),
    get_known_nodes=_get_known_functions,
    filter_node=_filter_function_node,
    build_rows=build_symbol_function_rows,
)


def compute_symbol_graph_metrics_modules(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    runtime: GraphRuntime | GraphRuntimeOptions | None = None,
) -> None:
    """Populate analytics.symbol_graph_metrics_modules from module symbol coupling."""
    compute_undirected_symbol_metrics(
        gateway,
        repo=repo,
        commit=commit,
        config=_MODULE_CONFIG,
        runtime=runtime,
    )


def compute_symbol_graph_metrics_functions(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    runtime: GraphRuntime | GraphRuntimeOptions | None = None,
) -> None:
    """Populate analytics.symbol_graph_metrics_functions from function symbol coupling."""
    compute_undirected_symbol_metrics(
        gateway,
        repo=repo,
        commit=commit,
        config=_FUNCTION_CONFIG,
        runtime=runtime,
    )
