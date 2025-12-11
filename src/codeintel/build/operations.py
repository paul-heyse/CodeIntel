"""Operation to build targets mapping.

This module provides the bridge between serving operations and the build system,
mapping operation requirements (datasets, graphs) to build targets.

Operations declare their requirements as:
- ``required_datasets``: table keys like "core.goids", "analytics.function_metrics"
- ``required_graphs``: runtime names like "callgraph", "importgraph"

The build system operates on targets like "ast", "call_graph", "function_metrics".
This module provides the mapping between these two views.

Note on Circular Import Avoidance
---------------------------------
This module uses lazy loading patterns (_LazyRegistry, _LazyCatalog) to avoid
circular imports. The build.registry imports this module's parent, and serving
operations may import from build, so we defer loading until functions are called.

Usage
-----
>>> from codeintel.build.operations import get_targets_for_operation
>>> targets = get_targets_for_operation("function.summary")
>>> targets.required_targets
frozenset({'call_graph', 'goids', 'function_metrics'})
>>> targets.graph_targets
frozenset({'call_graph'})
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from codeintel.build.targets import OutputTarget
    from codeintel.serving.operations.catalog import Operation

log = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass(frozen=True)
class OperationTargets:
    """Build targets required for a serving operation.

    This dataclass bridges the operation's declared requirements (datasets, graphs)
    to the build system's targets, enabling the build system to compute exactly
    what's needed for an operation.

    Attributes
    ----------
    operation_id
        Operation identifier (e.g., "function.summary").
    required_targets
        All build targets that must be computed for this operation.
        Union of graph_targets and data_targets.
    graph_targets
        Targets providing the required graph runtimes.
    data_targets
        Targets providing the required datasets.
    """

    operation_id: str
    required_targets: frozenset[str]
    graph_targets: frozenset[str]
    data_targets: frozenset[str]


# =============================================================================
# Lazy Module Access (avoids circular imports)
# =============================================================================


class _LazyRegistry:
    """Lazy accessor for build registry to avoid circular imports."""

    _all_targets: tuple[OutputTarget, ...] | None = None

    @classmethod
    def get_all_targets(cls) -> tuple[OutputTarget, ...]:
        """Get all registered targets.

        Returns
        -------
        tuple[OutputTarget, ...]
            All registered build targets.
        """
        if cls._all_targets is None:
            registry = importlib.import_module("codeintel.build.registry")
            cls._all_targets = registry.ALL_TARGETS
        # Return value - the if-block above guarantees _all_targets is set
        result = cls._all_targets
        if result is None:
            # This should never happen - defensive guard for type checker
            return ()
        return result


class _LazyCatalog:
    """Lazy accessor for operation catalog to avoid circular imports."""

    _catalog_module: ModuleType | None = None

    @classmethod
    def _get_module(cls) -> ModuleType:
        """Get the catalog module.

        Returns
        -------
        ModuleType
            The catalog module.
        """
        if cls._catalog_module is None:
            cls._catalog_module = importlib.import_module("codeintel.serving.operations.catalog")
        return cls._catalog_module

    @classmethod
    def get_operation(cls, op_id: str) -> Operation | None:
        """Look up an operation by ID.

        Parameters
        ----------
        op_id
            Operation identifier.

        Returns
        -------
        Operation | None
            The operation if found.
        """
        module = cls._get_module()
        return module.get_operation(op_id)

    @classmethod
    def iter_operations(cls) -> Iterator[Operation]:
        """Iterate all registered operations.

        Yields
        ------
        Operation
            Each registered operation.
        """
        module = cls._get_module()
        yield from module.iter_operations()


# =============================================================================
# Index Building
# =============================================================================


def _build_table_to_target_index() -> dict[str, str]:
    """Build mapping from table_key to target name.

    Iterates all registered targets and maps each table they produce
    to the target name.

    Returns
    -------
    dict[str, str]
        Mapping from table_key (e.g., "core.goids") to target name (e.g., "goids").
    """
    all_targets = _LazyRegistry.get_all_targets()

    index: dict[str, str] = {}
    for target in all_targets:
        for table in target.table_keys:
            index[table] = target.name
    return index


def _build_graph_to_target_index() -> dict[str, str]:
    """Build mapping from graph runtime name to target name.

    Maps the graph runtime names used by operations (e.g., "callgraph")
    to the build targets that produce them (e.g., "call_graph").

    Returns
    -------
    dict[str, str]
        Mapping from graph runtime name to target name.
    """
    return {
        "callgraph": "call_graph",
        "importgraph": "import_graph",
    }


@lru_cache(maxsize=1)
def _get_table_index() -> dict[str, str]:
    """Get the table-to-target index, building if needed.

    Returns
    -------
    dict[str, str]
        Mapping from table_key to target name.
    """
    return _build_table_to_target_index()


@lru_cache(maxsize=1)
def _get_graph_index() -> dict[str, str]:
    """Get the graph-to-target index, building if needed.

    Returns
    -------
    dict[str, str]
        Mapping from graph runtime name to target name.
    """
    return _build_graph_to_target_index()


# =============================================================================
# Target Resolution
# =============================================================================


def _resolve_datasets_to_targets(required_datasets: tuple[str, ...]) -> frozenset[str]:
    """Map required dataset table_keys to target names.

    Parameters
    ----------
    required_datasets
        Dataset table_keys from Operation.required_datasets.

    Returns
    -------
    frozenset[str]
        Target names that produce the required datasets.
    """
    table_index = _get_table_index()
    targets: set[str] = set()

    for table_key in required_datasets:
        target_name = table_index.get(table_key)
        if target_name is not None:
            targets.add(target_name)
        else:
            log.debug("operations: no target for table %s", table_key)

    return frozenset(targets)


def _resolve_graphs_to_targets(required_graphs: tuple[str, ...]) -> frozenset[str]:
    """Map required graph runtime names to target names.

    Parameters
    ----------
    required_graphs
        Graph runtime names from Operation.required_graphs.

    Returns
    -------
    frozenset[str]
        Target names that produce the required graph runtimes.
    """
    graph_index = _get_graph_index()
    targets: set[str] = set()

    for graph_name in required_graphs:
        target_name = graph_index.get(graph_name)
        if target_name is not None:
            targets.add(target_name)
        else:
            log.debug("operations: no target for graph %s", graph_name)

    return frozenset(targets)


def resolve_targets_for_operation(op: Operation) -> OperationTargets:
    """Resolve build targets for an operation.

    Parameters
    ----------
    op
        Operation to resolve targets for.

    Returns
    -------
    OperationTargets
        Build targets required for the operation.
    """
    graph_targets = _resolve_graphs_to_targets(op.required_graphs)
    data_targets = _resolve_datasets_to_targets(op.required_datasets)
    required_targets = graph_targets | data_targets

    return OperationTargets(
        operation_id=op.id,
        required_targets=required_targets,
        graph_targets=graph_targets,
        data_targets=data_targets,
    )


@lru_cache(maxsize=64)
def get_targets_for_operation(op_id: str) -> OperationTargets:
    """Get build targets required for an operation.

    This function maps the operation's declared requirements to the build
    system's targets, enabling the build system to compute exactly what's
    needed for an operation.

    Parameters
    ----------
    op_id
        Operation identifier (e.g., "function.summary", "datasets.list").

    Returns
    -------
    OperationTargets
        Build targets required for the operation.
        Returns empty targets if operation is unknown.

    Examples
    --------
    >>> targets = get_targets_for_operation("function.summary")
    >>> "call_graph" in targets.graph_targets
    True
    >>> "function_metrics" in targets.data_targets
    False
    """
    op = _LazyCatalog.get_operation(op_id)
    if op is None:
        log.warning("operations: unknown operation %s", op_id)
        return OperationTargets(
            operation_id=op_id,
            required_targets=frozenset(),
            graph_targets=frozenset(),
            data_targets=frozenset(),
        )

    return resolve_targets_for_operation(op)


def get_all_operation_targets() -> dict[str, OperationTargets]:
    """Get targets for all registered operations.

    Returns
    -------
    dict[str, OperationTargets]
        Mapping from operation ID to its required targets.
    """
    result: dict[str, OperationTargets] = {}
    for op in _LazyCatalog.iter_operations():
        result[op.id] = resolve_targets_for_operation(op)
    return result


__all__ = [
    "OperationTargets",
    "get_all_operation_targets",
    "get_targets_for_operation",
    "resolve_targets_for_operation",
]
