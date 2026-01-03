"""Introspection utilities for CLI operations.

Provide runtime discovery of operations, their metadata, and examples.
All introspection functions return `OperationSpec` directly from the registry,
providing a single source of truth for operation metadata.
"""

from __future__ import annotations

from codeintel.cli.execution.registry import OperationAlias, OperationSpec, get_registry


def get_operation_info(operation_id: str) -> OperationSpec | None:
    """Get detailed information about an operation.

    Parameters
    ----------
    operation_id
        Operation identifier.

    Returns
    -------
    OperationSpec | None
        Operation specification or None if not found.
    """
    registry = get_registry()
    return registry.get(operation_id)


def list_operations_by_group(*, include_hidden: bool = False) -> dict[str, list[str]]:
    """List operations grouped by their group.

    Parameters
    ----------
    include_hidden
        Whether to include hidden operations.

    Returns
    -------
    dict[str, list[str]]
        Operations grouped by group name.
    """
    registry = get_registry()
    result: dict[str, list[str]] = {}

    for spec in registry.list_operations(include_hidden=include_hidden):
        group = spec.group
        if group not in result:
            result[group] = []
        result[group].append(spec.operation_id)

    return result


def search_operations(query: str, *, include_hidden: bool = False) -> list[OperationSpec]:
    """Search operations by ID, name, or description.

    Parameters
    ----------
    query
        Search query.
    include_hidden
        Whether to include hidden operations.

    Returns
    -------
    list[OperationSpec]
        Matching operations.
    """
    registry = get_registry()
    query_lower = query.lower()
    results: list[OperationSpec] = []

    for spec in registry.list_operations(include_hidden=include_hidden):
        match_id = query_lower in spec.operation_id.lower()
        match_name = query_lower in spec.name.lower()
        match_desc = query_lower in spec.description.lower()
        if match_id or match_name or match_desc:
            results.append(spec)

    return results


def list_all_operations(*, include_hidden: bool = False) -> list[OperationSpec]:
    """List all registered operations.

    Parameters
    ----------
    include_hidden
        Whether to include hidden operations.

    Returns
    -------
    list[OperationSpec]
        All registered operation specifications.
    """
    registry = get_registry()
    return registry.list_operations(include_hidden=include_hidden)


def list_operation_aliases(operation_id: str) -> list[OperationAlias]:
    """List aliases for a canonical operation.

    Parameters
    ----------
    operation_id
        Operation identifier (canonical or alias).

    Returns
    -------
    list[OperationAlias]
        Alias metadata for the canonical operation.
    """
    registry = get_registry()
    spec = registry.get(operation_id)
    if spec is None:
        return []
    return registry.list_aliases(target_id=spec.operation_id)


def list_all_aliases() -> list[OperationAlias]:
    """List all registered operation aliases.

    Returns
    -------
    list[OperationAlias]
        Alias metadata for all registered aliases.
    """
    return get_registry().list_aliases()


__all__ = [
    "OperationAlias",
    "OperationSpec",
    "get_operation_info",
    "list_all_aliases",
    "list_all_operations",
    "list_operation_aliases",
    "list_operations_by_group",
    "search_operations",
]
