"""Introspection utilities for CLI operations.

Provide runtime discovery of operations, their metadata, and examples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from codeintel.cli.execution.registry import get_registry

if TYPE_CHECKING:
    from codeintel.cli.execution.registry import OperationSpec


@dataclass(frozen=True)
class OperationInfo:
    """Detailed information about an operation.

    Parameters
    ----------
    operation_id
        Unique identifier.
    name
        Display name.
    group
        Operation group (e.g., "jobs", "build").
    description
        Human-readable description.
    require_runtime
        Whether operation needs ResolvedRuntime.
    require_gateway
        Whether operation needs StorageGateway.
    require_graph_runtime
        Whether operation needs GraphRuntime.
    tags
        Optional tags for categorization.
    hidden
        Whether operation is hidden from help.
    """

    operation_id: str
    name: str
    group: str
    description: str
    require_runtime: bool
    require_gateway: bool
    require_graph_runtime: bool
    tags: tuple[str, ...]
    hidden: bool

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return asdict(self)


def get_operation_info(operation_id: str) -> OperationInfo | None:
    """Get detailed information about an operation.

    Parameters
    ----------
    operation_id
        Operation identifier.

    Returns
    -------
    OperationInfo | None
        Operation information or None if not found.
    """
    registry = get_registry()
    spec = registry.get(operation_id)

    if spec is None:
        return None

    return _spec_to_info(spec)


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


def search_operations(query: str, *, include_hidden: bool = False) -> list[OperationInfo]:
    """Search operations by ID, name, or description.

    Parameters
    ----------
    query
        Search query.
    include_hidden
        Whether to include hidden operations.

    Returns
    -------
    list[OperationInfo]
        Matching operations.
    """
    registry = get_registry()
    query_lower = query.lower()
    results = []

    for spec in registry.list_operations(include_hidden=include_hidden):
        match_id = query_lower in spec.operation_id.lower()
        match_name = query_lower in spec.name.lower()
        match_desc = query_lower in spec.description.lower()
        if match_id or match_name or match_desc:
            results.append(_spec_to_info(spec))

    return results


def list_all_operations(*, include_hidden: bool = False) -> list[OperationInfo]:
    """List all registered operations.

    Parameters
    ----------
    include_hidden
        Whether to include hidden operations.

    Returns
    -------
    list[OperationInfo]
        All operation info.
    """
    registry = get_registry()
    return [_spec_to_info(spec) for spec in registry.list_operations(include_hidden=include_hidden)]


def _spec_to_info(spec: OperationSpec) -> OperationInfo:
    """Convert OperationSpec to OperationInfo.

    Parameters
    ----------
    spec
        Operation specification.

    Returns
    -------
    OperationInfo
        Operation information.
    """
    return OperationInfo(
        operation_id=spec.operation_id,
        name=spec.name,
        group=spec.group,
        description=spec.description,
        require_runtime=spec.require_runtime,
        require_gateway=spec.require_gateway,
        require_graph_runtime=spec.require_graph_runtime,
        tags=spec.tags,
        hidden=spec.hidden,
    )


__all__ = [
    "OperationInfo",
    "get_operation_info",
    "list_all_operations",
    "list_operations_by_group",
    "search_operations",
]
