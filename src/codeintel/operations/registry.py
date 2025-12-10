"""Operation registry for discovery and lookup.

The registry stores all registered operations and provides methods
for discovery, lookup, and filtering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.operations.base import OperationSpec


@dataclass
class OperationRegistry:
    """Registry for all operations.

    Populated at import time via @operation decorator.
    Provides lookup and discovery for adapters.

    Parameters
    ----------
    _operations
        Internal storage for registered operations.

    Example
    -------
    >>> registry = OperationRegistry()
    >>> spec = registry.get("jobs.list")
    >>> if spec:
    ...     print(spec.description)
    """

    _operations: dict[str, OperationSpec] = field(default_factory=dict)

    def register(self, spec: OperationSpec) -> OperationSpec:
        """Register an operation.

        Parameters
        ----------
        spec
            The operation specification to register.

        Returns
        -------
        OperationSpec
            The registered spec.

        Raises
        ------
        ValueError
            If an operation with the same ID is already registered.
        """
        if spec.operation_id in self._operations:
            msg = f"Operation already registered: {spec.operation_id}"
            raise ValueError(msg)
        self._operations[spec.operation_id] = spec
        return spec

    def get(self, operation_id: str) -> OperationSpec | None:
        """Get operation by ID.

        Parameters
        ----------
        operation_id
            The unique operation identifier.

        Returns
        -------
        OperationSpec | None
            The operation spec if found, None otherwise.
        """
        return self._operations.get(operation_id)

    def list_operations(
        self,
        *,
        group: str | None = None,
        capabilities: frozenset[str] | None = None,
        include_hidden: bool = False,
        tags: tuple[str, ...] | None = None,
    ) -> list[OperationSpec]:
        """List operations with optional filters.

        Parameters
        ----------
        group
            Filter by operation group (e.g., "jobs", "datasets").
        capabilities
            Filter to operations whose capabilities are subset of this set.
        include_hidden
            Whether to include hidden operations.
        tags
            Filter to operations with any of these tags.

        Returns
        -------
        list[OperationSpec]
            Filtered and sorted list of operations.
        """
        ops = list(self._operations.values())

        if group is not None:
            ops = [op for op in ops if op.group == group]

        if capabilities is not None:
            # Filter to operations whose capabilities are subset
            ops = [op for op in ops if op.capabilities <= capabilities]

        if not include_hidden:
            ops = [op for op in ops if not op.hidden]

        if tags is not None:
            ops = [op for op in ops if any(t in op.tags for t in tags)]

        return sorted(ops, key=lambda op: op.operation_id)

    def list_groups(self) -> list[str]:
        """List all operation groups.

        Returns
        -------
        list[str]
            Sorted list of unique group names.
        """
        return sorted({op.group for op in self._operations.values()})

    def list_capabilities(self) -> list[str]:
        """List all required capabilities across operations.

        Returns
        -------
        list[str]
            Sorted list of unique capabilities.
        """
        caps: set[str] = set()
        for op in self._operations.values():
            caps.update(op.capabilities)
        return sorted(caps)

    def clear(self) -> None:
        """Clear all registered operations.

        Primarily for testing purposes.
        """
        self._operations.clear()


# Module-level default registry
_DEFAULT_REGISTRY = OperationRegistry()


def get_default_registry() -> OperationRegistry:
    """Get the default registry.

    Returns
    -------
    OperationRegistry
        The default singleton registry.
    """
    return _DEFAULT_REGISTRY


def create_isolated_registry() -> OperationRegistry:
    """Create an isolated registry for testing.

    Returns
    -------
    OperationRegistry
        A fresh registry instance.
    """
    return OperationRegistry()


__all__ = [
    "OperationRegistry",
    "create_isolated_registry",
    "get_default_registry",
]
