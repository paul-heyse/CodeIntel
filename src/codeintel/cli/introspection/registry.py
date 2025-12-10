"""Registry for operation specifications.

Provide a central registry where operations can be registered with their
specifications, enabling dynamic discovery and execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from codeintel.cli.execution import OperationCategory, OperationSpec

LOG = logging.getLogger(__name__)


@dataclass
class OperationRegistry:
    """Central registry for operation specifications.

    Parameters
    ----------
    operations
        Mapping of operation IDs to specifications.
    """

    operations: dict[str, OperationSpec[Any]] = field(default_factory=dict)

    def register[T](self, spec: OperationSpec[T]) -> OperationSpec[T]:
        """Register an operation specification.

        Parameters
        ----------
        spec
            Operation specification to register.

        Returns
        -------
        OperationSpec[T]
            The registered specification.

        Raises
        ------
        ValueError
            If operation ID is already registered.
        """
        if spec.operation_id in self.operations:
            msg = f"Operation already registered: {spec.operation_id}"
            raise ValueError(msg)

        self.operations[spec.operation_id] = spec
        LOG.debug("Registered operation: %s", spec.operation_id)
        return spec

    def get(self, operation_id: str) -> OperationSpec[Any] | None:
        """Get an operation specification by ID.

        Parameters
        ----------
        operation_id
            Operation identifier.

        Returns
        -------
        OperationSpec[Any] | None
            Specification or None if not found.
        """
        return self.operations.get(operation_id)

    def list_operations(
        self,
        *,
        category: OperationCategory | None = None,
    ) -> list[OperationSpec[Any]]:
        """List registered operations.

        Parameters
        ----------
        category
            Optional category filter.

        Returns
        -------
        list[OperationSpec[Any]]
            Matching operations.
        """
        ops = list(self.operations.values())
        if category is not None:
            ops = [op for op in ops if op.category == category]
        return ops

    def unregister(self, operation_id: str) -> bool:
        """Unregister an operation.

        Parameters
        ----------
        operation_id
            Operation to unregister.

        Returns
        -------
        bool
            True if operation was removed.
        """
        if operation_id in self.operations:
            del self.operations[operation_id]
            return True
        return False

    def clear(self) -> None:
        """Clear all registered operations."""
        self.operations.clear()

    def __len__(self) -> int:
        """Return the number of registered operations.

        Returns
        -------
        int
            Count of registered operations.
        """
        return len(self.operations)

    def __contains__(self, operation_id: str) -> bool:
        """Check if an operation is registered.

        Parameters
        ----------
        operation_id
            Operation identifier to check.

        Returns
        -------
        bool
            True if operation is registered.
        """
        return operation_id in self.operations


# Global registry instance
_REGISTRY = OperationRegistry()


def get_operation_registry() -> OperationRegistry:
    """Get the global operation registry.

    Returns
    -------
    OperationRegistry
        Global registry instance.
    """
    return _REGISTRY


def register_operation[T](spec: OperationSpec[T]) -> OperationSpec[T]:
    """Register an operation with the global registry.

    Parameters
    ----------
    spec
        Operation specification.

    Returns
    -------
    OperationSpec[T]
        Registered specification.
    """
    return _REGISTRY.register(spec)


__all__ = [
    "OperationRegistry",
    "get_operation_registry",
    "register_operation",
]
