"""Operation contracts registry documenting data sources for serving operations.

This module provides a declarative registry of all serving operations and their
data source contracts. Each operation declares:
- Where its data comes from (view, table, graph_engine, computed)
- What the source name is (e.g., "docs.v_function_architecture")
- Whether it supports pagination

The canonical source of truth is now `codeintel.serving.operations.catalog`.
This module provides a backward-compatible view focused on data source contracts.

This serves two purposes:
1. Documentation: Makes the data flow explicit and traceable
2. Validation: Tests can verify that implementations follow their contracts
"""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.serving.operations import (
    DataSourceType,
    Operation,
    iter_operations,
)

# Re-export DataSourceType for backward compatibility
__all__ = [
    "OPERATION_CONTRACTS",
    "DataSourceType",
    "OperationContract",
    "contracts_for_source",
    "contracts_using_view",
    "get_contract",
]


@dataclass(frozen=True)
class OperationContract:
    """Backend-centric view of an operation focused on data sources.

    This is a compatibility wrapper around the canonical Operation model,
    exposing only the fields relevant to data source contracts.

    Parameters
    ----------
    name
        Unique operation identifier (e.g., "function.architecture").
    data_source
        Type of data source (view, table, graph_engine, computed).
    source_name
        Specific source name (e.g., "docs.v_function_architecture").
    supports_pagination
        Whether the operation supports limit/offset pagination.
    description
        Human-readable description of what the operation does.
    repository_method
        Name of the repository method that fetches the data (if applicable).
    """

    name: str
    data_source: DataSourceType
    source_name: str
    supports_pagination: bool = False
    description: str = ""
    repository_method: str | None = None

    @classmethod
    def from_operation(cls, op: Operation) -> OperationContract:
        """Create an OperationContract from a canonical Operation.

        Parameters
        ----------
        op
            The canonical Operation to derive from.

        Returns
        -------
        OperationContract
            A contract view of the operation.
        """
        return cls(
            name=op.id,
            data_source=op.data_source,
            source_name=op.source_name or "",
            supports_pagination=op.supports_pagination,
            description=op.description or op.summary,
            repository_method=op.repository_method,
        )


def _build_operation_contracts() -> dict[str, OperationContract]:
    """Build operation contracts from the canonical catalog.

    Returns
    -------
    dict[str, OperationContract]
        Mapping from operation ID to OperationContract.
    """
    return {op.id: OperationContract.from_operation(op) for op in iter_operations()}


# =============================================================================
# Registry (derived from canonical catalog)
# =============================================================================

OPERATION_CONTRACTS: dict[str, OperationContract] = _build_operation_contracts()


def get_contract(operation_name: str) -> OperationContract | None:
    """Look up an operation contract by name.

    Parameters
    ----------
    operation_name
        The operation identifier (e.g., "function.architecture").

    Returns
    -------
    OperationContract | None
        The contract if found, otherwise None.
    """
    return OPERATION_CONTRACTS.get(operation_name)


def contracts_for_source(source_type: DataSourceType) -> list[OperationContract]:
    """Return all operation contracts that use a specific data source type.

    Parameters
    ----------
    source_type
        The type of data source to filter by.

    Returns
    -------
    list[OperationContract]
        Contracts matching the source type.
    """
    return [c for c in OPERATION_CONTRACTS.values() if c.data_source == source_type]


def contracts_using_view(view_name: str) -> list[OperationContract]:
    """Return all operation contracts that query a specific view.

    Parameters
    ----------
    view_name
        The view name to filter by (e.g., "docs.v_function_architecture").

    Returns
    -------
    list[OperationContract]
        Contracts that query the specified view.
    """
    return [
        c
        for c in OPERATION_CONTRACTS.values()
        if c.data_source == DataSourceType.VIEW and c.source_name == view_name
    ]
