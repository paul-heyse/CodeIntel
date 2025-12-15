"""Schema-aware RowBinding factory for contract integration.

This module provides utilities for creating RowBinding instances from
schema-generated models. Schema-generated bindings are the canonical
source of truth for row bindings.

Use ``get_or_create_row_binding()`` for the canonical way to obtain a
row binding, or ``get_row_binding()`` from
``codeintel.build.schemas.row_registry`` for the preferred API.

Architecture Reference: Section 5.3.2 - Update contracts
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from codeintel.build.schemas.provider_declared import declared_schema_provider
from codeintel.build.schemas.row_registry import get_row_binding
from codeintel.core.schemas.contract_primitives import RowBinding
from codeintel.core.schemas.row_models import (
    row_model_for_table_schema,
    row_serializer_for_table_schema,
)

if TYPE_CHECKING:
    from codeintel.core.schemas.row_models import GeneratedRowBinding

__all__ = [
    "generated_to_legacy_binding",
    "get_or_create_row_binding",
    "row_binding_from_schema",
    "row_serializer_from_schema",
]

log = logging.getLogger(__name__)


RowSerializer = Callable[[Mapping[str, object]], tuple[object, ...]]


def generated_to_legacy_binding(generated: GeneratedRowBinding) -> RowBinding:
    """Convert a GeneratedRowBinding to a legacy RowBinding.

    This adapter allows schema-generated bindings to be used in places
    that still expect the legacy RowBinding dataclass.

    Parameters
    ----------
    generated
        Schema-generated binding with provenance metadata.

    Returns
    -------
    RowBinding
        Legacy binding compatible with existing consumers.
    """
    return RowBinding(
        row_type=generated.row_model,
        to_tuple=generated.serializer,
    )


def row_binding_from_schema(table_key: str) -> RowBinding:
    """Create RowBinding from schema-generated models.

    This function generates a RowBinding by:
    1. Getting the row model from the schema registry
    2. Creating a serializer function from the Pandera schema

    Parameters
    ----------
    table_key
        Fully qualified table name (e.g., "analytics.function_metrics").

    Returns
    -------
    RowBinding
        A RowBinding with schema-generated row_type and to_tuple.

    Notes
    -----
    Wire this to replace manual RowBinding definitions in contracts.py.
    See architecture Section 5.3.2 - Update contracts for activation steps.
    """
    provider = declared_schema_provider()
    table_schema = provider.require_table_schema(table_key)
    row_model = row_model_for_table_schema(table_schema=table_schema)
    serializer = row_serializer_for_table_schema(table_schema=table_schema)

    return RowBinding(
        row_type=row_model,
        to_tuple=serializer,
    )


def row_serializer_from_schema(table_key: str) -> RowSerializer:
    """Create a row serializer function from schema column order.

    The serializer converts a row dict to a tuple using the column
    order defined in the Pandera schema, which should match the
    DuckDB table column order.

    Parameters
    ----------
    table_key
        Fully qualified table name.

    Returns
    -------
    Callable[[Mapping[str, object]], tuple[object, ...]]
        Function that serializes row dicts to tuples.
    """
    provider = declared_schema_provider()
    table_schema = provider.require_table_schema(table_key)
    return row_serializer_for_table_schema(table_schema=table_schema)


def get_or_create_row_binding(table_key: str) -> RowBinding:
    """Get RowBinding from schema-generated bindings.

    Parameters
    ----------
    table_key
        Fully qualified table name.

    Returns
    -------
    RowBinding
        The RowBinding for this table.

    Raises
    ------
    KeyError
        If no schema-generated RowBinding is available for the table.
    """
    try:
        generated = get_row_binding(table_key)
    except KeyError:
        msg = f"No schema-generated RowBinding available for {table_key}"
        raise KeyError(msg) from None
    return generated_to_legacy_binding(generated)
