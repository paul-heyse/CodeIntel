"""Schema-aware RowBinding factory for contract integration.

This module provides utilities for creating RowBinding instances from
schema-generated models, enabling the transition from manual TypedDict
definitions to Pandera-based models.

As of PR-67, schema-generated bindings are preferred over legacy manual
bindings. Use ``get_or_create_row_binding()`` for the canonical way to
obtain a row binding, or ``get_row_binding()`` from
``codeintel.build.schemas.row_registry`` for the new preferred API.

Architecture Reference: Section 5.3.2 - Update contracts
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from codeintel.build.schemas.provider_declared import declared_schema_provider
from codeintel.build.schemas.row_registry import get_row_binding
from codeintel.config.datasets.contracts import RowBinding, get_row_bindings
from codeintel.core.schemas.row_models import (
    row_model_for_table_schema,
    row_serializer_for_table_schema,
)

if TYPE_CHECKING:
    from codeintel.core.schemas.row_models import GeneratedRowBinding

__all__ = [
    "compare_row_bindings",
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
    """Get RowBinding, preferring schema-generated with legacy fallback.

    This function provides a migration-friendly way to get RowBinding:
    1. Try schema-generated binding via get_row_binding() (preferred)
    2. Fall back to legacy manual RowBinding if schema not found

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
        If no RowBinding is available (neither schema nor manual).

    Notes
    -----
    As of PR-67, schema-generated bindings are preferred over legacy manual
    bindings. This ensures bindings stay in sync with the canonical schema
    definitions.
    """
    # Prefer schema-generated bindings
    try:
        generated = get_row_binding(table_key)
        return generated_to_legacy_binding(generated)
    except KeyError:
        pass

    # Fall back to legacy manual bindings
    row_bindings = get_row_bindings()
    if table_key in row_bindings:
        log.debug("Using legacy manual binding for %s", table_key)
        return row_bindings[table_key]

    msg = f"No RowBinding available for {table_key}"
    raise KeyError(msg)


def compare_row_bindings(table_key: str) -> dict[str, object]:
    """Compare manual and schema-generated RowBindings for validation.

    Parameters
    ----------
    table_key
        Fully qualified table name.

    Returns
    -------
    dict[str, object]
        Comparison result with fields:
        - has_manual: bool
        - has_schema: bool
        - row_type_match: bool
        - column_order_match: bool
        - differences: list[str]
    """
    result: dict[str, object] = {
        "has_manual": False,
        "has_schema": False,
        "row_type_match": False,
        "column_order_match": True,
        "differences": [],
    }
    differences: list[str] = []

    row_bindings = get_row_bindings()
    manual_binding = row_bindings.get(table_key)
    result["has_manual"] = manual_binding is not None

    provider = declared_schema_provider()
    table_schema = provider.get_table_schema(table_key)
    result["has_schema"] = table_schema is not None

    if manual_binding is None or table_schema is None:
        if manual_binding is None:
            differences.append("No manual RowBinding defined")
        if table_schema is None:
            differences.append("No schema available from schema provider")
        result["differences"] = differences
        return result

    try:
        schema_model = row_model_for_table_schema(table_schema=table_schema)
        manual_annotations = getattr(manual_binding.row_type, "__annotations__", {})
        schema_annotations = getattr(schema_model, "__annotations__", {})

        manual_fields = list(manual_annotations.keys())
        schema_fields = list(schema_annotations.keys())

        if manual_fields == schema_fields:
            result["row_type_match"] = True
        elif set(manual_fields) == set(schema_fields):
            result["row_type_match"] = True
            result["column_order_match"] = False
            differences.append("Column order differs between manual and schema binding")
        else:
            missing = set(manual_fields) - set(schema_fields)
            extra = set(schema_fields) - set(manual_fields)
            if missing:
                differences.append(f"Fields in manual but not schema: {missing}")
            if extra:
                differences.append(f"Fields in schema but not manual: {extra}")
    except (AttributeError, TypeError, ValueError) as exc:
        differences.append(f"Could not compare row types: {exc}")

    result["differences"] = differences
    return result
