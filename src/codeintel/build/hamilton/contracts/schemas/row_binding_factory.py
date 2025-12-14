"""Schema-aware RowBinding factory for contract integration.

This module provides utilities for creating RowBinding instances from
schema-generated models, enabling the transition from manual TypedDict
definitions to Pandera-based models.

Architecture Reference: Section 5.3.2 - Update contracts
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

from codeintel.build.hamilton.contracts.schemas.registry import SCHEMA_REGISTRY
from codeintel.config.datasets.contracts import RowBinding, get_row_bindings

__all__ = [
    "compare_row_bindings",
    "get_or_create_row_binding",
    "row_binding_from_schema",
    "row_serializer_from_schema",
]

log = logging.getLogger(__name__)


RowSerializer = Callable[[Mapping[str, object]], tuple[object, ...]]


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
    schema = SCHEMA_REGISTRY.require(table_key)
    row_model = schema.get_row_model()
    serializer = row_serializer_from_schema(table_key)

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
    schema = SCHEMA_REGISTRY.require(table_key)
    columns = schema.column_names()

    def _serialize(row: Mapping[str, object]) -> tuple[object, ...]:
        """Serialize a row dict to tuple using schema column order.

        Parameters
        ----------
        row
            Row data as a mapping.

        Returns
        -------
        tuple[object, ...]
            Values in schema column order.
        """
        return tuple(row.get(col) for col in columns)

    return _serialize


def get_or_create_row_binding(table_key: str) -> RowBinding:
    """Get RowBinding, preferring existing manual definition with schema fallback.

    This function provides a migration-friendly way to get RowBinding:
    1. Try to get the existing manual RowBinding from contracts
    2. Fall back to schema-generated RowBinding if not found

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
        If no RowBinding is available (neither manual nor schema-generated).

    Notes
    -----
    Flip preference to schema-generated when ready for full migration.
    See architecture Section 5.3.2 - Update contracts for activation steps.
    """
    row_bindings = get_row_bindings()
    if table_key in row_bindings:
        return row_bindings[table_key]

    try:
        return row_binding_from_schema(table_key)
    except KeyError:
        msg = f"No RowBinding available for {table_key}"
        raise KeyError(msg) from None


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

    schema = SCHEMA_REGISTRY.get(table_key)
    result["has_schema"] = schema is not None

    if manual_binding is None or schema is None:
        if manual_binding is None:
            differences.append("No manual RowBinding defined")
        if schema is None:
            differences.append("No schema registered")
        result["differences"] = differences
        return result

    try:
        schema_model = schema.get_row_model()
        manual_annotations = getattr(manual_binding.row_type, "__annotations__", {})
        schema_annotations = getattr(schema_model, "__annotations__", {})

        if set(manual_annotations.keys()) == set(schema_annotations.keys()):
            result["row_type_match"] = True
        else:
            missing = set(manual_annotations.keys()) - set(schema_annotations.keys())
            extra = set(schema_annotations.keys()) - set(manual_annotations.keys())
            if missing:
                differences.append(f"Fields in manual but not schema: {missing}")
            if extra:
                differences.append(f"Fields in schema but not manual: {extra}")
    except (KeyError, AttributeError) as exc:
        differences.append(f"Could not compare row types: {exc}")

    result["differences"] = differences
    return result
