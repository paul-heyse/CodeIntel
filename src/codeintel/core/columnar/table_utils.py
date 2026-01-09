"""Arrow table helpers for contract-aligned schemas."""

from __future__ import annotations

import pyarrow as pa

from codeintel.core.columnar.conversion import empty_table_from_schema
from codeintel.core.schemas.arrow_gen import (
    ArrowSchemaMetadata,
    ExtrasPolicy,
    arrow_contract_for_table_schema,
)
from codeintel.core.schemas.service import get_schema_service


def arrow_schema_for_table(
    table_key: str,
    *,
    extras_policy: ExtrasPolicy | None,
) -> pa.Schema:
    """Return the Arrow schema for a table key.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    extras_policy
        Optional extras policy to apply when building the contract schema.

    Returns
    -------
    pyarrow.Schema
        Arrow schema for the table contract.
    """
    schema_service = get_schema_service()
    if extras_policy is None:
        arrow_schema = schema_service.get_arrow_schema(table_key)
        if arrow_schema is not None:
            return arrow_schema
    table_schema = schema_service.require_table_schema(table_key)
    metadata = None if extras_policy is None else ArrowSchemaMetadata(extras_policy=extras_policy)
    return arrow_contract_for_table_schema(table_schema=table_schema, metadata=metadata)


def empty_table_for_table(
    table_key: str,
    *,
    extras_policy: ExtrasPolicy | None = None,
) -> pa.Table:
    """Return an empty Arrow table using the table schema.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    extras_policy
        Optional extras policy to apply when building the contract schema.

    Returns
    -------
    pyarrow.Table
        Empty table configured with the table schema.
    """
    arrow_schema = arrow_schema_for_table(table_key, extras_policy=extras_policy)
    return empty_table_from_schema(arrow_schema)


__all__ = ["arrow_schema_for_table", "empty_table_for_table"]
