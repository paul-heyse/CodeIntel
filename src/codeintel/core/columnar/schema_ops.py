"""Schema helpers for Arrow table alignment and concatenation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import pyarrow as pa

from codeintel.core.columnar.nested_ops import is_allowed_promotion

SchemaPromoteOptions = Literal["default", "permissive"]
DEFAULT_SCHEMA_PROMOTE_OPTIONS: SchemaPromoteOptions = "permissive"


def unify_schemas(
    schemas: Sequence[pa.Schema],
    *,
    promote_options: SchemaPromoteOptions = DEFAULT_SCHEMA_PROMOTE_OPTIONS,
) -> pa.Schema:
    """Return a unified Arrow schema for the provided schemas.

    Parameters
    ----------
    schemas
        Schemas to unify by field name.
    promote_options
        Schema promotion behavior to use when unifying schemas.

    Returns
    -------
    pyarrow.Schema
        Unified schema covering the provided inputs.
    """
    if len(schemas) == 1:
        return schemas[0]
    try:
        return pa.unify_schemas(schemas, promote_options=promote_options)
    except TypeError:
        return pa.unify_schemas(schemas)


def concat_tables_unified(
    tables: Sequence[pa.Table],
    *,
    promote_options: SchemaPromoteOptions = DEFAULT_SCHEMA_PROMOTE_OPTIONS,
) -> pa.Table:
    """Concatenate tables after unifying schemas.

    Parameters
    ----------
    tables
        Tables to concatenate.
    promote_options
        Schema promotion behavior to use when unifying schemas.

    Returns
    -------
    pyarrow.Table
        Concatenated table with unified schema.
    """
    if not tables:
        return pa.table({})
    if len(tables) == 1:
        return tables[0]
    _, aligned = align_tables_to_schema(tables, promote_options=promote_options, safe=False)
    try:
        return pa.concat_tables(aligned, promote=True)
    except (pa.ArrowInvalid, pa.ArrowTypeError):
        return pa.concat_tables(aligned)


def align_tables_to_schema(
    tables: Sequence[pa.Table],
    *,
    schema: pa.Schema | None = None,
    promote_options: SchemaPromoteOptions = DEFAULT_SCHEMA_PROMOTE_OPTIONS,
    safe: bool = False,
) -> tuple[pa.Schema, list[pa.Table]]:
    """Return a target schema and tables aligned to it.

    Parameters
    ----------
    tables
        Tables to align.
    schema
        Optional explicit schema to align to. When omitted, a unified schema is used.
    promote_options
        Schema promotion behavior to use when unifying schemas.
    safe
        Whether to enforce safe Arrow casts during alignment.

    Returns
    -------
    tuple[pyarrow.Schema, list[pyarrow.Table]]
        Target schema and aligned tables.
    """
    if not tables:
        return pa.schema([]), []
    target_schema = schema or unify_schemas(
        [table.schema for table in tables],
        promote_options=promote_options,
    )
    aligned: list[pa.Table] = []
    for table in tables:
        if table.schema == target_schema:
            aligned.append(table)
            continue
        try:
            aligned.append(table.cast(target_schema, safe=safe))
        except (TypeError, ValueError, pa.ArrowInvalid, pa.ArrowNotImplementedError):
            aligned.append(table)
    return target_schema, aligned


def validate_contract_schema_promotions(
    contract_schema: pa.Schema,
    candidate_schema: pa.Schema,
) -> None:
    """Validate that candidate schema types can promote into the contract schema.

    Raises
    ------
    ValueError
        If a field promotion is disallowed.
    """
    for field in contract_schema:
        if field.name not in candidate_schema.names:
            continue
        candidate_field = candidate_schema.field(field.name)
        if is_allowed_promotion(candidate_field.type, field.type):
            continue
        msg = (
            "Disallowed schema promotion for "
            f"{field.name}: {candidate_field.type} -> {field.type}"
        )
        raise ValueError(msg)


__all__ = [
    "DEFAULT_SCHEMA_PROMOTE_OPTIONS",
    "SchemaPromoteOptions",
    "align_tables_to_schema",
    "concat_tables_unified",
    "unify_schemas",
    "validate_contract_schema_promotions",
]
