"""Schema helpers for Arrow table alignment and concatenation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import pyarrow as pa

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
    unified = unify_schemas([table.schema for table in tables], promote_options=promote_options)
    aligned: list[pa.Table] = []
    for table in tables:
        if table.schema == unified:
            aligned.append(table)
            continue
        try:
            aligned.append(table.cast(unified, safe=False))
        except (TypeError, ValueError, pa.ArrowInvalid, pa.ArrowNotImplementedError):
            aligned.append(table)
    try:
        return pa.concat_tables(aligned, promote=True)
    except (pa.ArrowInvalid, pa.ArrowTypeError):
        return pa.concat_tables(aligned)


__all__ = [
    "DEFAULT_SCHEMA_PROMOTE_OPTIONS",
    "SchemaPromoteOptions",
    "concat_tables_unified",
    "unify_schemas",
]
