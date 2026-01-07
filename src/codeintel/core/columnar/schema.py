"""Arrow schema utilities for columnar workflows."""

from __future__ import annotations

from collections.abc import Sequence

import pyarrow as pa

from codeintel.core.columnar.schema_ops import (
    DEFAULT_SCHEMA_PROMOTE_OPTIONS,
    SchemaPromoteOptions,
    unify_schemas,
)


def unify_schema_for_batches(
    batches: Sequence[pa.RecordBatch],
    *,
    base_schema: pa.Schema | None = None,
    promote_options: SchemaPromoteOptions = DEFAULT_SCHEMA_PROMOTE_OPTIONS,
) -> pa.Schema:
    """Return a unified Arrow schema for a sequence of record batches.

    Parameters
    ----------
    batches
        Record batches to unify.
    base_schema
        Optional base schema whose metadata should be preserved.
    promote_options
        Schema promotion behavior to use when unifying schemas.

    Returns
    -------
    pyarrow.Schema
        Unified schema covering all batches.
    """
    if not batches:
        return base_schema or pa.schema([])
    schemas = [batch.schema for batch in batches]
    if base_schema is not None:
        schemas.append(base_schema)
    unified = unify_schemas(schemas, promote_options=promote_options)
    if base_schema is not None and base_schema.metadata:
        unified = unified.with_metadata(base_schema.metadata)
    return unified


__all__ = [
    "DEFAULT_SCHEMA_PROMOTE_OPTIONS",
    "SchemaPromoteOptions",
    "unify_schema_for_batches",
]
