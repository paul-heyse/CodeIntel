"""Arrow schema metadata helpers."""

from __future__ import annotations

import pyarrow as pa

__all__ = [
    "arrow_schema_digest",
    "arrow_schema_hash",
]


def arrow_schema_hash(schema: pa.Schema) -> str | None:
    """Return the CodeIntel schema hash embedded in Arrow metadata.

    Parameters
    ----------
    schema
        PyArrow schema to inspect.

    Returns
    -------
    str | None
        Schema hash when present, otherwise None.
    """
    return _schema_metadata_value(schema, "codeintel.schema_hash")


def arrow_schema_digest(schema: pa.Schema) -> str | None:
    """Return the schema digest embedded in Arrow metadata.

    Parameters
    ----------
    schema
        PyArrow schema to inspect.

    Returns
    -------
    str | None
        Schema digest when present, otherwise None.
    """
    return _schema_metadata_value(schema, "codeintel.schema_digest")


def _schema_metadata_value(schema: pa.Schema, key: str) -> str | None:
    metadata = schema.metadata
    if not metadata:
        return None
    raw = metadata.get(key.encode("utf-8"))
    if raw is None:
        return None
    return raw.decode("utf-8")
