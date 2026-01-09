"""DuckDB relation coercion utilities."""

from __future__ import annotations

import re
import uuid
from typing import cast

import polars as pl
import pyarrow as pa

from codeintel.build.tabular.conversion import (
    arrow_reader_to_lazyframe,
    table_to_lazyframe,
    tabular_to_arrow_reader,
)
from codeintel.build.tabular.types import TabularInputWithRelation, TabularRelation
from codeintel.core.duckdb_types import DuckDBConnection, DuckDBRelation

_NAME_SANITIZER = re.compile(r"[^0-9A-Za-z_]+")


def _sanitize_name(prefix: str) -> str:
    cleaned = _NAME_SANITIZER.sub("_", prefix.strip())
    return cleaned if cleaned else "tmp"


def register_ephemeral(
    conn: DuckDBConnection,
    obj: TabularInputWithRelation,
    *,
    prefix: str = "tmp",
) -> str:
    """Register a tabular object under a unique ephemeral name.

    Parameters
    ----------
    conn
        DuckDB connection used for registration.
    obj
        Tabular object to register.
    prefix
        Name prefix used for the generated registration name.

    Returns
    -------
    str
        Name registered in DuckDB for the object.
    """
    safe_prefix = _sanitize_name(prefix)
    name = f"{safe_prefix}_{uuid.uuid4().hex}"
    conn.register(name, obj)
    return name


def coerce_to_relation(
    conn: DuckDBConnection,
    obj: TabularInputWithRelation,
    *,
    name_hint: str | None = None,
) -> TabularRelation:
    """Coerce a tabular input to a DuckDB relation.

    Parameters
    ----------
    conn
        DuckDB connection used for registration.
    obj
        Tabular object to coerce.
    name_hint
        Optional prefix for the registered name.

    Returns
    -------
    TabularRelation
        DuckDB relation for the provided object.
    """
    if isinstance(obj, DuckDBRelation):
        return obj
    name = register_ephemeral(conn, obj, prefix=name_hint or "tmp")
    return conn.table(name)


def relation_schema(relation: TabularRelation) -> pa.Schema:
    """Return the Arrow schema for a relation.

    Parameters
    ----------
    relation
        DuckDB relation to inspect.

    Returns
    -------
    pa.Schema
        Arrow schema for the relation.
    """
    return tabular_to_arrow_reader(relation, batch_size=None).schema


def relation_to_arrow_reader(relation: TabularRelation) -> pa.RecordBatchReader:
    """Return a streaming Arrow reader for a relation.

    Parameters
    ----------
    relation
        DuckDB relation to stream.

    Returns
    -------
    pa.RecordBatchReader
        Arrow reader for streaming record batches.
    """
    return tabular_to_arrow_reader(relation, batch_size=None)


def relation_to_polars(relation: TabularInputWithRelation) -> pl.LazyFrame:
    """Convert a tabular input to a Polars LazyFrame.

    Parameters
    ----------
    relation
        Tabular input to convert.

    Returns
    -------
    pl.LazyFrame
        Polars LazyFrame representing the relation.

    Raises
    ------
    TypeError
        If the relation cannot be coerced into a LazyFrame.
    """
    if isinstance(relation, pl.LazyFrame):
        return relation
    if isinstance(relation, pa.Table):
        return table_to_lazyframe(relation)
    if isinstance(relation, pa.RecordBatchReader):
        reader = cast("pa.RecordBatchReader", relation)
        return arrow_reader_to_lazyframe(reader)
    if isinstance(relation, DuckDBRelation):
        reader = tabular_to_arrow_reader(relation, batch_size=None)
        return arrow_reader_to_lazyframe(reader)
    msg = f"Unsupported tabular input type: {type(relation).__name__}"
    raise TypeError(msg)


__all__ = [
    "coerce_to_relation",
    "register_ephemeral",
    "relation_schema",
    "relation_to_arrow_reader",
    "relation_to_polars",
]
