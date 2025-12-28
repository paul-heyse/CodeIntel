"""DuckDB relation coercion utilities."""

from __future__ import annotations

import re
import uuid
from typing import cast

import polars as pl
import pyarrow as pa

from codeintel.build.tabular.conversion import arrow_reader_to_lazyframe, table_to_lazyframe
from codeintel.build.tabular.types import TabularInput, TabularRelation
from codeintel.storage.duckdb_types import DuckDBConnection, DuckDBRelation

_NAME_SANITIZER = re.compile(r"[^0-9A-Za-z_]+")


def _sanitize_name(prefix: str) -> str:
    cleaned = _NAME_SANITIZER.sub("_", prefix.strip())
    return cleaned if cleaned else "tmp"


def register_ephemeral(
    conn: DuckDBConnection,
    obj: TabularInput,
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
    obj: TabularInput,
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
    return relation.fetch_arrow_reader().schema


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
    return relation.fetch_arrow_reader()


def relation_to_polars(relation: TabularInput) -> pl.LazyFrame:
    """Convert a tabular input to a Polars LazyFrame.

    Parameters
    ----------
    relation
        Tabular input to convert.

    Returns
    -------
    pl.LazyFrame
        Polars LazyFrame representing the relation.
    """
    if isinstance(relation, pl.LazyFrame):
        return relation
    if isinstance(relation, pa.Table):
        frame = table_to_lazyframe(relation)
        return frame
    if isinstance(relation, pa.RecordBatchReader):
        reader = cast("pa.RecordBatchReader", relation)
        frame = arrow_reader_to_lazyframe(reader)
        return frame
    reader = relation.fetch_arrow_reader()
    frame = arrow_reader_to_lazyframe(reader)
    return frame


__all__ = [
    "coerce_to_relation",
    "register_ephemeral",
    "relation_schema",
    "relation_to_arrow_reader",
    "relation_to_polars",
]
