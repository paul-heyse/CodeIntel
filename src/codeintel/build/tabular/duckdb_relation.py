"""DuckDB relation coercion utilities."""

from __future__ import annotations

import re
import uuid
from typing import Literal, cast, overload

import polars as pl
import pyarrow as pa

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
    return relation.limit(0).arrow().schema


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


@overload
def relation_to_polars(
    relation: TabularInput,
    *,
    lazy: Literal[True] = True,
) -> pl.LazyFrame: ...


@overload
def relation_to_polars(
    relation: TabularInput,
    *,
    lazy: Literal[False],
) -> pl.DataFrame: ...


def relation_to_polars(
    relation: TabularInput,
    *,
    lazy: bool = True,
) -> pl.DataFrame | pl.LazyFrame:
    """Convert a tabular input to a Polars DataFrame/LazyFrame.

    Parameters
    ----------
    relation
        Tabular input to convert.
    lazy
        When True, return a Polars LazyFrame.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        Polars object representing the relation.
    """
    if isinstance(relation, pl.LazyFrame):
        return relation if lazy else relation.collect()
    if isinstance(relation, pl.DataFrame):
        return relation.lazy() if lazy else relation
    if isinstance(relation, pa.Table):
        frame = _polars_from_arrow(relation)
        return frame.lazy() if lazy else frame
    if isinstance(relation, pa.RecordBatchReader):
        reader = cast("pa.RecordBatchReader", relation)
        table = reader.read_all()
        frame = _polars_from_arrow(table)
        return frame.lazy() if lazy else frame
    table = relation.arrow()
    frame = _polars_from_arrow(table)
    return frame.lazy() if lazy else frame


def _polars_from_arrow(table: pa.Table) -> pl.DataFrame:
    frame = pl.from_arrow(table)
    if isinstance(frame, pl.Series):
        return frame.to_frame()
    return frame


__all__ = [
    "coerce_to_relation",
    "register_ephemeral",
    "relation_schema",
    "relation_to_arrow_reader",
    "relation_to_polars",
]
