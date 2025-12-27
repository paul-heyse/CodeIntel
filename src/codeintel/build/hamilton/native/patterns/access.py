"""Data access specifications for loader patterns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ibis.expr.types as ir

from codeintel.build.hamilton.native.patterns.loaders import load_query, load_table

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class DataAccessSpec:
    """Typed specification for a loader node."""

    domain: str
    target: str
    table_key: str
    sql: str | None = None
    node_name: str | None = None


def load_table_spec(spec: DataAccessSpec) -> Callable[..., ir.Table]:
    """Build a loader node from a table access spec.

    Returns
    -------
    Callable[..., ir.Table]
        Loader function that returns an Ibis table expression.

    Raises
    ------
    ValueError
        If the spec includes a SQL query.
    """
    if spec.sql:
        msg = "load_table_spec requires DataAccessSpec.sql to be None"
        raise ValueError(msg)
    return load_table(
        domain=spec.domain,
        target=spec.target,
        table_key=spec.table_key,
        node_name=spec.node_name,
    )


def load_query_spec(spec: DataAccessSpec) -> Callable[..., ir.Table]:
    """Build a loader node from a query access spec.

    Returns
    -------
    Callable[..., ir.Table]
        Loader function that returns an Ibis table expression.

    Raises
    ------
    ValueError
        If the spec does not include a SQL query.
    """
    if not spec.sql:
        msg = "load_query_spec requires DataAccessSpec.sql to be set"
        raise ValueError(msg)
    return load_query(
        domain=spec.domain,
        target=spec.target,
        table_key=spec.table_key,
        sql=spec.sql,
        node_name=spec.node_name,
    )


def load_access(spec: DataAccessSpec) -> Callable[..., ir.Table]:
    """Build a loader node from a table/query access spec.

    Returns
    -------
    Callable[..., ir.Table]
        Loader function that returns an Ibis table expression.
    """
    if spec.sql:
        return load_query_spec(spec)
    return load_table_spec(spec)


__all__ = [
    "DataAccessSpec",
    "load_access",
    "load_query_spec",
    "load_table_spec",
]
