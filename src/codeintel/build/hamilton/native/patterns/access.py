"""Data access specifications for loader patterns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.hamilton.native.patterns.loaders import load_table
from codeintel.build.tabular.types import InferableTabularInput

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


def load_table_spec(spec: DataAccessSpec) -> Callable[..., InferableTabularInput]:
    """Build a loader node from a table access spec.

    Returns
    -------
    Callable[..., InferableTabularInput]
        Loader function that returns a tabular input.

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


def load_access(spec: DataAccessSpec) -> Callable[..., InferableTabularInput]:
    """Build a loader node from a table/query access spec.

    Returns
    -------
    Callable[..., InferableTabularInput]
        Loader function that returns a tabular input.

    Raises
    ------
    ValueError
        If the spec includes a SQL query.
    """
    if spec.sql:
        msg = "SQL-backed access specs are deprecated; use dataset-backed loaders instead"
        raise ValueError(msg)
    return load_table_spec(spec)


__all__ = [
    "DataAccessSpec",
    "load_access",
    "load_table_spec",
]
