"""Reusable subDAG pipeline for row-oriented DuckDB targets.

This module defines a small, reusable Hamilton subDAG that turns a sequence of
row tuples into a persisted DuckDB table write (via ``DuckDBRowsSaver``)
and then into a ``TargetRunRecord``.

It is intended to be used with Hamilton's ``@subdag`` decorator to reduce
boilerplate in native target modules that produce row-oriented data rather
than Ibis expressions.

Additionally, this module provides helper utilities for converting mapping rows
(dicts or TypedDicts) to tuples in a specified column order:

- ``row_to_tuple``: Convert a single mapping row to a tuple
- ``rows_to_tuples``: Convert a sequence of mapping rows to a tuple of tuples

Notes
-----
Hamilton namespaces nodes created by ``@subdag`` using dotted names
(``<namespace>.<node_name>``). This is acceptable for internal pipeline nodes;
the public target node (e.g., ``t__coverage_test_edges``) remains a stable identifier.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hamilton.function_modifiers import source, tag
from hamilton.function_modifiers.adapters import SaveToDecorator

from codeintel.build.hamilton.materializers import DuckDBRowsSaver
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materialization,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.targets import TargetGraph
    from codeintel.hamilton.records import TargetRunRecord


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_="materialization",
    env=source("env"),
    graph=source("graph"),
    target_name=source("target_name"),
    table_key=source("table_key"),
    columns=source("columns"),
)
@tag(node_type="compute")
def rows_to_save(
    rows: Sequence[tuple[Any, ...]] | None,
) -> Sequence[tuple[Any, ...]] | None:
    """Return the row sequence to be materialized.

    Parameters
    ----------
    rows
        Sequence of row tuples to persist, or None to indicate no output.

    Returns
    -------
    Sequence[tuple[Any, ...]] | None
        The same row sequence.
    """
    return rows


@tag(node_type="materialize")
def record(
    env: BuildEnv,
    graph: TargetGraph,
    target_name: str,
    table_key: str,
    materialization: dict[str, Any],
) -> TargetRunRecord:
    """Convert a saver metadata dict into a TargetRunRecord.

    Parameters
    ----------
    env
        Build environment for manifest persistence and expected output refs.
    graph
        Target graph used to resolve the OutputTarget contract.
    target_name
        Target name for which the record is being produced.
    table_key
        Table key expected to be materialized for this target.
    materialization
        Materialization metadata dict returned by the Hamilton saver node.

    Returns
    -------
    TargetRunRecord
        Record describing succeeded/skipped/failed completion.
    """
    return record_from_duckdb_materialization(
        env=env,
        graph=graph,
        target_name=target_name,
        expected_table_key=table_key,
        materialization=materialization,
    )


# ============================================================================
# Row Conversion Helpers
# ============================================================================


def row_to_tuple(row: Mapping[str, object], columns: tuple[str, ...]) -> tuple[object, ...]:
    """Convert a mapping row to a tuple in column order.

    This utility is useful when converting TypedDict or dict rows to the tuple
    format expected by ``DuckDBRowsSaver``.

    Parameters
    ----------
    row
        Row mapping from column name to value (e.g., a dict or TypedDict).
    columns
        Column names in the desired order for the output tuple.

    Returns
    -------
    tuple[object, ...]
        Values extracted from the row in the specified column order.
        Missing columns will produce ``None`` values.

    Examples
    --------
    >>> cols = ("id", "name", "value")
    >>> row = {"id": 1, "name": "test", "value": 42}
    >>> row_to_tuple(row, cols)
    (1, 'test', 42)

    >>> # Missing columns produce None
    >>> row_to_tuple({"id": 1}, ("id", "name"))
    (1, None)
    """
    return tuple(row.get(col) for col in columns)


def rows_to_tuples(
    rows: Sequence[Mapping[str, object]],
    columns: tuple[str, ...],
) -> tuple[tuple[object, ...], ...]:
    """Convert a sequence of mapping rows to a tuple of tuples in column order.

    This utility is useful when converting a list of TypedDict or dict rows
    to the format expected by ``DuckDBRowsSaver``.

    Parameters
    ----------
    rows
        Sequence of row mappings from column name to value.
    columns
        Column names in the desired order for the output tuples.

    Returns
    -------
    tuple[tuple[object, ...], ...]
        Tuple of row tuples with values in the specified column order.
        Missing columns will produce ``None`` values.

    Examples
    --------
    >>> cols = ("id", "name")
    >>> rows = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
    >>> rows_to_tuples(rows, cols)
    ((1, 'a'), (2, 'b'))
    """
    return tuple(row_to_tuple(row, columns) for row in rows)


__all__ = [
    "record",
    "row_to_tuple",
    "rows_to_save",
    "rows_to_tuples",
]
