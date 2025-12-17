"""Reusable subDAG pipeline for row-oriented DuckDB targets.

This module defines a small, reusable Hamilton subDAG that turns a sequence of
row tuples into a persisted DuckDB table write (via ``DuckDBRowsSaver``)
and then into a ``TargetRunRecord``.

It is intended to be used with Hamilton's ``@subdag`` decorator to reduce
boilerplate in native target modules that produce row-oriented data rather
than Ibis expressions.

Notes
-----
Hamilton namespaces nodes created by ``@subdag`` using dotted names
(``<namespace>.<node_name>``). This is acceptable for internal pipeline nodes;
the public target node (e.g., ``t__coverage_test_edges``) remains a stable identifier.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from hamilton.function_modifiers import source, tag
from hamilton.function_modifiers.adapters import SaveToDecorator

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers import DuckDBRowsSaver
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materialization,
)
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


__all__ = [
    "record",
    "rows_to_save",
]
