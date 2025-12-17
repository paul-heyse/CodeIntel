"""Reusable subDAG pipeline for Ibis-to-DuckDB targets.

This module defines a small, reusable Hamilton subDAG that turns an Ibis table
expression into a persisted DuckDB table write (via ``DuckDBIbisTableSaver``)
and then into a ``TargetRunRecord``.

It is intended to be used with Hamilton's ``@subdag`` decorator to reduce
boilerplate in native target modules.

Notes
-----
Hamilton namespaces nodes created by ``@subdag`` using dotted names
(``<namespace>.<node_name>``). This is acceptable for internal pipeline nodes;
the public target node (e.g., ``t__risk_factors``) remains a stable identifier.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hamilton.function_modifiers import source, tag
from hamilton.function_modifiers.adapters import SaveToDecorator

from codeintel.build.hamilton.materializers import DuckDBIbisTableSaver
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materialization,
)

if TYPE_CHECKING:
    import ibis.expr.types as ir

    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.targets import TargetGraph
    from codeintel.hamilton.records import TargetRunRecord


@SaveToDecorator(
    [DuckDBIbisTableSaver],
    output_name_="materialization",
    env=source("env"),
    graph=source("graph"),
    target_name=source("target_name"),
    table_key=source("table_key"),
)
@tag(node_type="compute")
def ibis_expr_to_save(expr: ir.Table | None) -> ir.Table | None:
    """Return the Ibis expression to be materialized.

    Parameters
    ----------
    expr
        Ibis table expression to persist, or None to indicate no output.

    Returns
    -------
    ir.Table | None
        The same Ibis table expression.
    """
    return expr


@tag(node_type="compute")
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
    "ibis_expr_to_save",
    "record",
]
