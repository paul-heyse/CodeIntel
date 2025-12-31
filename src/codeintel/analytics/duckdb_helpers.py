"""DuckDB helpers for analytics aggregation."""

from __future__ import annotations

from collections.abc import Sequence

from codeintel.storage.duckdb_types import DuckDBRelation


def aggregate_relation(
    relation: DuckDBRelation,
    *,
    aggs: Sequence[str],
    group_by: str,
) -> DuckDBRelation:
    """Aggregate a DuckDB relation with typed expressions.

    Parameters
    ----------
    relation
        DuckDB relation to aggregate.
    aggs
        Aggregate expressions or SQL fragments to apply.
    group_by
        Comma-delimited grouping clause to apply.

    Returns
    -------
    DuckDBRelation
        Aggregated relation with the requested grouping.

    Raises
    ------
    ValueError
        If no aggregate expressions are provided.
    """
    if not aggs:
        msg = "aggregate_relation requires at least one aggregate expression"
        raise ValueError(msg)
    return relation.aggregate(", ".join(aggs), group_by)
